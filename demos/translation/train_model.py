import logging
import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtext.vocab import Vocab

from demos.translation.arguments import ModelArguments, TrainingArguments
from demos.translation.data_loader import create_data_loaders
from demos.translation.vocab import padding
from model import make_model
from train.data import Batch
from train.label_smoothing import LabelSmoothing
from train.learning_rate import rate
from train.loss import SimpleLossCompute
from train.routine import run_epoch
from utils import GeneratorWithLength

logger = logging.getLogger("translationLogger")


def translation_model(vocab_src_len, vocab_tgt_len, d_model, N) -> nn.Module:
    return make_model(vocab_src_len, vocab_tgt_len, d_model=d_model, N=N)


def train_worker(
    gpu,
    gpu_num,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    is_distributed=False,
) -> nn.Module:
    device = torch.device(training_args.device, gpu)
    logger.info(f"Training process starting, using device {device}.")

    model = translation_model(
        len(vocab_src), len(vocab_tgt), model_args.d_model, model_args.N
    ).to(device)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group("nccl", rank=gpu, world_size=gpu_num)
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    # This value equals the index of <blank> in `specials``
    # when doing build_vocab_from_iterator
    pad_idx = vocab_tgt[padding]

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    train_dataloader, valid_dataloader = create_data_loaders(
        device,
        vocab_src,
        vocab_tgt,
        training_args.training_size,
        training_args.validation_size,
        batch_size=training_args.batch_size,
        max_padding=model_args.max_padding,
        is_distributed=is_distributed,
        should_check_tokens=training_args.should_check_tokens,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_args.base_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_args.d_model, factor=0.01, warmup=training_args.warmup
        ),
    )
    loss_compute = SimpleLossCompute(model.generator, criterion)

    for epoch in range(training_args.num_epochs):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        print("Epoch:", epoch)
        model.train()
        run_epoch(
            model,
            GeneratorWithLength(
                (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
                len(train_dataloader),
            ),
            loss_compute,
            optimizer,
            lr_scheduler,
            mode="train",
            accum_iter=training_args.accum_iter,
        )

        model.eval()
        total_loss, total_token = run_epoch(
            model,
            [Batch(b[0], b[1], pad_idx) for b in valid_dataloader],
            loss_compute,
            None,
            None,
            mode="eval",
        )
        logger.info(
            "Validation Average Loss: %.2f" % (total_loss / total_token)
        )
        torch.cuda.empty_cache()

    logger.info(f"Training process on {device} finished.")
    # Save only from the main process.
    if is_main_process:
        torch.save(module.state_dict(), model_args.model_path)
    return model


def train_distributed_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
) -> nn.Module:
    gpu_num = torch.cuda.device_count()
    logger.info(f"Number of GPUs detected: {gpu_num}")

    os.environ["MASTER_ADDR"] = "localhost"
    port = random.randint(10000, 65535)
    os.environ["MASTER_PORT"] = str(port)

    mp.spawn(
        train_worker,
        nprocs=gpu_num,
        args=(gpu_num, model_args, training_args, vocab_src, vocab_tgt, True),
    )


def train_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
) -> nn.Module:
    if training_args.distributed:
        logger.info("Distributed mode is enabled, training in parallel.")
        model = train_distributed_model(
            model_args, training_args, vocab_src, vocab_tgt
        )
    else:
        logger.info("Distributed mode is disabled.")
        model = train_worker(
            0, 1, model_args, training_args, vocab_src, vocab_tgt
        )
    return model
