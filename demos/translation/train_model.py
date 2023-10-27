import torch
from torch import nn
from torchtext.vocab import Vocab

from demos.translation.arguments import ModelArguments, TrainingArguments
from demos.translation.data_loader import create_data_loaders
from demos.translation.tokenizer import padding
from model import make_model
from train.data import Batch
from train.label_smoothing import LabelSmoothing
from train.learning_rate import rate
from train.loss import SimpleLossCompute
from train.routine import run_epoch


def translation_model(vocab_src_len, vocab_tgt_len, d_model, N) -> nn.Module:
    return make_model(vocab_src_len, vocab_tgt_len, d_model=d_model, N=N)


def train_worker(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
) -> nn.Module:
    device = torch.device(training_args.device)
    print(f"Training process starting, using {device} device.")

    # This value equals the index of <blank> in `specials`` when doing build_vocab_from_iterator
    pad_idx = vocab_tgt[padding]

    model = translation_model(
        len(vocab_src), len(vocab_tgt), model_args.d_model, model_args.N
    )
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    train_dataloader, valid_dataloader = create_data_loaders(
        device,
        vocab_src,
        vocab_tgt,
        training_args.training_size,
        training_args.validation_size,
        batch_size=training_args.batch_size,
        max_padding=model_args.max_padding,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_args.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_args.d_model, factor=0.1, warmup=training_args.warmup
        ),
    )
    loss_compute = SimpleLossCompute(model.generator, criterion)

    for epoch in range(training_args.num_epochs):
        print("Epoch:", epoch)
        model.train()
        run_epoch(
            model,
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            loss_compute,
            optimizer,
            lr_scheduler,
            mode="train",
            accum_iter=training_args.accum_iter,
        )

        model.eval()
        total_loss, total_token = run_epoch(
            model,
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            loss_compute,
            None,
            None,
            mode="eval",
        )
        print("Validation Average Loss: %.2f" % (total_loss / total_token))

    print("Training process finished.")
    torch.save(model.state_dict(), model_args.model_path)
    return model


def train_distributed_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
) -> nn.Module:
    raise NotImplementedError


def train_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
) -> nn.Module:
    if training_args.distributed:
        print(f"Distributed mode is enabled, training in parallel.")
        model = train_distributed_model(model_args, training_args, vocab_src, vocab_tgt)
    else:
        model = train_worker(model_args, training_args, vocab_src, vocab_tgt)
    return model
