import torch
from demos.translation.data_loader import create_data_loaders
from demos.translation.tokenizer import padding
from model import make_model
from train.data import Batch
from train.label_smoothing import LabelSmoothing
from train.learning_rate import rate
from train.loss import SimpleLossCompute
from train.routine import run_epoch

def translation_model(vocab_src_len, vocab_tgt_len, d_model, N):
    return make_model(vocab_src_len, vocab_tgt_len, d_model=d_model, N=N)

def train_worker(config):
    print("Training process starting...")
    vocab_src, vocab_tgt = config['vocab_src'], config['vocab_tgt']

    # This value equals the index of <blank> in `specials`` when doing build_vocab_from_iterator
    pad_idx = vocab_tgt[padding]

    device = config['device']
    batch_size = config['batch_size']
    max_padding = config['max_padding']
    d_model = config['d_model']
    N = config['N']
    is_distributed = config['distributed']
    lr = config['base_lr']
    warmup = config['warmup']
    accum_iter = config["accum_iter"]
    num_epochs = config['num_epochs']

    model = translation_model(len(vocab_src), len(vocab_tgt), d_model, N)
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    train_dataloader, valid_dataloader = create_data_loaders(
        device,
        vocab_src,
        vocab_tgt,
        batch_size=batch_size,
        max_padding=max_padding,
        is_distributed=is_distributed
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step,
            d_model,
            factor=0.1,
            warmup=warmup
        )
    )
    loss_compute = SimpleLossCompute(model.generator, criterion)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        model.train()
        run_epoch(
            model,
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            loss_compute,
            optimizer,
            lr_scheduler,
            mode='train',
            accum_iter=accum_iter
        )

        model.eval()
        total_loss, total_token = run_epoch(
            model,
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            loss_compute,
            None,
            None,
            mode='eval'
        )
        print('Validation Average Loss: %.2f' % (total_loss / total_token))

    print("Training process finished.")
    torch.save(model.state_dict(), config['model_path'])

def train_distributed_model(config):
    raise NotImplementedError

def train_model(config):
    if config['distributed']:
        train_distributed_model(config)
    else:
        train_worker(config)
