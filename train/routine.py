import time

from tqdm import tqdm

custom_bar_format = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    + "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
)


def run_epoch(
    model, data_iter, loss_compute, optimizer, scheduler, mode, accum_iter=1
):
    tokens = 0
    total_loss = 0
    total_token = 0
    accum_step = 0

    bar = tqdm(data_iter, bar_format=custom_bar_format)
    start = time.time()
    for i, batch in enumerate(bar):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train":
            loss_node.backward()
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_step += 1
            scheduler.step()

        tokens += batch.ntokens
        total_token += batch.ntokens
        total_loss += loss
        if i % accum_iter == 0 and mode == "train":
            elapsed = time.time() - start
            cur_loss_each_token = loss / batch.ntokens
            tokens_processing_rate = tokens / elapsed
            lr = optimizer.param_groups[0]["lr"]
            postfix = "Loss: %4.2f, Tokens/Sec: %6.1f, LR: %4.2e" % (
                cur_loss_each_token,
                tokens_processing_rate,
                lr,
            )
            bar.set_postfix_str(postfix, refresh=False)
            start = time.time()
            tokens = 0
        del loss, loss_node
    return total_loss.item(), total_token.item()
