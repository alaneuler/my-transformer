import time


def run_epoch(model, data_iter, loss_compute, optimizer, scheduler, mode, accum_iter=1):
    tokens = 0
    total_loss = 0
    total_token = 0
    accum_step = 0

    start = time.time()
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
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
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "  Step: %3d | Accum: %3d | Loss: %4.2f | Tokens/Sec: %7.1f | LR: %4.2e"
                )
                % (i, accum_step, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss, loss_node
    return total_loss.item(), total_token.item()
