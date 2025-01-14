import torch


def calc_loss_batch(input_batch, target_batch, model, device) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(loader, model, device, num_batches=None) -> float:
    total_loss = 0
    if len(loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))

    for i, (input_batch, target_batch) in enumerate(loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches
