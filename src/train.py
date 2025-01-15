import os
import torch
from tqdm import tqdm
from data import download_shakespeare, train_test_split, create_dataloader
from finetuning import replace_linear_with_lora
from training import calc_loss_batch, calc_loss_loader
from text_generation import generate_text
from model import GPTModel
from config import BASE_CONFIG
import matplotlib.pyplot as plt


def train(
    model,
    train_dataloader,
    val_dataloader,
    epochs,
    optimizer,
    device,
    eval_freq=100,
    eval_iter=5,
) -> tuple:
    train_losses, val_losses = [], []
    global_step = -1
    tokens_seen = 0
    model.train()
    for epoch in range(epochs):
        for input_batch, target_batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(
                        train_dataloader, model, device, num_batches=eval_iter
                    )
                    val_loss = calc_loss_loader(
                        val_dataloader, model, device, num_batches=eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(
                        f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}, Tokens seen: {tokens_seen}"
                    )
                    print(
                        "Generated text:", generate_text(model, "Hello, ", 10, device)
                    )

                model.train()
    return train_losses, val_losses


def fine_tune_lora(
    model,
    train_dataloader,
    val_dataloader,
    epochs,
    optimizer,
    device,
    rank=16,
    alpha=16,
    eval_freq=100,
    eval_iter=5,
) -> tuple:
    replace_linear_with_lora(model, rank, alpha)
    return train(
        model,
        train_dataloader,
        val_dataloader,
        epochs,
        optimizer,
        device,
        eval_freq,
        eval_iter,
    )


if __name__ == "__main__":
    data_path = "data/input.txt"
    download_shakespeare(data_path)
    with open(data_path) as fp:
        data = fp.read()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = train_test_split(data, 0.9)
    train_dataloader = create_dataloader(
        train_data,
        BASE_CONFIG["context_length"],
        BASE_CONFIG["context_length"],
        2,
        shuffle=True,
    )
    test_dataloader = create_dataloader(
        test_data,
        BASE_CONFIG["context_length"],
        BASE_CONFIG["context_length"],
        2,
        shuffle=True,
    )
    print("Train dataloader length:", len(train_dataloader))
    print("Test dataloader length:", len(test_dataloader))

    gpt = GPTModel(BASE_CONFIG)
    gpt.to(device)

    optimizer = torch.optim.AdamW(gpt.parameters(), 0.01)

    train_losses, val_losses = train(
        gpt, train_dataloader, test_dataloader, 2, optimizer, device
    )

    os.makedirs("trained_models", exist_ok=True)
    torch.save(gpt.state_dict(), "trained_models/gpt_model.pth")

    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.show()
