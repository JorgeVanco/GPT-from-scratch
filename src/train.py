import os
import torch
from tqdm import tqdm
from data import (
    download_shakespeare,
    download_the_verdict,
    train_test_split,
    create_dataloader,
)
from finetuning import replace_linear_with_lora
from training import calc_loss_batch, calc_loss_loader
from text_generation import generate_text
from model import GPTModel
from config import CHOOSE_MODEL, model_configs
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
    start_context="Every effort moves you",
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
                        f"\nEpoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}, Tokens seen: {tokens_seen}"
                    )
                    print(
                        generate_text(model, start_context, 50, device).replace(
                            "\n", " "
                        ),
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
    start_context="Every effort moves you",
) -> tuple:
    total_params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
    print(f"Total trainable parameters before: {total_params:,}")

    for param in gpt.parameters():
        param.requires_grad = False
    total_params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
    print(f"Total trainable parameters after: {total_params:,}")
    replace_linear_with_lora(model, rank, alpha)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable LoRA parameters: {total_params:,}")
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.001, weight_decay=0.1)
    return train(
        model,
        train_dataloader,
        val_dataloader,
        epochs,
        optimizer,
        device,
        eval_freq,
        eval_iter,
        start_context,
    )


if __name__ == "__main__":
    data_path = "data/the-verdict.txt"
    download_data = True
    start_context = "Every effort moves you"

    if download_data:
        # download_shakespeare(data_path)
        download_the_verdict(data_path)

    with open(data_path) as fp:
        data = fp.read()

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 256,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }

    CHOOSE_MODEL = "gpt2-small (124M)"

    my_config = {
        "emb_dim": 200,
        "n_layers": 2,
        "n_heads": 4,
        "context_length": 16,
    }
    my_config = None
    if my_config is not None:
        BASE_CONFIG.update(my_config)
    else:
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = train_test_split(data, 0.9)
    torch.manual_seed(123)
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
        shuffle=False,
        drop_last=False,
    )
    print("Train dataloader length:", len(train_dataloader))
    print("Test dataloader length:", len(test_dataloader))

    gpt = GPTModel(BASE_CONFIG)
    gpt.to(device)

    print("Model configuration:", BASE_CONFIG)
    print(f"Model size: {sum(p.numel() for p in gpt.parameters()):,} parameters")

    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.0004, weight_decay=0.1)
    epochs = 10
    train_losses, val_losses = train(
        gpt,
        train_dataloader,
        test_dataloader,
        epochs,
        optimizer,
        device,
        eval_freq=5,
        eval_iter=5,
        start_context=start_context,
    )

    train_loss = calc_loss_loader(train_dataloader, gpt, device)
    val_loss = calc_loss_loader(test_dataloader, gpt, device)
    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")

    os.makedirs("trained_models", exist_ok=True)
    torch.save(gpt.state_dict(), "trained_models/gpt_model_verdict.pth")

    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.show()
