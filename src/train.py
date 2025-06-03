import os
import torch
from tqdm import tqdm
from src.data import (
    download_shakespeare,
    download_the_verdict,
    train_test_split,
    create_dataloader,
)
from src.finetuning import replace_linear_with_lora
from src.training import calc_loss_batch, calc_loss_loader
from src.text_generation import generate_text
from src.model import GPTModel
from src.config import CHOOSE_MODEL, model_configs
import matplotlib.pyplot as plt
import wandb
from src.loading_pretrained_weights import (
    get_huggingface_gpt2,
    load_weights,
)


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
    """Train the GPT model on the provided data."""
    train_losses, val_losses = [], []
    global_step = -1
    tokens_seen = 0

    text_table = wandb.Table(columns=["epoch", "tokens seen", "text"])

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
                    wandb.log(
                        {"train_loss": train_loss, "val_loss": val_loss},
                        step=tokens_seen,
                    )
                    print(
                        f"\nEpoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}, Tokens seen: {tokens_seen}"
                    )
                    generated_text = generate_text(
                        model, start_context, 50, device, top_k=10, temperature=0.7
                    ).replace("\n", " ")
                    text_table.add_data(epoch, tokens_seen, generated_text)
                    text_table = wandb.Table(
                        columns=text_table.columns, data=text_table.data
                    )
                    wandb.log({"Generated Text Table": text_table})
                    print(generated_text)

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
    """Fine-tune the model using LoRA."""
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

    wandb.init(project="gpt2")

    useLora = True
    use_gpt2_config = False
    load_from_huggingface = True
    CHOOSE_MODEL = "gpt2-small (124M)"
    start_context = "Every effort moves you"
    batch_size = 8
    epochs = 10

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 256,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }

    my_config = {
        "emb_dim": 200,
        "n_layers": 2,
        "n_heads": 4,
        "context_length": 16,
    }

    DATA_PATH = "data/the-verdict.txt"
    DOWNLOAD_DATA = True

    if DOWNLOAD_DATA:
        # download_shakespeare(data_path)
        download_the_verdict(DATA_PATH)

    with open(DATA_PATH, encoding="utf-8") as fp:
        data = fp.read()

    if my_config is not None and not use_gpt2_config:
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
        batch_size,
        shuffle=True,
    )
    test_dataloader = create_dataloader(
        test_data,
        BASE_CONFIG["context_length"],
        BASE_CONFIG["context_length"],
        batch_size,
        shuffle=False,
        drop_last=False,
    )
    print("Train dataloader length:", len(train_dataloader))
    print("Test dataloader length:", len(test_dataloader))

    gpt = GPTModel(BASE_CONFIG)
    if load_from_huggingface and use_gpt2_config:
        hf_model = get_huggingface_gpt2(CHOOSE_MODEL)
        load_weights(gpt, hf_model, BASE_CONFIG)
    gpt.to(device)

    print("Model configuration:", BASE_CONFIG)
    print(f"Model size: {sum(p.numel() for p in gpt.parameters()):,} parameters")

    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.0004, weight_decay=0.1)

    if useLora:
        train_losses, val_losses = fine_tune_lora(
            gpt,
            train_dataloader,
            test_dataloader,
            epochs,
            optimizer,
            device,
            rank=32,
            alpha=64,
            eval_freq=50,
            eval_iter=50,
            start_context=start_context,
        )
    else:
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
