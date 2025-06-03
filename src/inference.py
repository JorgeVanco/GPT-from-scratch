import torch
from src.loading_pretrained_weights import (
    get_huggingface_gpt2,
    load_weights,
)
from src.model import GPTModel
from src.text_generation import generate_text
from src.config import BASE_CONFIG, model_configs


def generate_text_simple(model, idx, max_new_tokens, context_size) -> torch.Tensor:
    """Generate text using the model."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer) -> torch.Tensor:
    """Convert text to token IDs using the specified tokenizer."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs back to text using the specified tokenizer."""
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


if __name__ == "__main__":

    PRETRAINED_WEIGHTS_PATH = "trained_models/gpt_model_verdict.pth"
    LOAD_GPT2_WEIGHTS = False
    CHOOSE_MODEL = "gpt2-small (124M)"
    context_length = 256  # 1024

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": context_length,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }

    MY_CONFIG = {
        "emb_dim": 200,
        "n_layers": 2,
        "n_heads": 4,
        "context_length": 16,
    }
    MY_CONFIG = None
    if MY_CONFIG is not None:
        BASE_CONFIG.update(MY_CONFIG)
    else:
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    gpt = GPTModel(BASE_CONFIG)

    if PRETRAINED_WEIGHTS_PATH is not None:
        gpt.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH))
    if LOAD_GPT2_WEIGHTS:
        hf_model = get_huggingface_gpt2(CHOOSE_MODEL)
        load_weights(gpt, hf_model, BASE_CONFIG)
    gpt.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.to(device)

    start_context = "Every effort moves you"
    print(generate_text(gpt, start_context, 50, device))
