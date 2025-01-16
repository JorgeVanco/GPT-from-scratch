import torch
from loading_pretrained_weights import (
    get_huggingface_gpt2,
    load_weights,
)
from model import GPTModel
from text_generation import generate_text
from config import BASE_CONFIG, model_configs


def generate_text_simple(model, idx, max_new_tokens, context_size) -> torch.Tensor:
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
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


if __name__ == "__main__":

    pretrained_weights_path = "trained_models/gpt_model_verdict.pth"
    load_gpt2_weights = False
    CHOOSE_MODEL = "gpt2-small (124M)"
    context_length = 256  # 1024

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": context_length,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }

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

    gpt = GPTModel(BASE_CONFIG)

    if pretrained_weights_path is not None:
        gpt.load_state_dict(torch.load(pretrained_weights_path))
    if load_gpt2_weights:
        hf_model = get_huggingface_gpt2(CHOOSE_MODEL)
        load_weights(gpt, hf_model, BASE_CONFIG)
    gpt.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.to(device)

    start_context = "Every effort moves you"
    print(generate_text(gpt, start_context, 50, device))
