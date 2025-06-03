import torch
import tiktoken


def text_to_tokens(text, tokenizer) -> torch.Tensor:
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)
    return tokens_tensor


def tokens_to_text(tokens, tokenizer) -> str:
    return tokenizer.decode(tokens.squeeze().tolist())


def generate_text(
    model,
    context,
    max_length,
    device,
    tokenizer="gpt2",
    temperature=0.0,
    top_k=None,
    eos_id=None,
) -> str:
    encoder = tiktoken.get_encoding(tokenizer)
    context_encoded = text_to_tokens(context, encoder).to(device)

    context_length = model.context_length

    for _ in range(max_length):
        input_tokens = context_encoded[:, -context_length:]

        with torch.no_grad():
            logits = model(input_tokens)[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(-torch.inf).to(logits.device), logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            output_tokens = torch.multinomial(probas, num_samples=1)

        else:
            output_tokens = torch.argmax(logits, dim=-1, keepdim=True)

        if output_tokens[0][0] == eos_id:
            break

        context_encoded = torch.cat((context_encoded, output_tokens), dim=1)
    return tokens_to_text(context_encoded, encoder)
