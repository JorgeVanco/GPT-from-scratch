from typing import Literal
import torch
import numpy as np
from transformers import GPT2Model


def get_huggingface_gpt2(
    model_name: Literal[
        "gpt2-small (124M)",
        "gpt2-medium (355M)",
        "gpt2-large (774M)",
        "gpt2-xl (1558M)",
    ],
) -> GPT2Model:
    model_names = {
        "gpt2-small (124M)": "openai-community/gpt2",
        "gpt2-medium (355M)": "openai-community/gpt2-medium",
        "gpt2-large (774M)": "openai-community/gpt2-large",
        "gpt2-xl (1558M)": "openai-community/gpt2-xl",
    }
    gpt_hf = GPT2Model.from_pretrained(model_names[model_name], cache_dir="checkpoints")
    gpt_hf.eval()
    return gpt_hf


def assign_check(left, right) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


def load_weights(gpt, gpt_hf, BASE_CONFIG) -> None:

    d = gpt_hf.state_dict()

    gpt.position_embedding.weight = assign_check(
        gpt.position_embedding.weight, d["wpe.weight"]
    )
    gpt.token_embedding.weight = assign_check(
        gpt.token_embedding.weight, d["wte.weight"]
    )

    for b in range(BASE_CONFIG["n_layers"]):
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.transformer_blocks[b].attention.W_query.weight = assign_check(
            gpt.transformer_blocks[b].attention.W_query.weight, q_w.T
        )
        gpt.transformer_blocks[b].attention.W_key.weight = assign_check(
            gpt.transformer_blocks[b].attention.W_key.weight, k_w.T
        )
        gpt.transformer_blocks[b].attention.W_value.weight = assign_check(
            gpt.transformer_blocks[b].attention.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.transformer_blocks[b].attention.W_query.bias = assign_check(
            gpt.transformer_blocks[b].attention.W_query.bias, q_b
        )
        gpt.transformer_blocks[b].attention.W_key.bias = assign_check(
            gpt.transformer_blocks[b].attention.W_key.bias, k_b
        )
        gpt.transformer_blocks[b].attention.W_value.bias = assign_check(
            gpt.transformer_blocks[b].attention.W_value.bias, v_b
        )

        gpt.transformer_blocks[b].attention.linear.weight = assign_check(
            gpt.transformer_blocks[b].attention.linear.weight,
            d[f"h.{b}.attn.c_proj.weight"].T,
        )
        gpt.transformer_blocks[b].attention.linear.bias = assign_check(
            gpt.transformer_blocks[b].attention.linear.bias,
            d[f"h.{b}.attn.c_proj.bias"],
        )

        gpt.transformer_blocks[b].ff.layers[0].weight = assign_check(
            gpt.transformer_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T
        )
        gpt.transformer_blocks[b].ff.layers[0].bias = assign_check(
            gpt.transformer_blocks[b].ff.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"]
        )
        gpt.transformer_blocks[b].ff.layers[2].weight = assign_check(
            gpt.transformer_blocks[b].ff.layers[2].weight,
            d[f"h.{b}.mlp.c_proj.weight"].T,
        )
        gpt.transformer_blocks[b].ff.layers[2].bias = assign_check(
            gpt.transformer_blocks[b].ff.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"]
        )

        gpt.transformer_blocks[b].layer_norm1.scale = assign_check(
            gpt.transformer_blocks[b].layer_norm1.scale, d[f"h.{b}.ln_1.weight"]
        )
        gpt.transformer_blocks[b].layer_norm1.shift = assign_check(
            gpt.transformer_blocks[b].layer_norm1.shift, d[f"h.{b}.ln_1.bias"]
        )
        gpt.transformer_blocks[b].layer_norm2.scale = assign_check(
            gpt.transformer_blocks[b].layer_norm2.scale, d[f"h.{b}.ln_2.weight"]
        )
        gpt.transformer_blocks[b].layer_norm2.shift = assign_check(
            gpt.transformer_blocks[b].layer_norm2.shift, d[f"h.{b}.ln_2.bias"]
        )

        gpt.final_layer_norm.scale = assign_check(
            gpt.final_layer_norm.scale, d[f"ln_f.weight"]
        )
        gpt.final_layer_norm.shift = assign_check(
            gpt.final_layer_norm.shift, d[f"ln_f.bias"]
        )
        gpt.out_head.weight = assign_check(gpt.out_head.weight, d["wte.weight"])
