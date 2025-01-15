import torch.nn as nn
import torch


class GPTModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        vocabulary_size = cfg["vocab_size"]
        embedding_dim = cfg["emb_dim"]
        context_length = cfg["context_length"]
        num_heads = cfg["n_heads"]
        drop_rate = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

        self.dropout = nn.Dropout(drop_rate)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    context_length,
                    embedding_dim,
                    num_heads,
                    drop_rate,
                    qkv_bias,
                )
                for _ in range(cfg["n_layers"])
            ]
        )
        self.final_layer_norm = LayerNorm(embedding_dim)
        self.out_head = nn.Linear(embedding_dim, vocabulary_size, bias=False)

    def forward(self, x):
        b, token_length = x.shape
        x = self.token_embedding(x) + self.position_embedding(
            torch.arange(0, token_length, device=x.device)
        )
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.out_head(x)
        return logits


class MultiheadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False
    ) -> None:
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout = nn.Dropout(p=dropout)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.linear = nn.Linear(d_out, d_out)

        self.register_buffer(
            "mask", torch.triu(torch.ones((context_length, context_length)), diagonal=1)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        b, num_tokens, emb_dim = x.shape  # batch, num_tokens, emb_dimension
        queries = self.W_query(x)  # batch, num_tokens, d_out
        keys = self.W_key(x)  # batch, num_tokens, d_out
        values = self.W_value(x)  # batch, num_tokens, d_out

        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  # batch, num_tokens, num_heads, head_dim
        keys = keys.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  # batch, num_tokens, num_heads, head_dim
        values = values.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  # batch, num_tokens, num_heads, head_dim

        queries = queries.transpose(1, 2)  # batch, num_heads, num_tokens, head_dim
        keys = keys.transpose(1, 2)  # batch, num_heads, num_tokens, head_dim
        values = values.transpose(1, 2)  # batch, num_heads, num_tokens, head_dim

        attn_scores = queries @ keys.transpose(
            2, 3
        )  # batch, num_heads, num_tokens, num_tokens
        bool_mask = self.mask.bool()[:num_tokens, :num_tokens]  # num_tokens, num_tokens
        attn_scores.masked_fill_(
            bool_mask, -torch.inf
        )  # batch, num_heads, num_tokens, num_tokens
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[3]) ** 0.5, dim=3
        )  # batch, num_heads, num_tokens, num_tokens
        attn_weights = self.dropout(attn_weights)

        z = attn_weights @ values  # batch, num_heads, num_tokens, head_dim
        z = z.transpose(1, 2)  # batch, num_tokens, num_heads, head_dim
        z = z.contiguous().view(b, num_tokens, self.d_out)  # batch, num_tokens, d_out

        z = self.linear(z)  # batch, num_tokens, d_out
        return z


class TransformerBlock(nn.Module):
    def __init__(
        self,
        context_length,
        embedding_dim,
        num_heads,
        dropout,
        qkv_bias=False,
    ) -> None:
        super().__init__()
        self.layer_norm1 = LayerNorm(embedding_dim)
        self.layer_norm2 = LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

        self.attention = MultiheadAttention(
            embedding_dim, embedding_dim, context_length, num_heads, dropout, qkv_bias
        )

        self.ff = FeedForward(embedding_dim)

    def forward(self, x):
        x_res = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x_res + x

        x_res = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x_res + x

        return x


class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.layers = nn.Sequential(
            *[
                nn.Linear(emb_dim, 4 * emb_dim),
                nn.GELU(),
                nn.Linear(4 * emb_dim, emb_dim),
            ]
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

        self.eps = 1e-5

    def forward(self, x: torch.tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x + self.shift
