import torch
import torch.nn as nn
import math


class QFormerLayer(nn.Module):
    """Single Q-Former block: self-attn (queries interact) -> cross-attn -> FFN."""

    def __init__(self, hidden_dim=768, vision_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries, vision_features, self_attn_mask=None):
        # Self-attention: 32 queries interact with each other
        residual = queries
        queries = self.norm1(queries)
        queries = residual + self.self_attn(
            queries, queries, queries, attn_mask=self_attn_mask
        )[0]

        # Cross-attention: queries attend to visual features
        residual = queries
        queries = self.norm2(queries)
        queries = residual + self.cross_attn(
            queries, vision_features, vision_features
        )[0]

        # Feed-forward
        residual = queries
        queries = self.norm3(queries)
        queries = residual + self.ffn(queries)

        return queries


class VisionProjector(nn.Module):
    """Linear projection + LayerNorm if vision dim differs from Q-Former hidden dim."""

    def __init__(self, vision_dim, hidden_dim):
        super().__init__()
        self.need_proj = vision_dim != hidden_dim
        if self.need_proj:
            self.proj = nn.Linear(vision_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features):
        if self.need_proj:
            vision_features = self.norm(self.proj(vision_features))
        return vision_features


class MiniQFormer(nn.Module):
    """Mini Q-Former: receives visual features and outputs 32 learnable query embeddings.

    The 32 queries interact via self-attention and gather information from visual
    features via cross-attention across multiple transformer layers.

    Args:
        vision_dim: dimension of input visual features (768 for CLIP ViT-B/32)
        hidden_dim: internal hidden dimension
        num_queries: number of learnable query tokens (default 32)
        num_layers: number of Q-Former layers (default 2 for "mini")
        num_heads: number of attention heads
        dropout: dropout rate
    """

    def __init__(
        self,
        vision_dim=768,
        hidden_dim=768,
        num_queries=32,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        self.vision_projector = VisionProjector(vision_dim, hidden_dim)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # Learnable position embedding for queries (optional, helps distinguish queries)
        self.query_pos = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.normal_(self.query_pos, std=0.02)

    def forward(self, vision_features, self_attn_mask=None):
        """
        Args:
            vision_features: (batch, seq_len, vision_dim) — patch-level features
            self_attn_mask: optional attention mask for self-attention (e.g. causal)

        Returns:
            queries: (batch, num_queries, hidden_dim) — 32 refined query embeddings
        """
        batch_size = vision_features.shape[0]

        # Project vision features if needed
        vision_features = self.vision_projector(vision_features)

        # Expand learnable queries to batch: (num_queries, hidden_dim) -> (batch, num_queries, hidden_dim)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + self.query_pos.unsqueeze(0)

        for layer in self.layers:
            queries = layer(queries, vision_features, self_attn_mask)

        return self.final_norm(queries)


if __name__ == "__main__":
    # Quick smoke test
    print("Testing MiniQFormer...")

    model = MiniQFormer(
        vision_dim=768,
        hidden_dim=768,
        num_queries=32,
        num_layers=2,
        num_heads=8,
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Simulate CLIP ViT-B/32 last_hidden_state: (batch, 50, 768)
    dummy_vision_features = torch.randn(4, 50, 768)

    with torch.no_grad():
        queries = model(dummy_vision_features)

    print(f"Input  shape:  {dummy_vision_features.shape}  # (batch, num_patches+CLS, vision_dim)")
    print(f"Output shape:  {queries.shape}              # (batch, num_queries, hidden_dim)")
    print(f"Number of queries: {queries.shape[1]}")
    print("Test passed.")
