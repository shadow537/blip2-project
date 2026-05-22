import torch
import torch.nn as nn
from mini_qformer import MiniQFormer
from frozen_opt import FrozenOPTDecoder


class BLIPModel(nn.Module):
    """Full BLIP-2: frozen vision encoder + trainable MiniQFormer + frozen OPT decoder.

    Training targets:  MiniQFormer + LanguageProjection
    Frozen:            Vision Encoder (features pre-extracted) + OPT-125m

    Forward returns logits and can compute cross-entropy loss directly.
    """

    def __init__(
        self,
        opt_path=r"D:\blip2-main\opt-125m",
        vision_dim=768,
        hidden_dim=768,
        num_queries=32,
        num_qformer_layers=2,
        num_heads=8,
        dropout=0.1,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.num_queries = num_queries

        # Frozen OPT decoder (loads tokenizer + LM)
        self.opt_decoder = FrozenOPTDecoder(model_path=opt_path, device=device)
        self.embed_dim = self.opt_decoder.embed_dim  # 768
        self.pad_token_id = self.opt_decoder.tokenizer.pad_token_id

        # Trainable: MiniQFormer + LanguageProjection
        self.qformer = MiniQFormer(
            vision_dim=vision_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_qformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            lm_embed_dim=self.embed_dim,
        )

    def train(self, mode=True):
        super().train(mode)
        # OPT stays frozen regardless
        self.opt_decoder.model.eval()
        return self

    def forward(self, vision_features, input_ids, labels=None):
        """Forward pass with optional loss computation.

        Args:
            vision_features: (batch, 50, 768) pre-extracted CLIP last_hidden_state
            input_ids:       (batch, text_len) tokenized caption
            labels:          (batch, text_len) tokenized caption for CE loss.
                             If None, defaults to input_ids (standard LM objective).

        Returns:
            dict with keys:
                - logits: (batch, 32 + text_len, vocab_size)
                - loss:   scalar cross-entropy loss (only if labels or input_ids provided)
        """
        batch_size = vision_features.shape[0]
        text_len = input_ids.shape[1]

        # 1. Q-Former + LanguageProjection → visual queries
        visual_queries = self.qformer(vision_features.to(self.device))
        # visual_queries: (batch, 32, embed_dim)

        # 2. Get text embeddings from frozen OPT
        text_embeds = self.opt_decoder.get_text_embeddings(input_ids.to(self.device))
        # text_embeds: (batch, text_len, embed_dim)

        # 3. Prepend visual queries to text embeddings
        inputs_embeds = torch.cat([visual_queries, text_embeds], dim=1)
        # inputs_embeds: (batch, 32 + text_len, embed_dim)

        # 4. Build attention mask
        text_mask = (input_ids != self.pad_token_id).long().to(self.device)
        visual_mask = torch.ones(batch_size, self.num_queries, dtype=torch.long, device=self.device)
        attention_mask = torch.cat([visual_mask, text_mask], dim=1)

        # 5. Forward through frozen OPT
        outputs = self.opt_decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (batch, 32 + text_len, vocab_size)

        # 6. Compute cross-entropy loss
        loss = None
        if labels is None:
            labels = input_ids

        # Mask padding positions in labels so they don't contribute to loss
        labels = labels.clone()
        labels[input_ids == self.pad_token_id] = -100

        # Build full labels: also ignore visual prefix positions
        visual_labels = torch.full(
            (batch_size, self.num_queries), -100,
            dtype=torch.long, device=self.device,
        )
        full_labels = torch.cat([visual_labels, labels.to(self.device)], dim=1)
        # full_labels: (batch, 32 + text_len), visual + PAD positions = -100

        # Hugging Face internally shifts logits/labels for autoregressive loss
        loss = self.opt_decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels,
        ).loss

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, vision_features, prompt=None, max_new_tokens=32, **gen_kwargs):
        """Generate a caption from vision features.

        Args:
            vision_features: (batch, 50, 768)
            prompt: optional text prompt string or list of strings.
                    If None, uses BOS token only.
            max_new_tokens: max tokens to generate.

        Returns:
            list of decoded text strings.
        """
        self.eval()

        batch_size = vision_features.shape[0]
        if prompt is None:
            bos = self.opt_decoder.tokenizer.bos_token or ""
            prompt = [bos] * batch_size
        elif isinstance(prompt, str):
            prompt = [prompt] * batch_size

        enc = self.opt_decoder.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]

        visual_queries = self.qformer(vision_features.to(self.device))
        output_ids = self.opt_decoder.generate(
            visual_queries, input_ids, max_new_tokens=max_new_tokens, **gen_kwargs
        )

        return [self.opt_decoder.decode(ids) for ids in output_ids]


if __name__ == "__main__":
    print("=== BLIP Model Assembly Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = BLIPModel(
        opt_path=r"D:\blip2-main\opt-125m",
        vision_dim=768,
        hidden_dim=768,
        num_queries=32,
        num_qformer_layers=2,
        num_heads=8,
        device=device,
    )

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total params:    {total:>12,}")
    print(f"Trainable:       {trainable:>12,}  (MiniQFormer + Projection)")
    print(f"Frozen:          {frozen:>12,}  (Vision Encoder + OPT)")

    # Smoke test with dummy data
    vision_feat = torch.randn(2, 50, 768)
    captions = ["a dog running on grass", "a red car parked"]

    enc = model.opt_decoder.tokenizer(captions, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"]
    labels = enc["input_ids"].clone()

    print(f"\nVision features: {vision_feat.shape}")
    print(f"Input IDs:       {input_ids.shape}")
    print(f"Caption tokens:  {input_ids[0].tolist()}")

    out = model(vision_feat, input_ids, labels)
    print(f"Logits:          {out['logits'].shape}")
    print(f"Loss:            {out['loss']:.4f}")

    # Quick generate test
    gen_texts = model.generate(vision_feat[:1], prompt="a", max_new_tokens=8)
    print(f"\nGenerated: {gen_texts[0]}")

    print("\nTest passed.")
