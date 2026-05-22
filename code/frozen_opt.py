import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class FrozenOPTDecoder(nn.Module):
    """Frozen OPT-125m as language decoder, conditioned on Q-Former visual queries.

    The 32 projected visual queries from MiniQFormer are prepended to text token
    embeddings as "soft visual prompts". The OPT model is kept frozen throughout.

    Args:
        model_path: path to local OPT-125m directory
        device: device to run the model on
    """

    def __init__(self, model_path=r"D:\blip2-main\opt-125m", device="cuda"):
        super().__init__()
        self.device = device

        # Load model and tokenizer from local path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token = "[PAD]"

        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        # Resize embeddings for the new pad token (frozen, so the new row stays random)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.config = self.model.config
        self.embed_dim = self.config.hidden_size  # 768 for opt-125m

        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_text_embeddings(self, input_ids):
        """Convert token IDs to text embeddings using the frozen embedding layer."""
        return self.model.model.decoder.embed_tokens(input_ids)

    def build_inputs_embeds(self, visual_queries, input_ids):
        """Prepend 32 visual query embeddings to text token embeddings.

        Args:
            visual_queries: (batch, 32, embed_dim) — projected Q-Former output
            input_ids: (batch, text_len) — tokenized text

        Returns:
            inputs_embeds:  (batch, 32 + text_len, embed_dim)
            attention_mask: (batch, 32 + text_len)
        """
        batch_size = input_ids.shape[0]
        text_embeds = self.get_text_embeddings(input_ids.to(self.device))

        # Build attention mask: 1 for visual queries + text tokens, 0 for padding
        text_mask = (input_ids != self.tokenizer.pad_token_id).long()
        visual_mask = torch.ones(batch_size, visual_queries.shape[1], dtype=torch.long)

        inputs_embeds = torch.cat([visual_queries, text_embeds], dim=1)
        attention_mask = torch.cat([visual_mask, text_mask], dim=1).to(self.device)

        return inputs_embeds, attention_mask

    @torch.no_grad()
    def generate(self, visual_queries, input_ids, max_new_tokens=32, **gen_kwargs):
        """Generate text conditioned on visual queries.

        Args:
            visual_queries: (batch, 32, embed_dim) from Q-Former + LanguageProjection
            input_ids: (batch, text_len) tokenized prompt (e.g. BOS token only)
            max_new_tokens: max tokens to generate

        Returns:
            output_ids: (batch, 32 + text_len + new_tokens)
        """
        inputs_embeds, attention_mask = self.build_inputs_embeds(visual_queries, input_ids)

        defaults = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        defaults.update(gen_kwargs)

        output_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **defaults,
        )
        return output_ids

    def decode(self, output_ids, skip_special_tokens=True):
        return self.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)


if __name__ == "__main__":
    print("Loading frozen OPT-125m...")
    decoder = FrozenOPTDecoder(model_path=r"D:\blip2-main\opt-125m", device="cpu")
    total = sum(p.numel() for p in decoder.model.parameters())
    trainable = sum(p.numel() for p in decoder.model.parameters() if p.requires_grad)
    print(f"OPT embed dim: {decoder.embed_dim}")
    print(f"OPT total params: {total:,}")
    print(f"OPT trainable params: {trainable:,}  (should be 0)")

    # Smoke test: prepend visual queries and generate
    dummy_queries = torch.randn(2, 32, decoder.embed_dim)  # simulated Q-Former output
    prompt = ["a photo of", "a picture showing"]

    # Tokenize prompt
    enc = decoder.tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"]

    print(f"\nVisual queries: {dummy_queries.shape}")
    print(f"Input IDs:      {input_ids.shape}")

    embeds, mask = decoder.build_inputs_embeds(dummy_queries, input_ids)
    print(f"Combined embeds: {embeds.shape}  (32 visual + {input_ids.shape[1]} text)")
    print(f"Attention mask:  {mask.shape}")

    # Quick generate
    output_ids = decoder.generate(dummy_queries, input_ids, max_new_tokens=8)
    print(f"Output IDs:      {output_ids.shape}")

    for i, ids in enumerate(output_ids):
        text = decoder.decode(ids)
        print(f"  Sample {i}: {text}")

    print("\nTest passed.")
