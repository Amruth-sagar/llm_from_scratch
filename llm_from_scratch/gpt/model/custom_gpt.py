from llm_from_scratch.gpt.model.blocks import TransformerBlockRoPE
import torch.nn as nn
import torch
import torch.nn.functional as F


class CustomGPTwithRoPE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.dropout_after_embed = nn.Dropout(cfg["drop_p_after_embed"])
        
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlockRoPE(cfg) for _ in range(cfg["num_tf_blocks"])]
        )

        self.final_layer_norm = nn.LayerNorm(cfg["embed_dim"])
        self.out_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, x, doc_ids=None, attn_mask=None):
        x = self.token_embedding(x)
        x = self.dropout_after_embed(x)
        for block in self.transformerBlocks:
            x = block(x, doc_ids=doc_ids, attn_mask=attn_mask)
        x = self.final_layer_norm(x)
        logits = self.out_head(x)

        return logits
    
    @torch.no_grad()
    def resize_embedding_and_lm_head(self, new_vocab_size):
        d_model = self.token_embedding.embedding_dim
        old_vocab_size = self.token_embedding.num_embeddings
        
        assert new_vocab_size > old_vocab_size, "New vocab should be larger"

        device = self.token_embedding.weight.device
        dtype = self.token_embedding.weight.dtype
        
        previous_token_embedding_weights = self.token_embedding.weight
        previous_out_head_weights = self.out_head.weight

        self.token_embedding = nn.Embedding(new_vocab_size, d_model, device=device, dtype=dtype)
        # Since bias=False is hardcoded for out_head, it is ignored.
        self.out_head = nn.Linear(d_model, new_vocab_size, device=device, dtype=dtype, bias=False)

        self.token_embedding.weight[:old_vocab_size].copy_(previous_token_embedding_weights)
        self.out_head.weight[:old_vocab_size].copy_(previous_out_head_weights)
        
        std = d_model ** -0.5
        nn.init.normal_(self.token_embedding.weight[old_vocab_size:], mean=0, std=std)
        nn.init.normal_(self.out_head.weight[old_vocab_size:], mean=0, std=std)

    @torch.no_grad()
    def generate(self, input_tokens, context_len, max_new_tokens, eos_token_id, temperature=1.0, top_k=None):
        self.eval()

        device = next(self.parameters()).device
        tokens = torch.tensor([input_tokens], dtype=torch.long, device=device)
        
        input_len = tokens.size(1)
        if input_len + max_new_tokens > context_len:
            max_new_tokens = context_len - input_len
            print(f"Warning: max_new_tokens adjusted to {max_new_tokens} to fit context_len")

        for _ in range(max_new_tokens):
            logits = self(tokens)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # Stop if EOS token is generated
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break

            tokens = torch.cat((tokens, next_id), dim=1)
        
        return tokens.squeeze(0).tolist()

