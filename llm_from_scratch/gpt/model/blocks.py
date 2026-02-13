import torch
import torch.nn as nn


class FF(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_out, 4 * d_out), nn.GELU(), nn.Linear(4 * d_out, d_out)
        )

    def forward(self, x):
        return self.ff(x)


# RoPE rotates queries and keys by a position-dependent angle.
# RoPE:
#   d_out --> get pairs of dimensions (d1, d2) (d3, d4) ... (d_out-1, d_out)
#   Each pair has a rotation angle, r_1, r_2 ... r_{d_out/2} and 
#   
#   r_i = base ^ {-2i/d} = 10000 ^ {-2i/d}.
# 
#   Rotate dimension pairs of Q and K by r_i * position (base angle times token position)
#   Then take the dot product as usual.   

def rotate_half(x):
    # x.shape is (batch_size, sequence_len, dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2,x1), dim=-1).flatten(-2)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, base, max_seq_len, head_dim):
        super().__init__()
        self.base = base
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        assert self.head_dim % 2 == 0, "RoPE required even number of dimension in Q and K"

        inv_freq = 1.0 / (self.base ** (torch.arange(0,self.head_dim,2).float()/self.head_dim))

        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(self.max_seq_len)

    def _build_cache(self, max_seq_len):
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # freqs.shape = (max_seq_len, num_pairs) = (max_seq_len , head_dim//2)
        # example: if we have head_dim = 4 and max_seq_len only 2, then freqs = [[0*r_1, 0*r_2],[1*r_1, 1*r_2]]
        # each row = ( position * r_i ), for i \in 1 to head_dim//2
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)

        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def _build_position_ids(self, doc_ids):

        batch, seq_len = doc_ids.shape
        device = doc_ids.device

        boundary = torch.zeros_like(doc_ids, dtype=torch.bool)
        boundary[:, 0] = True
        boundary[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]

        arange = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        boundary_pos = torch.where(boundary, arange, torch.zeros_like(arange))
        last_boundary_pos = boundary_pos.cummax(dim=1).values

        position_ids = arange - last_boundary_pos

        return position_ids

    def forward(self, q, k, doc_ids=None):
        # Q or K's shape = (batch, num_heads, seq_len, head_dim)
        seq_len = q.shape[2]

        if doc_ids is None:
            # They will be broadcasted across batch dimension
            cos = self.cos_cached[:seq_len].view(1, 1, seq_len, -1)       # (1, 1, seq_len, head_dim//2)
            sin = self.sin_cached[:seq_len].view(1, 1, seq_len, -1)       # (1, 1, seq_len, head_dim//2)
            
        else:
            position_ids = self._build_position_ids(doc_ids).to(q.device)
            cos = self.cos_cached[position_ids].unsqueeze(1)     # (batch, 1, seq_len, head_dim//2)
            sin = self.sin_cached[position_ids].unsqueeze(1)     # (batch, 1, seq_len, head_dim//2)

        cos = cos.repeat_interleave(2, dim=-1)       # (batch (or) 1, 1, seq_len, head_dim)
        sin = sin.repeat_interleave(2, dim=-1)       # (batch (or) 1, 1, seq_len, head_dim)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        return q, k
    

class MaskedMultiheadAttnWithRoPE(nn.Module):
    def __init__(self, num_heads, d_in, d_out, context_len, attn_dropout, rope_base, max_seq_len, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out should be divisible by num_heads"

        self.Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.V = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(attn_dropout)

        self.d_in, self.d_out, self.head_dim = d_in, d_out, (d_out // num_heads)
        self.num_heads = num_heads

        self.rope = RotaryPositionalEmbedding(rope_base, max_seq_len, self.head_dim)

        self.register_buffer(
            "attention_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool(),
            persistent=False
        )

    
    def forward(self, x, doc_ids=None, attn_mask=None):
        batch, seq_len, d_in = x.shape
        q, k, v  = self.Q(x), self.K(x), self.V(x)

        # Dividing a d_out feature into num_heads * head_dim features
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        # (batch, seq_len, num_heads, head_dim) ==> (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2) 
        v = v.transpose(1, 2)

        # RoPE works on head_dim
        if doc_ids is None:
            q, k = self.rope(q, k)
        else:
            q, k = self.rope(q, k, doc_ids)
        
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # resultant shape is (batch, num_heads, seq_len, seq_len)
        attention_scores = q @ k.transpose(2, 3)
        
        
        if doc_ids is None:
            mask_bool = self.attention_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
            
        else:
            mask_bool = torch.tril((doc_ids.unsqueeze(1) == doc_ids.unsqueeze(2))) ^ True   # (batch, seq_len, seq_len)
            mask_bool = mask_bool.unsqueeze(1)      # (batch, 1, seq_len, seq_len)

        # when <pad> tokens are used
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :]  # (batch, 1, 1, seq_len)
            mask_bool = mask_bool | attn_mask

        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # row-wise softmax 
        attention_weights = torch.softmax(attention_scores / (self.head_dim ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # = (batch, num_heads, seq_len, head_dim) -> transpose -> (batch, seq_len, num_heads, head_dim)
        context_vec = (attention_weights @ v).transpose(1,2)

        # as if all the head_dim vectors are concatenated to become one big d_out context vec
        concatenated_ctx_vec = context_vec.contiguous().view(batch, seq_len, self.d_out)

        concatenated_ctx_vec = self.out_proj(concatenated_ctx_vec)
        return concatenated_ctx_vec



class TransformerBlockRoPE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.pre_layernorm_1 = nn.LayerNorm(cfg["embed_dim"])
        self.pre_layernorm_2 = nn.LayerNorm(cfg["embed_dim"])

        self.mmha = MaskedMultiheadAttnWithRoPE(
            num_heads=cfg["num_heads"],
            d_in=cfg["embed_dim"],
            d_out=cfg["embed_dim"],
            context_len=cfg["context_len"],
            attn_dropout=cfg["drop_p_for_mmha"],
            rope_base=cfg["rope_base"], 
            max_seq_len=cfg["max_seq_len"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.dropout_befor_skip_conn = nn.Dropout(cfg["drop_p_post_mmha"])

        self.ff = FF(cfg["embed_dim"])

    def forward(self, x, doc_ids=None, attn_mask=None):

        # --> x --> LN --> MMHA --> Drop --> [+] --> x
        #     |                              |
        #     -------------------------------

        x_from_skip_conn = x
        x = self.pre_layernorm_1(x)
        x = self.mmha(x, doc_ids=doc_ids, attn_mask=attn_mask)
        x = self.dropout_befor_skip_conn(x)
        x = x + x_from_skip_conn

        # --> x --> LN --> FF --> Drop --> [+] --> x
        #     |                            |
        #     -----------------------------

        x_from_skip_conn = x
        x = self.pre_layernorm_2(x)
        x = self.ff(x)
        x = self.dropout_befor_skip_conn(x)
        x = x + x_from_skip_conn

        return x
