import torch
import torch.nn.functional as F

def cross_entropy(logits, targets):
    batch_size, num_tokens, vocab_size = logits.shape

    logits = logits.view(batch_size * num_tokens, vocab_size)
    targets = targets.view(batch_size * num_tokens)

    loss = F.cross_entropy(
        logits, 
        targets,
        reduction="mean",
        ignore_index=-100
    )

    return loss


def perplexity(ce_loss):
    return torch.exp(ce_loss)