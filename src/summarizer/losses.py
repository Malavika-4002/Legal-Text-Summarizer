import torch
import torch.nn.functional as F

def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int
):
    """
    logits:  (B, T, V)
    targets: (B, T)
    """

    B, T, V = logits.size()

    logits = logits.view(B * T, V)
    targets = targets.view(B * T)

    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=pad_id,
        reduction="mean"
    )

    return loss
