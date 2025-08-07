import torch
import torch.nn.functional as F
from torch import Tensor

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Get the embedding of the last non-masked token."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]