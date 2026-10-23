import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens


def compute_probs(logits: torch.Tensor, temperatures: torch.Tensor):
    # temperatures must broadcast against logits without the vocab dim, e.g. [bs] for [bs, V]
    return torch.softmax(logits.float() / temperatures.unsqueeze(-1), dim=-1)


def sample_from_probs(probs: torch.Tensor):
    # argmax(p / Exp(1)) draws from p; p does not need to be normalized
    return probs.div(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)


def rejection_sample(target_probs: torch.Tensor, draft_tokens: torch.Tensor, draft_probs: torch.Tensor):
    """Verify one draft token per sequence.

    target_probs: [bs, 2, V], target distributions at [last token, draft token]
    draft_tokens: [bs], drafted from draft_probs: [bs, V]
    Returns tokens [bs, 2] and num_accepted [bs]; when the draft is rejected only tokens[:, 0] is valid.
    The output has exactly the target distribution (Leviathan et al., 2023).
    """
    p, d = target_probs[:, 0], draft_tokens.unsqueeze(1)
    p_d = p.gather(1, d).squeeze(1)
    q_d = draft_probs.gather(1, d).squeeze(1)
    accepted = torch.rand_like(p_d) * q_d < p_d    # accept with probability min(1, p/q)
    corrected = sample_from_probs((p - draft_probs).clamp_min_(0))
    first = torch.where(accepted, draft_tokens, corrected)
    bonus = sample_from_probs(target_probs[:, 1])
    return torch.stack([first, bonus], dim=1), accepted.long()
