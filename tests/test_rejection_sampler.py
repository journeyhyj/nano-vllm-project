# rejection_sample must emit exactly the target distribution, whatever the draft distribution, including a draft
# distribution that only covers part of the vocabulary (the draft vocabulary, spread with zeros as ModelRunner does).
# Usage: python tests/test_rejection_sampler.py   (needs a GPU)
import torch
from nanovllm.layers.sampler import rejection_sample, sample_from_probs, compute_probs

V, N = 10, 1_000_000


def chi_square_p_value(counts: torch.Tensor, probs: torch.Tensor) -> float:
    expected = probs.double().cpu() * counts.sum().item()
    stat = ((counts.double().cpu() - expected) ** 2 / expected).sum()
    return torch.special.gammaincc(torch.tensor((len(probs) - 1) / 2, dtype=torch.float64), stat / 2).item()


def run(seed: int, draft_ids: torch.Tensor | None):
    torch.manual_seed(seed)
    p = torch.softmax(torch.randn(2, V) * 1.5, -1).cuda()    # target distributions at [last token, draft token]
    num_draft = V if draft_ids is None else len(draft_ids)
    q_draft = torch.softmax(torch.randn(num_draft), -1).cuda().expand(N, -1).contiguous()
    draft_tokens = sample_from_probs(q_draft)
    q = q_draft
    if draft_ids is not None:
        draft_tokens = draft_ids[draft_tokens]
        q = q_draft.new_zeros(N, V).index_copy_(1, draft_ids, q_draft)
    tokens, accepted = rejection_sample(p.expand(N, 2, V).contiguous(), draft_tokens, q)
    return p, q[0], tokens, accepted


def test_target_distribution():
    for draft_ids in (None, torch.tensor([7, 0, 3, 5, 2], device="cuda")):
        p_values = []
        for seed in range(8):
            p, q, tokens, accepted = run(seed, draft_ids)
            p_values.append(chi_square_p_value(torch.bincount(tokens[:, 0], minlength=V), p[0]))
            expected_acceptance = torch.minimum(p[0], q).sum().item()
            assert abs(accepted.float().mean().item() - expected_acceptance) < 3e-3
            bonus = torch.bincount(tokens[accepted.bool(), 1], minlength=V)    # only used when the draft is accepted
            p_values.append(chi_square_p_value(bonus, p[1]))
        name = "full draft vocabulary" if draft_ids is None else "draft vocabulary of 5 / 10 tokens"
        print(f"{name}: chi-square p-values {[round(x, 3) for x in p_values]}")
        assert min(p_values) > 1e-3


def test_greedy_limit():
    # at temperature -> 0 a draft is accepted iff it is the target argmax, otherwise the argmax is emitted
    torch.manual_seed(0)
    draft_ids = torch.tensor([7, 0, 3, 5, 2], device="cuda")
    logits = torch.randn(1000, 2, V, device="cuda") * 3
    t = torch.full((1000,), 1e-5, device="cuda")
    p = compute_probs(logits, t.view(-1, 1))
    q_draft = compute_probs(torch.randn(1000, len(draft_ids), device="cuda") * 3, t)
    draft_tokens = draft_ids[q_draft.argmax(-1)]
    tokens, accepted = rejection_sample(p, draft_tokens, q_draft.new_zeros(1000, V).index_copy_(1, draft_ids, q_draft))
    target = logits[:, 0].argmax(-1)
    assert torch.equal(tokens[:, 0], target) and torch.equal(accepted.bool(), draft_tokens == target)


if __name__ == "__main__":
    test_target_distribution()
    test_greedy_limit()
    print("PASS")
