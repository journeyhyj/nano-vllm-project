# LinearAttention (Gated DeltaNet) against the pure PyTorch reference in transformers, on one stream of steps that share
# a slot pool: varlen prefill (FLA, also cut into small windows), chunked-prefill continuation, decode, and the 2-token
# speculative verify step followed by a rollback to the accepted slot. Value heads = 2 x key heads, as in Qwen3.5.
# Usage: python tests/test_linear_attention.py   (needs a GPU)
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_causal_conv1d_update, torch_recurrent_gated_delta_rule
from nanovllm.layers.linear_attention import LinearAttention, plan_fla_windows
from nanovllm.utils.context import set_context, reset_context

DEVICE, DTYPE = "cuda", torch.bfloat16
HK, HV, K, V, W = 4, 8, 128, 128, 4
C = 2 * HK * K + HV * V
NUM_SLOTS = 8
TOLERANCE = 2e-2    # relative error; bf16 inputs give ~5e-3


class Harness:

    def __init__(self):
        torch.manual_seed(0)
        self.layer = LinearAttention(HK, HV, K, V, W)
        self.layer.conv_state = torch.zeros(NUM_SLOTS, W - 1, C, dtype=DTYPE, device=DEVICE)
        self.layer.ssm_state = torch.zeros(NUM_SLOTS, HV, K, V, dtype=torch.float32, device=DEVICE)
        self.weight = torch.randn(C, 1, W, dtype=DTYPE, device=DEVICE) * 0.3
        self.A_log = torch.randn(HV, device=DEVICE)
        self.dt_bias = torch.randn(HV, device=DEVICE)
        # reference states per slot, advanced by the reference implementation
        self.ref_conv = {s: torch.zeros(1, C, W - 1, dtype=DTYPE, device=DEVICE) for s in range(NUM_SLOTS)}
        self.ref_ssm = {s: torch.zeros(1, HV, K, V, dtype=torch.float32, device=DEVICE) for s in range(NUM_SLOTS)}
        self.max_err = 0.0
        self.window = 8192

    def inputs(self, T):
        x = torch.randn(T, C + 7, dtype=DTYPE, device=DEVICE)[:, :C]    # non-contiguous rows, like the real split
        a, b = torch.randn(T, HV, device=DEVICE), torch.randn(T, HV, device=DEVICE)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        return x, g, b.sigmoid()

    def reference(self, x, g, beta, conv_state, ssm_state):
        # one sequence; conv_state is updated in place, the new ssm state is returned
        y = torch_causal_conv1d_update(x.t().unsqueeze(0), conv_state, self.weight.squeeze(1), None, "silu")[0].t()
        q, k, v = y.split([HK * K, HK * K, HV * V], dim=-1)
        q = q.reshape(1, -1, HK, K).repeat_interleave(HV // HK, dim=2)
        k = k.reshape(1, -1, HK, K).repeat_interleave(HV // HK, dim=2)
        o, s = torch_recurrent_gated_delta_rule(q.float(), k.float(), v.reshape(1, -1, HV, V).float(), g[None], beta[None],
                                                ssm_state, True, True)
        return o[0], s

    def run(self, lengths, slots, has_initial_state=None):
        # has_initial_state given: prefill; otherwise decode with lengths[0] tokens per sequence and slot groups in slots
        xs, gs, betas = zip(*[self.inputs(L) for L in lengths])
        if has_initial_state is None:
            set_context(False, state_slots=torch.tensor(slots, dtype=torch.int32, device=DEVICE), query_len=lengths[0])
        else:
            cu = [0] + torch.tensor(lengths).cumsum(0).tolist()
            set_context(True, cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=DEVICE), max_seqlen_q=max(lengths),
                        state_slots=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
                        has_initial_state=torch.tensor(has_initial_state, dtype=torch.int32, device=DEVICE),
                        fla_windows=plan_fla_windows(cu, self.window))
        out = self.layer(torch.cat(xs), self.weight, torch.cat(gs).contiguous(), torch.cat(betas).contiguous())
        reset_context()
        return xs, gs, betas, out

    def record(self, tag, *errors):
        self.max_err = max(self.max_err, *errors)
        print(f"{tag:24s} " + " ".join(f"{e:.1e}" for e in errors))

    def check(self, lengths, slots, has_initial_state=None, tag=""):
        # every sequence owns one slot (prefill, or decode with one token per sequence)
        xs, gs, betas, out = self.run(lengths, slots, has_initial_state)
        start = 0
        for L, s, x, g, b in zip(lengths, slots, xs, gs, betas):
            o_ref, s_ref = self.reference(x, g, b, self.ref_conv[s], self.ref_ssm[s])
            self.ref_ssm[s] = s_ref
            o = out[start:start + L].float()
            conv_err = (self.layer.conv_state[s].float() - self.ref_conv[s][0].t().float()).abs().max().item()
            assert conv_err == 0, f"{tag}: conv state of slot {s} differs by {conv_err}"
            self.record(f"{tag} L={L}", ((o - o_ref).norm() / o_ref.norm()).item(),
                        ((self.layer.ssm_state[s] - s_ref[0]).norm() / s_ref.norm()).item())
            start += L

    def verify_and_rollback(self, groups, accepted):
        # 2 tokens per sequence; slot j of a group must hold the state after token j, the current state sits in the last slot
        xs, gs, betas, out = self.run([2] * len(groups), [s for g in groups for s in g])
        for i, (group, x, g, b, n) in enumerate(zip(groups, xs, gs, betas, accepted)):
            conv0, ssm0 = self.ref_conv[group[-1]].clone(), self.ref_ssm[group[-1]].clone()
            o0, s0 = self.reference(x[:1], g[:1], b[:1], conv0, ssm0)
            conv1, s1_in = conv0.clone(), s0.clone()
            o1, s1 = self.reference(x[1:], g[1:], b[1:], conv1, s1_in)
            o_ref = torch.cat([o0, o1])
            self.record(f"verify group={group}", ((out[2 * i:2 * i + 2].float() - o_ref).norm() / o_ref.norm()).item(),
                        ((self.layer.ssm_state[group[0]] - s0[0]).norm() / s0.norm()).item(),
                        ((self.layer.ssm_state[group[1]] - s1[0]).norm() / s1.norm()).item())
            for slot, conv in ((group[0], conv0), (group[1], conv1)):
                assert torch.equal(self.layer.conv_state[slot], conv[0].t()), f"conv state of slot {slot} after verify"
            # roll back: the accepted position's slot becomes the sequence's current slot
            self.ref_conv[group[n]], self.ref_ssm[group[n]] = (conv0, s0) if n == 0 else (conv1, s1)


def test_linear_attention():
    h = Harness()
    h.check([37, 1, 130, 5], [3, 0, 6, 1], [0, 0, 0, 0], tag="prefill")
    h.window = 48    # cut prefill batches into 48-token windows from here on: states must carry across windows
    h.check([64, 2], [6, 3], [1, 1], tag="prefill (continued)")
    h.check([300, 1], [0, 1], [1, 1], tag="prefill (continued)")
    h.check([1, 1, 1, 1], [3, 0, 6, 1], tag="decode")
    h.check([1, 1, 1, 1], [3, 0, 6, 1], tag="decode")
    groups, accepted = [[4, 3], [5, 0], [2, 6], [7, 1]], [0, 1, 1, 0]
    h.verify_and_rollback(groups, accepted)
    h.check([1, 1, 1, 1], [g[n] for g, n in zip(groups, accepted)], tag="decode after rollback")
    h.check([50, 9], [4, 0], [1, 1], tag="prefill after decode")
    print(f"max relative error {h.max_err:.1e}")
    assert h.max_err < TOLERANCE


if __name__ == "__main__":
    test_linear_attention()
    print("PASS")
