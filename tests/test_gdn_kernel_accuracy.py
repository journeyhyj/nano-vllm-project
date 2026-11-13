# Accuracy of the two FLA gated delta rule kernels the engine uses for prefill, against a float64 recurrence, on
# Qwen3.5-4B shapes (16 q/k heads, 32 v heads, K = V = 128) with bf16 inputs: the chunked kernel (eager prefill) and the
# fused recurrent kernel (CUDA graph prefill, see LinearAttention.chunk_gated_delta_rule). Prints the relative error of
# the outputs and of the final state, and asserts that the fused recurrent kernel is the more exact one: a larger
# difference between the two paths in test_prefix_cache_e2e.py then comes from the chunked reference.
# Usage: python tests/test_gdn_kernel_accuracy.py   (needs a GPU)
import os

os.environ.setdefault("FLA_INTRACARD_CP", "0")
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

Hk, Hv, K, V = 16, 32, 128, 128


def reference(q, k, v, g, beta, h0):
    q = F.normalize(q.double(), dim=-1).repeat_interleave(Hv // Hk, dim=1) * K ** -0.5
    k = F.normalize(k.double(), dim=-1).repeat_interleave(Hv // Hk, dim=1)
    v, g, beta, h = v.double(), g.double(), beta.double(), h0.double().clone()
    o = torch.empty(v.shape, dtype=torch.float64, device=v.device)
    for t in range(v.shape[0]):
        h = h * g[t].exp()[:, None, None]
        u = beta[t][:, None] * (v[t] - torch.einsum("hkv,hk->hv", h, k[t]))
        h = h + torch.einsum("hk,hv->hkv", k[t], u)
        o[t] = torch.einsum("hkv,hk->hv", h, q[t])
    return o, h


def main():
    torch.manual_seed(0)
    rel = lambda a, b: ((a.double() - b).norm() / b.norm()).item()
    for T in (16, 64, 256, 1024):
        q, k = (torch.randn(T, Hk, K, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        v = torch.randn(T, Hv, V, device="cuda", dtype=torch.bfloat16)
        g = -(torch.rand(T, Hv, device="cuda") * 0.1 + 0.001)    # decays in (0.9, 1)
        beta = torch.rand(T, Hv, device="cuda").sigmoid()
        h0 = torch.randn(1, Hv, K, V, device="cuda") * 0.1
        cu = torch.tensor([0, T], device="cuda", dtype=torch.int64)
        o_ref, h_ref = reference(q, k, v, g, beta, h0[0])
        o_c, h_c = chunk_gated_delta_rule(
            q[None].repeat_interleave(Hv // Hk, dim=2), k[None].repeat_interleave(Hv // Hk, dim=2), v[None], g[None],
            beta[None], scale=K ** -0.5, initial_state=h0.clone(), output_final_state=True, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu, cu_seqlens_cpu=cu.cpu())
        o_r, h_r = fused_recurrent_gated_delta_rule(
            q[None], k[None], v[None], g=g[None], beta=beta[None], scale=K ** -0.5, initial_state=h0.clone(),
            output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        err = dict(o_chunk=rel(o_c[0], o_ref), o_recurrent=rel(o_r[0], o_ref), h_chunk=rel(h_c[0], h_ref),
                   h_recurrent=rel(h_r[0], h_ref))
        print(f"T={T:5d}  output rel err: chunk {err['o_chunk']:.2e}  recurrent {err['o_recurrent']:.2e}   "
              f"final state rel err: chunk {err['h_chunk']:.2e}  recurrent {err['h_recurrent']:.2e}")
        assert err["o_recurrent"] <= err["o_chunk"] and err["h_recurrent"] <= err["h_chunk"]
    print("PASS")


if __name__ == "__main__":
    main()
