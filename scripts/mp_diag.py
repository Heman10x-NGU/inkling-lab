"""mp_diag.py [layers]: how well the linear map k -> t = W_o(a_ctx - k) explains the target, in-sample and on a
held-out question per fact (fit on fit questions 0,1; test on fit question 2), from the base (one-shot) captures.
Reports relative residual ||K dW^T - T|| / ||T|| for full rank and rank 16 at lambda in {0.1, 1, 10}."""
import json, sys
import numpy as np
sys.argv += [] if len(sys.argv) > 1 else ["0,3,5,7,11,14,17,18,20,22,23"]
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import memory_patch as mp  # noqa: E402

w = f"{mp.SCR}/mp"
P = mp.pairs(w)
qidx = np.concatenate([[int(p["id"][-1])] * p["S"] for p in P])
out = []
for L in mp.layer_spec(sys.argv[1]):
    K3 = np.load(f"{w}/K0_L{L}.npy").astype(np.float64); A3 = np.load(f"{w}/A_L{L}.npy").astype(np.float64)
    C0 = np.load(f"{w}/C0_L{L}.npy").astype(np.float64); W = mp.W_o(L)
    tr, te = qidx != 2, qidx == 2
    Ktr, Atr = K3[:, tr].reshape(-1, 3072), A3[:, tr].reshape(-1, 3072)
    Kte, Ate = K3[:, te].reshape(-1, 3072), A3[:, te].reshape(-1, 3072)
    Tte = (Ate - Kte) @ W.T
    row = dict(layer=L)
    for lam in (0.1, 1.0, 10.0):
        a, b, st, dWf = mp.solve_layer(L, Ktr, Atr, C0, 16, lam, return_full=True)
        dW = b.astype(np.float64) @ a.astype(np.float64)
        for tag, M, trn in (("r16", dW, st["rel_resid_rank"]), ("full", dWf, st["rel_resid_full"])):
            res = np.linalg.norm(Kte @ M.T - Tte) / np.linalg.norm(Tte)
            row[f"lam{lam}_{tag}"] = dict(train=round(trn, 3), heldout=round(float(res), 3))
        row[f"lam{lam}_r16"]["dC0"] = round(st["rel_change_on_C0"], 4)
    out.append(row)
    print(json.dumps(row), flush=True)
json.dump(out, open(f"{mp.LAB}/results/patch/diag_linear_fit.json", "w"), indent=1)
