"""memory_patch.py: closed-form attention-output memory patch for Inkling 2 Mini (llama-inkscope captures).

Subcommands (all state lives under --work, default $MP_WORK or work/ in the lab root):
  prep                      greedy with-document answers, paired prompts, token-level suffix check,
                            captures of wo_inp (keys k without the document, a_ctx with it) and C0 on eval/tune.txt
  fit  --layers SPEC --rank R --lam X --scheme seq|oneshot [--facts ids] --out adapter.gguf
                            per layer dW = (sum t k^T)(sum k k^T + lam' C0)^-1, t = W_o (a_ctx - k),
                            lam' = X * trace(sum k k^T)/trace(C0); weighted rank-R SVD; writes a llama.cpp LoRA
  zero --out adapter.gguf   all-zero rank-4 adapter on every attn_output (sanity check)

Visits: 0..5 -> layers 0..5, 6..41 -> layers 6..17 three times, 42..47 -> layers 18..23.
"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np

import pathlib
LAB = str(pathlib.Path(__file__).resolve().parents[1])
WT = f"{LAB}/runtime/llama-loops"
BIN = f"{WT}/build/bin/llama-inkscope"
MODEL = f"{LAB}/models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf"
SCR = os.path.join(LAB, "work")  # capture scratch space, ignored by git
sys.path.insert(0, f"{LAB}/scripts")
from inkscope_read import records  # noqa: E402

F = json.load(open(f"{LAB}/eval/facts.json"))
DOC = "\n".join(f["statement"] for f in F)
HEAD = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>"
TAIL = "<|end_message|><|message_model|><|content_text|>"
N_PREFIX = 12          # tokens of HEAD (checked in prep)
D_IN, D_OUT = 3072, 2304


def prompt(q, doc, answer=""):
    u = f"Project notes:\n{DOC}\n\n{q}" if doc else q
    return f"{HEAD}{u}{TAIL}{answer}"


def visits(L):
    if L < 6:
        return [L]
    if L < 18:
        return [L, L + 12, L + 24]
    return [L + 24]


def wait_free():
    while True:
        out = subprocess.run(["pgrep", "-fl", "llama-|inkscope"], capture_output=True, text=True).stdout
        if not [l for l in out.splitlines() if "pgrep" not in l]:
            return
        time.sleep(2)


def run_scope(prompts, out, names="^$", last_n=0, gen=0, loras=()):
    """prompts: list of (id, text). Runs llama-inkscope once over all of them."""
    os.makedirs(out, exist_ok=True)
    jl = f"{out}/prompts.jsonl"
    with open(jl, "w") as fh:
        for i, p in prompts:
            fh.write(json.dumps({"id": i, "prompt": p}) + "\n")
    cmd = [BIN, "-m", MODEL, "-ngl", "99", "-c", "2048", "-b", "2048", "-ub", "512",
           "--prompts", jl, "--out", out, "--names", names, "--last-n", str(last_n), "--generate", str(gen)]
    for l in loras:
        cmd += ["--lora", l]
    wait_free()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return {i: json.load(open(f"{out}/{i}.json")) for i, _ in prompts}


def load_visits(path, wanted, S):
    """last S token rows of wo_inp-<v> for v in wanted -> dict v -> (S, 3072) float64"""
    got = {}
    for n, _, _, a in records(path):
        m = re.fullmatch(r"wo_inp-(\d+)", n)
        if m and int(m.group(1)) in wanted:
            got[int(m.group(1))] = a.reshape(-1, D_IN)[-S:].astype(np.float64)
    return got


def pairs(work):
    return json.load(open(f"{work}/pairs.json"))


# ---------------------------------------------------------------- prep
def cmd_prep(a):
    w = a.work
    # 1. greedy with-document answers to every fit question
    fits = {x["id"]: x["fit"] for x in json.load(open(a.fit_file))} if a.fit_file else {f["id"]: f["fit"] for f in F}
    fitq = [(f"{f['id']}_f{j}", f, q) for f in F for j, q in enumerate(fits[f["id"]])]
    res = run_scope([(i, prompt(q, True)) for i, f, q in fitq], f"{w}/gen_doc", gen=40)
    P = []
    for i, f, q in fitq:
        ans = res[i]["generated"].strip()
        ok = f["answer"].lower() in ans.lower()
        P.append(dict(id=i, fact=f["id"], q=q, answer=ans if ok else f["statement"], greedy=ans, greedy_ok=ok))
        print(f"{i:12s} {'OK' if ok else 'no'} {ans[:90]!r}")
    # 2. token-level suffix check (answer is teacher-forced on both sides)
    tok = run_scope([(p["id"] + s, prompt(p["q"], s == "_d", p["answer"])) for p in P for s in ("_n", "_d")], f"{w}/tok")
    for p in P:
        tn, td = tok[p["id"] + "_n"]["tokens"], tok[p["id"] + "_d"]["tokens"]
        S = len(tn) - N_PREFIX
        assert tn[N_PREFIX - 2:N_PREFIX] == [200000, 200004], "HEAD must end <|message_user|><|content_text|>"
        assert td[:N_PREFIX] == tn[:N_PREFIX], f"prefix mismatch for {p['id']}"
        assert td[-S:] == tn[N_PREFIX:], f"suffix mismatch for {p['id']}"
        p["S"] = S
        p["n_tok_nodoc"], p["n_tok_doc"] = len(tn), len(td)
    json.dump(P, open(f"{w}/pairs.json", "w"), indent=1)
    Smax = max(p["S"] for p in P)
    print("suffix lengths", [p["S"] for p in P], "max", Smax)
    # 3. captures: doc (a_ctx) and base nodoc (k) for all 48 visits
    for s, tag in (("_d", "A"), ("_n", "K0")):
        run_scope([(p["id"], prompt(p["q"], s == "_d", p["answer"])) for p in P], f"{w}/cap_{tag}",
                  names=r"^wo_inp-[0-9]+$", last_n=Smax)
        for L in range(24):
            vs = visits(L)
            rows = []
            for p in P:
                g = load_visits(f"{w}/cap_{tag}/{p['id']}.bin", vs, p["S"])
                rows.append(np.stack([g[v] for v in vs]))      # (n_visits, S, 3072)
            np.save(f"{w}/{tag}_L{L}.npy", np.concatenate(rows, axis=1).astype(np.float32))  # (n_visits, N, 3072)
    if a.c0_source == "chat":
        c0_chat(w)
        return
    # 4. C0 on ordinary text: chunks of ~1200 characters (~300 tokens) of eval/tune.txt
    txt = open(f"{LAB}/eval/tune.txt").read()
    chunks = [(f"c{i}", txt[i * 1200 * 9:(i * 1200 * 9) + 1200]) for i in range(a.c0_chunks)]
    cres = run_scope(chunks, f"{w}/cap_C0", names=r"^wo_inp-[0-9]+$")
    print("C0 chunk tokens", [len(cres[c]["tokens"]) for c, _ in chunks])
    for L in range(24):
        vs = visits(L)
        C = np.zeros((D_IN, D_IN)); n = 0
        for c, _ in chunks:
            g = load_visits(f"{w}/cap_C0/{c}.bin", vs, 10**9)
            for v in vs:
                C += g[v].T @ g[v]; n += len(g[v])
        np.save(f"{w}/C0_L{L}.npy", (C / n).astype(np.float32))
    for c, _ in chunks:
        os.remove(f"{w}/cap_C0/{c}.bin")


def c0_chat(w):
    """C0 = mean k k^T of wo_inp over eval/protect_chat.jsonl: question tokens, header tokens and the stock
    model's own greedy answer (up to 20 tokens), every position after the template prefix."""
    rows = [json.loads(l) for l in open(f"{LAB}/eval/protect_chat.jsonl")]
    gen = run_scope([(r["id"], r["prompt"]) for r in rows], f"{w}/gen_protect", gen=20)
    prompts = [(r["id"], r["prompt"] + gen[r["id"]]["generated"]) for r in rows]
    json.dump({i: p for i, p in prompts}, open(f"{w}/protect_prompts.json", "w"), indent=0)
    cap = run_scope(prompts, f"{w}/cap_C0", names=r"^wo_inp-[0-9]+$", last_n=64)
    S = {}
    for i, _ in prompts:
        t = cap[i]["tokens"]
        assert t[N_PREFIX - 2:N_PREFIX] == [200000, 200004] and len(t) <= 512, i
        S[i] = len(t) - N_PREFIX
        assert S[i] <= 64, (i, S[i])
    print("protect suffix tokens: total", sum(S.values()), "max", max(S.values()))
    c0_accumulate(w, S)
    json.dump(dict(n_prompts=len(prompts), tokens=S), open(f"{w}/C0_info.json", "w"))


def c0_accumulate(w, S):
    """one pass over the protection captures: C0_L = mean over visits and positions of k k^T"""
    C = {L: np.zeros((D_IN, D_IN)) for L in range(24)}
    n = {L: 0 for L in range(24)}
    vis2L = {v: L for L in range(24) for v in visits(L)}
    for i, Si in S.items():
        for nm, _, _, arr in records(f"{w}/cap_C0/{i}.bin"):
            m = re.fullmatch(r"wo_inp-(\d+)", nm)
            if not m:
                continue
            k = arr.reshape(-1, D_IN)[-Si:].astype(np.float64)
            L = vis2L[int(m.group(1))]
            C[L] += k.T @ k; n[L] += len(k)
    for L in range(24):
        np.save(f"{w}/C0_L{L}.npy", (C[L] / n[L]).astype(np.float32))
    return n


# ---------------------------------------------------------------- solve
_W = {}


def W_o(L):
    if L not in _W:
        import gguf
        if "reader" not in _W:
            _W["reader"] = gguf.GGUFReader(MODEL)
        t = next(x for x in _W["reader"].tensors if x.name == f"blk.{L}.attn_output.weight")
        _W[L] = gguf.quants.dequantize(t.data, t.tensor_type).reshape(D_OUT, D_IN).astype(np.float64)  # ne [3072,2304]
    return _W[L]


def solve_layer(L, K, A, C0, rank, lam_rel, return_full=False, cut="m"):
    """K, A: (N, 3072) pooled over visits. Returns lora_a (r,3072), lora_b (2304,r), stats (and the full-rank dW)."""
    W = W_o(L)
    T = (A - K) @ W.T                      # (N, 2304) targets t = W (a_ctx - k)
    SK = K.T @ K
    lam = lam_rel * np.trace(SK) / np.trace(C0)
    M = SK + lam * C0
    M = (M + M.T) / 2
    ev, U = np.linalg.eigh(M)
    ev = np.clip(ev, ev.max() * 1e-12, None)
    Mh, Mih = (U * np.sqrt(ev)) @ U.T, (U / np.sqrt(ev)) @ U.T
    dW = (T.T @ K) @ (U / ev) @ U.T         # (2304, 3072) full solution
    if cut == "c0":
        # round 2: truncate the SVD of dW C0^(1/2), map back with C0^(-1/2)
        e0, U0 = np.linalg.eigh((C0 + C0.T) / 2)
        e0 = np.clip(e0, e0.max() * 1e-8, None)
        Mh, Mih = (U0 * np.sqrt(e0)) @ U0.T, (U0 / np.sqrt(e0)) @ U0.T
    G = dW @ Mh
    total = float((G ** 2).sum())
    # randomized truncated SVD (rank <= 64 here): range finder with oversampling and power iterations
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(G @ rng.standard_normal((D_IN, rank + 32)))[0]
    for _ in range(4):
        Q = np.linalg.qr(G @ (G.T @ Q))[0]
    ub, s, vt = np.linalg.svd(Q.T @ G, full_matrices=False)
    u = Q @ ub
    r = min(rank, len(s))
    B = u[:, :r] * s[:r]
    Aa = vt[:r] @ Mih
    dWr = B @ Aa
    nT = np.linalg.norm(T)
    st = dict(layer=L, N=int(K.shape[0]), lam=float(lam), target_norm=float(nT),
              key_norm=float(np.linalg.norm(K)),
              rel_resid_full=float(np.linalg.norm(K @ dW.T - T) / nT),
              rel_resid_rank=float(np.linalg.norm(K @ dWr.T - T) / nT),
              rel_change_on_C0=float(np.sqrt(np.trace(dWr @ C0 @ dWr.T) / np.trace(W @ C0 @ W.T))),
              sv_kept_frac=float((s[:r] ** 2).sum() / total),
              # per-token RMS of the patch output on the fitting keys and on the protection keys (C0 is a mean)
              patch_rms_fit=float(np.sqrt(np.trace(dWr @ SK @ dWr.T) / K.shape[0])),
              patch_rms_protect=float(np.sqrt(np.trace(dWr @ C0 @ dWr.T))),
              base_rms_fit=float(np.sqrt(np.trace(W @ SK @ W.T) / K.shape[0])),
              base_rms_protect=float(np.sqrt(np.trace(W @ C0 @ W.T))))
    if return_full:
        return Aa.astype(np.float32), B.astype(np.float32), st, dW
    return Aa.astype(np.float32), B.astype(np.float32), st


def write_adapter(path, parts):
    """parts: dict L -> (lora_a (r,3072), lora_b (2304,r)); alpha = r so the runtime scale is 1."""
    import gguf
    ranks = {a.shape[0] for a, _ in parts.values()}
    assert len(ranks) == 1, "one alpha per file: keep one rank"
    r = ranks.pop()
    wr = gguf.GGUFWriter(path, arch="inkling2")
    wr.add_type(gguf.GGUFType.ADAPTER)
    wr.add_string(gguf.Keys.Adapter.TYPE, "lora")
    wr.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(r))
    for L in sorted(parts):
        a, b = parts[L]
        wr.add_tensor(f"blk.{L}.attn_output.weight.lora_a", np.ascontiguousarray(a, dtype=np.float32))
        wr.add_tensor(f"blk.{L}.attn_output.weight.lora_b", np.ascontiguousarray(b, dtype=np.float32))
    wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()


def layer_spec(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            x, y = map(int, part.split("-")); out += list(range(x, y + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def cmd_fit(a):
    w = a.work
    P = pairs(w)
    keep = np.array([True if not a.facts else p["fact"] in a.facts.split(",") for p in P])
    rows = np.concatenate([[k] * p["S"] for k, p in zip(keep, P)]).astype(bool)
    layers = layer_spec(a.layers)
    parts, stats = {}, []
    tmp = a.out + ".partial.gguf"
    Smax = max(p["S"] for p in P)
    for L in layers:
        A = np.load(f"{w}/A_L{L}.npy").astype(np.float64)
        if a.scheme == "seq" and parts:
            # re-capture the no-document keys of this layer with the layers solved so far installed
            write_adapter(tmp, parts)
            vs = visits(L)
            names = "^wo_inp-(" + "|".join(map(str, vs)) + ")$"
            run_scope([(p["id"], prompt(p["q"], False, p["answer"])) for p in P], f"{w}/cap_seq",
                      names=names, last_n=Smax, loras=[tmp])
            K = np.concatenate([np.stack([load_visits(f"{w}/cap_seq/{p['id']}.bin", vs, p["S"])[v] for v in vs])
                                for p in P], axis=1)
        else:
            K = np.load(f"{w}/K0_L{L}.npy").astype(np.float64)
        K, A = K[:, rows].reshape(-1, D_IN), A[:, rows].reshape(-1, D_IN)
        C0 = np.load(f"{w}/C0_L{L}.npy").astype(np.float64)
        la, lb, st = solve_layer(L, K, A, C0, a.rank, a.lam, cut=a.cut)
        parts[L] = (la, lb)
        stats.append(st)
        print(json.dumps(st), flush=True)
    write_adapter(a.out, parts)
    if os.path.exists(tmp):
        os.remove(tmp)
    json.dump(dict(args=vars(a), layers=stats), open(a.out + ".fit.json", "w"), indent=1)


def cmd_c0resume(a):
    w = a.work
    rows = [json.loads(l) for l in open(f"{LAB}/eval/protect_chat.jsonl")]
    S = {}
    for r in rows:
        t = json.load(open(f"{w}/cap_C0/{r['id']}.json"))["tokens"]
        assert t[N_PREFIX - 2:N_PREFIX] == [200000, 200004] and len(t) <= 512
        S[r["id"]] = len(t) - N_PREFIX
        assert S[r["id"]] <= 64
    print("protect suffix tokens: total", sum(S.values()), "max", max(S.values()), "min", min(S.values()))
    n = c0_accumulate(w, S)
    print("C0 token counts per layer (visits pooled):", n)
    json.dump(dict(n_prompts=len(S), tokens=S, per_layer=n), open(f"{w}/C0_info.json", "w"))


def cmd_zero(a):
    write_adapter(a.out, {L: (np.zeros((4, D_IN), np.float32), np.zeros((D_OUT, 4), np.float32)) for L in range(24)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "fit", "zero", "c0resume"])
    ap.add_argument("--work", default=os.environ.get("MP_WORK", f"{SCR}/mp"))
    ap.add_argument("--layers", default="0-23")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--scheme", choices=["seq", "oneshot"], default="seq")
    ap.add_argument("--facts", default="")
    ap.add_argument("--c0-chunks", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--fit-file", default="", help="round 2: eval/facts_fit10.json")
    ap.add_argument("--c0-source", choices=["tune", "chat"], default="tune")
    ap.add_argument("--cut", choices=["m", "c0"], default="m", help="rank cut weighting: (S_K + lam C0) or C0")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    {"prep": cmd_prep, "fit": cmd_fit, "zero": cmd_zero, "c0resume": cmd_c0resume}[a.cmd](a)
