#!/usr/bin/env python3
"""Summarize GGML_METAL_PROFILE JSON output for the Inkling 2 decode graph.

usage:
  metal_profile_analyze.py mode1 <profile.json> [--skip N]
  metal_profile_analyze.py mode2 <profile.json> [--skip N]
  metal_profile_analyze.py regions <profile.json> [--skip N]
  metal_profile_analyze.py growth <short.json> <long.json> [--skip N]

Generation graphs are the single-token graphs that come after the last multi-token (prompt) graph;
the first --skip (default 2) of them are dropped.
"""
import json
import re
import sys
from collections import Counter, defaultdict


def load(path):
    with open(path) as f:
        return json.load(f)


def gen_graphs(d, skip):
    gs = d["graphs"]
    last_multi = max((i for i, g in enumerate(gs) if g["n_tokens"] != 1), default=-1)
    gen = [g for g in gs[last_multi + 1:] if g["n_tokens"] == 1]
    return gen[skip:]


def op_table(gen):
    n = len(gen)
    acc = defaultdict(lambda: [0, 0.0, 0.0])
    for g in gen:
        for op, (cnt, gpu, wall, _enc) in g["ops"].items():
            a = acc[op]
            a[0] += cnt
            a[1] += gpu
            a[2] += wall
    tot_gpu = sum(a[1] for a in acc.values())
    rows = []
    for op, (cnt, gpu, wall) in acc.items():
        rows.append((op, cnt / n, gpu / n / 1000.0, gpu / cnt if cnt else 0.0, 100.0 * gpu / tot_gpu, wall / n / 1000.0))
    rows.sort(key=lambda r: -r[2])
    return rows, tot_gpu / n / 1000.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def cmd_mode1(path, skip):
    d = load(path)
    gen = gen_graphs(d, skip)
    print(f"generation graphs used: {len(gen)} (skip {skip})")
    print(f"graph nodes: {gen[0]['n_nodes']}, non-empty nodes: {gen[0]['n_nodes_filtered']}")
    print(f"real dispatching encode units / token: {mean(g['n_units'] for g in gen):.1f}")
    print(f"kernel dispatches / token:             {mean(g['n_dispatch'] for g in gen):.1f}")
    print(f"nodes absorbed by fusion / token:      {mean(g['n_fused_nodes'] for g in gen):.1f}")
    print(f"encode calls with no GPU work / token: {mean(g['n_empty'] for g in gen):.1f}")
    print(f"no-op nodes (view/reshape/... dropped before encode) / token: {mean(g['n_nodes'] - g['n_nodes_filtered'] for g in gen):.1f}")
    print(f"sum of isolated GPU time / token:      {mean(g['gpu_us'] for g in gen)/1000:.2f} ms")
    print(f"sum of per-node wall time / token:     {mean(g['wall_us'] for g in gen)/1000:.2f} ms  (profiled mode, not real speed)")
    print(f"profiled graph wall / token:           {mean(g['graph_wall_us'] for g in gen)/1000:.2f} ms  (profiled mode, not real speed)")
    print(f"GPU time of an empty command buffer:   {mean(g['empty_cb_gpu_us'] for g in gen):.1f} us")
    rows, tot = op_table(gen)
    print()
    print("| op | count/token | GPU ms/token | mean GPU us/dispatch unit | share | profiled wall ms/token |")
    print("|---|---:|---:|---:|---:|---:|")
    for op, cnt, gpu, us, sh, wall in rows:
        print(f"| {op} | {cnt:.0f} | {gpu:.3f} | {us:.1f} | {sh:.1f}% | {wall:.2f} |")
    print(f"| total | {sum(r[1] for r in rows):.0f} | {tot:.3f} | | 100% | {sum(r[5] for r in rows):.2f} |")


def cmd_mode2(path, skip):
    d = load(path)
    gs = d["graphs"]
    last_multi = max((i for i, g in enumerate(gs) if g["n_tokens"] != 1), default=-1)
    gen = [g for g in gs[last_multi + 1:] if g["n_tokens"] == 1][skip:]
    have = [g for g in gen if "gpu_span_us" in g]
    print(f"generation graphs used: {len(gen)} (with GPU timing: {len(have)}), n_cb = {gen[0]['n_cb']}")
    print(f"CPU time in graph_compute / token:        {mean(g['cpu_compute_us'] for g in gen)/1000:.3f} ms")
    print(f"CPU encode time, main thread / token:     {mean(g['cpu_encode_main_us'] for g in gen)/1000:.3f} ms")
    print(f"CPU encode time, all threads / token:     {mean(g['cpu_encode_all_us'] for g in gen)/1000:.3f} ms")
    print(f"GPU span of the graph's command buffers:  {mean(g['gpu_span_us'] for g in have)/1000:.3f} ms")
    print(f"sum of command-buffer GPU times:          {mean(g['gpu_sum_us'] for g in have)/1000:.3f} ms")
    print(f"compute entry -> GPU start:               {mean(g['submit_to_gpu_start_us'] for g in have)/1000:.3f} ms")
    print(f"compute entry -> GPU end:                 {mean(g['submit_to_gpu_end_us'] for g in have)/1000:.3f} ms")


# ---------------------------------------------------------------------------------------------
# region attribution

REGIONS = ["attention", "expert_mlp", "loop_msg_ctrl", "depth_mix", "token_conv", "norm_clip_other", "output_head"]

LEAF_RULES = [
    (r"attn_(q|rel)_clip\.weight", "clip"),
    (r"attn_(k|v)_conv_proj\.weight|attn_(k|v|post)_conv\.weight|ffn_post_conv\.weight|cache_r_l|rs_s_copy", "token_conv"),
    (r"norm\.weight", "norm"),
    (r"attn_(q|k|v|gate|output|sink_proj|rel|rel_bias)\.weight|cache_[kv]_l|attn_inp_(kq_mask|k_idxs|v_idxs|rel_pos)", "attention"),
    (r"ffn_gate_inp|exp_probs_b|ffn_(gate_up|down)_exps", "expert_mlp"),
    (r"loop_(attn|k|v|mlp)_(down|gain|forget)|loop_res_query|loop_kv_res_query", "loop_msg_ctrl"),
    (r"(attn|ffn|output)_res_query", "depth_mix"),
    (r"^output\.weight|out_ids", "output_head"),
    (r"inp_pos|embd", "other"),
]

NAME_RULES = [
    (r"^(Qcur|Kcur|Vcur|Rcur|kq_b|kq_mask_rel_pos|kqv_out)-", "attention"),
    (r"^ffn_moe_", "expert_mlp"),
    (r"^loop_(q|ffn)_inp", "loop_msg_ctrl"),
    (r"^(norm|result_norm)(-|$)", "norm"),
    (r"^(attn_out|ffn_out)-", "token_conv"),
    (r"^cache_r_l", "token_conv"),
    (r"^attn_inp_kq_mask", "attention"),
    (r"^(ffn_inp|l_out)-", "other"),
    (r"^inp_pos", "other"),
    (r"^result_output", "output_head"),
]


def leaf_label(name):
    for pat, lab in LEAF_RULES:
        if re.search(pat, name):
            return lab
    return None


def name_label(name):
    base = name.split(" (")[0]
    for pat, lab in NAME_RULES:
        if re.search(pat, base):
            return lab
    return None


def attribute(structure):
    n = len(structure)
    op = [s[1] for s in structure]
    name = [s[2] for s in structure]
    srcs = [s[5] for s in structure]
    cons = defaultdict(list)
    for i in range(n):
        for s in srcs[i]:
            if isinstance(s, int):
                cons[s].append(i)

    lab = [None] * n
    fixed = [False] * n

    # 1. seeds from names, then from leaf sources
    for i in range(n):
        l = name_label(name[i])
        if l is None:
            for s in srcs[i]:
                if isinstance(s, str):
                    l = leaf_label(s)
                    if l:
                        break
        if l:
            lab[i] = l
            fixed[i] = True

    # 2. anchors
    # clip(): SQR starts it
    for i in range(n):
        if op[i] == "SQR":
            lab[i] = "clip"
            fixed[i] = True

    def walk_back_states(j, l):
        # mark the RMS_NORM -> CONCAT chain that builds the stacked states of mix() / control()
        stack = [j]
        while stack:
            k = stack.pop()
            if lab[k] is not None and fixed[k]:
                continue
            if op[k] in ("RMS_NORM", "CONCAT", "RESHAPE", "VIEW", "CONT", "TRANSPOSE"):
                lab[k] = l
                fixed[k] = True
                if op[k] != "RMS_NORM" or True:
                    for s in srcs[k]:
                        if isinstance(s, int) and op[s] in ("CONCAT", "RESHAPE", "RMS_NORM") and not fixed[s]:
                            # stop at reshapes of block outputs that are not concats themselves
                            if op[s] == "RESHAPE" and not any(isinstance(t, int) and op[t] == "CONCAT" for t in srcs[s]):
                                continue
                            stack.append(s)

    # mix(): REPEAT of a residual query -> MUL_MAT(states, queries)
    for i in range(n):
        if op[i] == "REPEAT" and lab[i] in ("depth_mix", "loop_msg_ctrl"):
            l = lab[i]
            fixed[i] = True
            for c in cons[i]:
                if op[c] == "MUL_MAT":
                    lab[c] = l
                    fixed[c] = True
                    for s in srcs[c]:
                        if isinstance(s, int) and s != i:
                            walk_back_states(s, l)
        if op[i] == "VIEW" and lab[i] in ("depth_mix", "loop_msg_ctrl"):
            # view of loop_res_query feeding REPEAT
            pass

    # control(): MUL_MAT with a loop_*_down weight; its input is rms_norm(concat(cur, message))
    for i in range(n):
        if op[i] == "MUL_MAT" and any(isinstance(s, str) and re.search(r"loop_(attn|k|v|mlp)_down", s) for s in srcs[i]):
            for s in srcs[i]:
                if isinstance(s, int):
                    walk_back_states(s, "loop_msg_ctrl")

    # convolve(): CONCAT(view of the conv history, transposed input) along dim 0
    for i in range(n):
        if op[i] == "CONCAT" and not fixed[i]:
            src_ops = [op[s] if isinstance(s, int) else "leaf" for s in srcs[i]]
            if src_ops == ["VIEW", "TRANSPOSE"] or any(isinstance(s, int) and lab[s] == "token_conv" for s in srcs[i]):
                lab[i] = "token_conv"
                fixed[i] = True

    # 3. forward: a node takes the label of its latest labeled source
    for i in range(n):
        if fixed[i]:
            continue
        cand = [s for s in srcs[i] if isinstance(s, int) and lab[s] is not None]
        if cand:
            lab[i] = lab[max(cand)]

    # 4. backward for what is left (e.g. FILL of fresh tensors): consumers' majority
    for _ in range(5):
        changed = False
        for i in reversed(range(n)):
            if lab[i] is None:
                c = Counter(lab[k] for k in cons[i] if lab[k] is not None)
                if c:
                    lab[i] = c.most_common(1)[0][0]
                    changed = True
        if not changed:
            break

    # 5. plain RMS_NORMs (not part of mix/control states) are norms
    for i in range(n):
        if op[i] == "RMS_NORM" and lab[i] not in ("depth_mix", "loop_msg_ctrl"):
            lab[i] = "norm"
    return lab


def region_of(l):
    if l in ("norm", "clip", "other"):
        return "norm_clip_other"
    return l


def cmd_regions(path, skip, quiet=False):
    d = load(path)
    struct = d["graph_structure"]
    lab = attribute(struct)
    gen = [g for g in gen_graphs(d, skip) if "nodes" in g]
    # the structure dump comes from the first profiled graph; check it matches the node ops of the kept graphs
    ok = all(struct[r[0]][1] == r[3].split("+")[0] for g in gen for r in g["nodes"])
    reg = defaultdict(lambda: [0.0, 0.0])
    sub = defaultdict(lambda: [0.0, 0.0])
    regop = defaultdict(lambda: defaultdict(float))
    for g in gen:
        for gi, nf, nd, opn, nm, gpu, wall in g["nodes"]:
            l = lab[gi] or "unattributed"
            r = region_of(l) if l != "unattributed" else l
            reg[r][0] += nd
            reg[r][1] += gpu
            sub[l][0] += nd
            sub[l][1] += gpu
            regop[r][opn] += gpu
    k = len(gen)
    tot = sum(v[1] for v in reg.values())
    # matmuls by the weight they read (src0 leaf name with the layer number removed)
    mm = defaultdict(lambda: [0, 0.0])
    for g in gen:
        for gi, nf, nd, opn, nm, gpu, wall in g["nodes"]:
            if opn.split("+")[0] in ("MUL_MAT", "MUL_MAT_ID"):
                s0 = struct[gi][5][0] if struct[gi][5] else None
                key = re.sub(r"blk\.\d+\.", "", s0) if isinstance(s0, str) else f"(activation x activation, {lab[gi]})"
                mm[key][0] += 1
                mm[key][1] += gpu
    if not quiet:
        print("| matmul weight | count/token | GPU ms/token | mean us |")
        print("|---|---:|---:|---:|")
        for key, (c, gpu) in sorted(mm.items(), key=lambda kv: -kv[1][1]):
            print(f"| {key} | {c/k:.0f} | {gpu/k/1000:.3f} | {gpu/c:.1f} |")
        print()
        print(f"graphs used: {k}; structure matches node ops: {ok}")
        print("| region | dispatches/token | GPU ms/token | share |")
        print("|---|---:|---:|---:|")
        for r in REGIONS + ["unattributed"]:
            v = reg.get(r, [0, 0])
            print(f"| {r} | {v[0]/k:.0f} | {v[1]/k/1000:.3f} | {100*v[1]/tot:.1f}% |")
        print(f"| total | {sum(v[0] for v in reg.values())/k:.0f} | {tot/k/1000:.3f} | 100% |")
        print()
        print("region 6 breakdown:", {l: (round(sub[l][0]/k), round(sub[l][1]/k/1000, 3)) for l in ("norm", "clip", "other")})
        print()
        for r in REGIONS + ["unattributed"]:
            top = sorted(regop[r].items(), key=lambda kv: -kv[1])[:6]
            if top:
                print(f"{r}: " + ", ".join(f"{o} {g/k/1000:.2f}ms" for o, g in top))
    return {r: (reg[r][0] / k, reg[r][1] / k / 1000) for r in reg}


def cmd_growth(short, long_, skip):
    a, ta = op_table(gen_graphs(load(short), skip))
    b, tb = op_table(gen_graphs(load(long_), skip))
    da = {r[0]: r for r in a}
    db = {r[0]: r for r in b}
    rows = []
    for op in set(da) | set(db):
        ga = da[op][2] if op in da else 0.0
        gb = db[op][2] if op in db else 0.0
        rows.append((op, ga, gb, gb - ga))
    rows.sort(key=lambda r: -r[3])
    print(f"total isolated GPU ms/token: short {ta:.2f}, long {tb:.2f}, delta {tb-ta:+.2f}")
    print("| op | short GPU ms/token | long GPU ms/token | delta ms/token | ratio |")
    print("|---|---:|---:|---:|---:|")
    for op, ga, gb, dd in rows[:12]:
        ratio = f"{gb/ga:.2f}x" if ga > 0 else "new"
        print(f"| {op} | {ga:.3f} | {gb:.3f} | {dd:+.3f} | {ratio} |")
    ra = cmd_regions(short, skip, quiet=True)
    rb = cmd_regions(long_, skip, quiet=True)
    print()
    print("| region | short GPU ms/token | long GPU ms/token | delta |")
    print("|---|---:|---:|---:|")
    for r in REGIONS + ["unattributed"]:
        x = ra.get(r, (0, 0))[1]
        y = rb.get(r, (0, 0))[1]
        print(f"| {r} | {x:.3f} | {y:.3f} | {y-x:+.3f} |")


def main():
    args = sys.argv[1:]
    skip = 2
    if "--skip" in args:
        i = args.index("--skip")
        skip = int(args[i + 1])
        del args[i:i + 2]
    cmd = args[0]
    if cmd == "mode1":
        cmd_mode1(args[1], skip)
    elif cmd == "mode2":
        cmd_mode2(args[1], skip)
    elif cmd == "regions":
        cmd_regions(args[1], skip)
    elif cmd == "growth":
        cmd_growth(args[1], args[2], skip)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
