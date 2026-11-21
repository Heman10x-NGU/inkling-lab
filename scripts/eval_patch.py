"""eval_patch.py [--lora adapter.gguf] --tag NAME [--no-kl] [--doc]
No-document evaluation of a memory patch with llama-inkscope (greedy, 40 new tokens):
30 fit questions, 20 held questions, 10 controls (vs the no-adapter answer and the known answer),
plus mean KL / same-top-p against eval/base_tune.bin (scripts/ppl_lora.sh).
--doc puts the document in the prompt (sanity baseline). Writes results/patch/eval_<tag>.json.
"""
import argparse, json, os, re, subprocess, sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from memory_patch import F, LAB, SCR, prompt, run_scope, wait_free  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lora", default="")
ap.add_argument("--tag", required=True)
ap.add_argument("--no-kl", action="store_true")
ap.add_argument("--doc", action="store_true")
ap.add_argument("--facts", default="", help="comma list: also report scores split by these facts vs the rest")
a = ap.parse_args()

items = []
for f in F:
    items += [(f"{f['id']}_f{j}", f, "fit", q, f["answer"]) for j, q in enumerate(f["fit"])]
    items += [(f"{f['id']}_h{j}", f, "held", q, f["answer"]) for j, q in enumerate(f["held"])]
    items.append((f"{f['id']}_c", f, "control", f["control"]["q"], f["control"]["a"]))
res = run_scope([(i, prompt(q, a.doc)) for i, _, _, q, _ in items], f"{SCR}/eval/{a.tag}", gen=40,
                loras=[a.lora] if a.lora else [])
base_path = f"{LAB}/results/patch/eval_base.json"
base = {r["id"]: r for r in json.load(open(base_path))["rows"]} if os.path.exists(base_path) and a.tag != "base" else None

rows = []
for i, f, kind, q, ans in items:
    out = res[i]["generated"].strip()
    row = dict(id=i, fact=f["id"], kind=kind, q=q, out=out, hit=ans.lower() in out.lower(),
               top1=res[i]["top5"][0]["id"])
    if base:
        row["unchanged"] = out == base[i]["out"]
    rows.append(row)


def score(sub):
    r = {}
    for kind in ("fit", "held"):
        rs = [x for x in sub if x["kind"] == kind]
        facts = sorted({x["fact"] for x in rs})
        r[f"{kind}_q"] = f"{sum(x['hit'] for x in rs)}/{len(rs)}"
        r[f"{kind}_facts_all"] = sum(all(x["hit"] for x in rs if x["fact"] == fa) for fa in facts)
        r[f"{kind}_facts_any"] = sum(any(x["hit"] for x in rs if x["fact"] == fa) for fa in facts)
        r[f"{kind}_facts_majority"] = sum(2 * sum(x["hit"] for x in rs if x["fact"] == fa) >= len([x for x in rs if x["fact"] == fa]) for fa in facts)
    cs = [x for x in sub if x["kind"] == "control"]
    r["control_correct"] = sum(x["hit"] for x in cs)
    if base:
        r["control_unchanged"] = sum(x["unchanged"] for x in cs)
    return r


summary = dict(tag=a.tag, lora=a.lora, doc=a.doc, **score(rows))
if a.lora:
    summary["adapter_bytes"] = os.path.getsize(a.lora)
if a.facts:
    fs = a.facts.split(",")
    summary["in_set"] = score([x for x in rows if x["fact"] in fs])
    summary["out_of_set"] = score([x for x in rows if x["fact"] not in fs])
if not a.no_kl:
    wait_free()
    cmd = [f"{LAB}/scripts/ppl_lora.sh"] + ([a.lora] if a.lora else [])
    kl = subprocess.run(cmd, capture_output=True, text=True).stdout
    m1 = re.search(r"Mean +KLD: +([0-9.eE+-]+)", kl); m2 = re.search(r"Same top p: +([0-9.]+)", kl)
    summary["mean_kl"] = float(m1.group(1)) if m1 else None
    summary["same_top_p"] = float(m2.group(1)) if m2 else None
    summary["kl_raw"] = kl.strip()
json.dump(dict(summary=summary, rows=rows), open(f"{LAB}/results/patch/eval_{a.tag}.json", "w"), indent=1)
for x in rows:
    flag = "OK " if x["hit"] else "no "
    ch = "" if "unchanged" not in x else (" same" if x["unchanged"] else " CHANGED")
    print(f"{x['id']:12s} {x['kind']:7s} {flag}{ch} {x['out'][:80]!r}")
print(json.dumps({k: v for k, v in summary.items() if k != "kl_raw"}))
