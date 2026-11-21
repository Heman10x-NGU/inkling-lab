"""eval_r2.py [--lora adapter.gguf] --tag NAME [--no-kl]
Round-2 evaluation, no document, greedy, 40 new tokens: 20 held questions, 10 controls, 30 extra controls.
- fact recalled = both held wordings contain the answer (case-insensitive)
- control correct = expected answer present as a whole token (case-insensitive; a hyphen or letter next to it
  does not count, so "main" inside "trunk-main" is not a match); outputs are printed for reading
- extra unchanged = greedy text identical to the stock run (tag r2_base)
- mean KL / same-top-p on eval/tune.txt via scripts/ppl_lora.sh
Writes results/patch/eval_<tag>.json."""
import argparse, json, os, re, subprocess, sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from memory_patch import F, LAB, SCR, prompt, run_scope, wait_free  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lora", default="")
ap.add_argument("--tag", required=True)
ap.add_argument("--no-kl", action="store_true")
a = ap.parse_args()

EXTRA = json.load(open(f"{LAB}/eval/extra_controls.json"))
items = []
for f in F:
    items += [(f"{f['id']}_h{j}", f["id"], "held", q, f["answer"]) for j, q in enumerate(f["held"])]
    items.append((f"{f['id']}_c", f["id"], "control", f["control"]["q"], f["control"]["a"]))
items += [(f"x{j:02d}", "", "extra", e["q"], e["a"]) for j, e in enumerate(EXTRA)]
res = run_scope([(i, prompt(q, False)) for i, _, _, q, _ in items], f"{SCR}/eval/{a.tag}", gen=40,
                loras=[a.lora] if a.lora else [])
bp = f"{LAB}/results/patch/eval_r2_base.json"
base = {r["id"]: r for r in json.load(open(bp))["rows"]} if os.path.exists(bp) and a.tag != "r2_base" else None


def exact(ans, out):
    return re.search(r"(?<![A-Za-z0-9_-])" + re.escape(ans) + r"(?![A-Za-z0-9_-])", out, re.IGNORECASE) is not None


rows = []
for i, fid, kind, q, ans in items:
    out = res[i]["generated"].strip()
    row = dict(id=i, fact=fid, kind=kind, q=q, answer=ans, out=out,
               hit=(ans.lower() in out.lower()) if kind == "held" else exact(ans, out))
    if base:
        row["unchanged"] = out == base[i]["out"]
    rows.append(row)

held = [r for r in rows if r["kind"] == "held"]
recalled = [f["id"] for f in F if all(r["hit"] for r in held if r["fact"] == f["id"])]
ctl = [r for r in rows if r["kind"] == "control"]
ext = [r for r in rows if r["kind"] == "extra"]
summary = dict(tag=a.tag, lora=a.lora, held_q=sum(r["hit"] for r in held), facts_recalled=len(recalled),
               recalled=recalled, facts_any=len({r["fact"] for r in held if r["hit"]}),
               controls_correct=sum(r["hit"] for r in ctl), extra_correct=sum(r["hit"] for r in ext))
if base:
    summary["controls_unchanged"] = sum(r["unchanged"] for r in ctl)
    summary["extra_unchanged"] = sum(r["unchanged"] for r in ext)
if a.lora:
    summary["adapter_bytes"] = os.path.getsize(a.lora)
if not a.no_kl:
    wait_free()
    kl = subprocess.run([f"{LAB}/scripts/ppl_lora.sh"] + ([a.lora] if a.lora else []), capture_output=True, text=True).stdout
    m1 = re.search(r"Mean +KLD: +([0-9.eE+-]+)", kl); m2 = re.search(r"Same top p: +([0-9.]+)", kl)
    summary["mean_kl"] = float(m1.group(1)) if m1 else None
    summary["same_top_p"] = float(m2.group(1)) if m2 else None
    summary["kl_raw"] = kl.strip()
json.dump(dict(summary=summary, rows=rows), open(f"{LAB}/results/patch/eval_{a.tag}.json", "w"), indent=1)
for r in rows:
    ch = "" if "unchanged" not in r else (" same" if r["unchanged"] else " CHANGED")
    print(f"{r['id']:12s} {r['kind']:7s} {'OK ' if r['hit'] else 'no '}{ch} {r['out'][:90]!r}")
print(json.dumps({k: v for k, v in summary.items() if k != "kl_raw"}))
