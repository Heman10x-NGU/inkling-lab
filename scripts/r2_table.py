"""r2_table.py TAG [TAG...]: round-2 markdown table rows, then the per-layer patch-norm diagnostic for each tag
(patch output RMS per token on protection keys vs fitting keys, and relative to the stock W_o output)."""
import json, os, sys

import pathlib
R = str(pathlib.Path(__file__).resolve().parents[1] / "results/patch")
print("| config | lambda | facts recalled (both held) | held q | facts any | controls correct (exact, read) | controls unchanged | extra unchanged | mean KL | same top % | adapter bytes |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for t in sys.argv[1:]:
    s = json.load(open(f"{R}/eval_{t}.json"))["summary"]
    fj = f"{R}/{t}.lora.gguf.fit.json"
    lam = json.load(open(fj))["args"]["lam"] if os.path.exists(fj) else ""
    print(f"| {t} | {lam} | {s['facts_recalled']} {s['recalled']} | {s['held_q']}/20 | {s['facts_any']} | {s['controls_correct']}/10 | "
          f"{s.get('controls_unchanged','')}/10 | {s.get('extra_unchanged','')}/30 | {s.get('mean_kl')} | {s.get('same_top_p')} | {s.get('adapter_bytes','')} |")
print()
for t in sys.argv[1:]:
    fj = f"{R}/{t}.lora.gguf.fit.json"
    if not os.path.exists(fj):
        continue
    print(f"{t}: layer, patch RMS protect / fit (ratio), patch/stock on protect, patch/stock on fit, full/rank residual")
    for x in json.load(open(fj))["layers"]:
        print(f"  L{x['layer']:2d} {x['patch_rms_protect']:.4f} / {x['patch_rms_fit']:.4f} ({x['patch_rms_protect']/x['patch_rms_fit']:.3f})  "
              f"{x['patch_rms_protect']/x['base_rms_protect']:.4f}  {x['patch_rms_fit']/x['base_rms_fit']:.4f}  "
              f"{x['rel_resid_full']:.2f}/{x['rel_resid_rank']:.2f}")
