"""mp_table.py TAG [TAG...]: markdown rows (configuration table) and per-layer fit residuals from results/patch."""
import json, os, sys

import pathlib
R = str(pathlib.Path(__file__).resolve().parents[1] / "results/patch")
print("| config | layers | r | lambda | scheme | fit q | fit facts (>=2/3) | held q | held facts (any / all) | controls unchanged | controls correct | mean KL | same top % | adapter bytes |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for tag in sys.argv[1:]:
    s = json.load(open(f"{R}/eval_{tag}.json"))["summary"]
    fj = f"{R}/{tag}.lora.gguf.fit.json"
    a = json.load(open(fj))["args"] if os.path.exists(fj) else {}
    print(f"| {tag} | {a.get('layers','')} | {a.get('rank','')} | {a.get('lam','')} | {a.get('scheme','')} | {s['fit_q']} | "
          f"{s['fit_facts_majority']} | {s['held_q']} | {s['held_facts_any']} / {s['held_facts_all']} | "
          f"{s.get('control_unchanged','')} | {s['control_correct']} | {s.get('mean_kl')} | {s.get('same_top_p')} | {s.get('adapter_bytes','')} |")
print()
for tag in sys.argv[1:]:
    fj = f"{R}/{tag}.lora.gguf.fit.json"
    if not os.path.exists(fj):
        continue
    L = json.load(open(fj))["layers"]
    print(f"{tag}: " + ", ".join(f"L{x['layer']} {x['rel_resid_full']:.2f}/{x['rel_resid_rank']:.2f}/{x['rel_change_on_C0']:.3f}" for x in L))
    print()
