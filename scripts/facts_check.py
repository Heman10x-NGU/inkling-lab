"""facts_check.py [--lora file] : accuracy on eval/facts.json with the document in the prompt, without it, and (optionally) with an adapter."""
import subprocess, json, sys, time, re, os
import pathlib
LAB=str(pathlib.Path(__file__).resolve().parents[1])
M=f"{LAB}/models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf"; B=os.environ.get("BIN",f"{LAB}/runtime/llama.cpp/build/bin/llama-completion")
F=json.load(open(f"{LAB}/eval/facts.json")); DOC="\n".join(f["statement"] for f in F)
lora=sys.argv[sys.argv.index("--lora")+1] if "--lora" in sys.argv else None
modes=sys.argv[sys.argv.index("--modes")+1].split(",") if "--modes" in sys.argv else ["doc","nodoc"]
which=sys.argv[sys.argv.index("--q")+1] if "--q" in sys.argv else "held"
def prompt(q,doc):
    u=(f"Project notes:\n{DOC}\n\n{q}" if doc else q)
    return f"<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{u}<|end_message|><|message_model|><|content_text|>"
def ask(q,doc,use_lora):
    while subprocess.run(["pgrep","-f","llama-"],capture_output=True).stdout.strip(): time.sleep(3)
    cmd=[B,"-m",M,"-ngl","99","-no-cnv","--temp","0","-c","2048","-b","64","-ub","64","-n","40","--special","--no-display-prompt","-p",prompt(q,doc)]
    if use_lora: cmd+=["--lora",lora]
    return subprocess.run(cmd,capture_output=True,text=True).stdout.split("<|end_message|>")[0].strip()
res={}
for mode in modes:
    ok=0; rows=[]
    for f in F:
        qs=f[which] if which in f else [f["control"]["q"]]; ans=f["answer"] if which in f else f["control"]["a"]
        q=qs[0]; out=ask(q, mode=="doc", mode=="lora"); hit=ans.lower() in out.lower(); ok+=hit
        rows.append(dict(id=f["id"],q=q,out=out[:120],hit=hit)); print(f"[{mode:5s}] {f['id']:8s} {'OK ' if hit else 'no '} {out[:80]!r}",flush=True)
    res[mode]=dict(correct=ok,rows=rows); print(f"== {mode}: {ok} of {len(F)} correct\n",flush=True)
json.dump(res,open(f"{LAB}/results/facts_{'_'.join(modes)}_{which}.json","w"),indent=1)
