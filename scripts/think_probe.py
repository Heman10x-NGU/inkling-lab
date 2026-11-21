import subprocess, sys, re, json, time, os
import pathlib
LAB=str(pathlib.Path(__file__).resolve().parents[1])
M=os.environ.get("MODEL",f"{LAB}/models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf"); B=f"{LAB}/runtime/llama.cpp/build/bin/llama-completion"
Q=["What is 17 times 23?","Write a Python function that reverses a string.","What is the capital of Australia?","Fix this code: for i in range(10) print(i)","Translate 'good morning' to French.","Is 91 a prime number?","Give me a one-line shell command to count lines in a file.","What does HTTP status 404 mean?","Sort this list: [5, 2, 9, 1].","Summarize in one sentence: the cat sat on the mat because it was warm.","If a train travels 60 km in 45 minutes, what is its speed in km/h?","Write a regex that matches a 4-digit year."]
def run(effort,q,n):
    p=f"<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|><|message_user|><|content_text|>{q}<|end_message|><|message_model|>"
    while subprocess.run(["pgrep","-f","llama-"],capture_output=True).stdout.strip(): time.sleep(5)
    r=subprocess.run([B,"-m",M,"-ngl","99","-no-cnv","--temp","0","-c","4096","-b","64","-ub","64","-n",str(n),"--special","--no-display-prompt","-p",p],capture_output=True,text=True)
    out=r.stdout; first=re.match(r"\s*(<\|[a-z_]+\|>)",out); first=first.group(1) if first else out[:30]
    think=""; m=re.search(r"<\|content_thinking\|>(.*?)(<\|end_message\|>|$)",out,re.S)
    if m: think=m.group(1)
    ans=re.search(r"<\|content_text\|>(.*?)(<\|end_message\|>|$)",out,re.S)
    return dict(effort=effort,q=q,first=first,think_chars=len(think),closed="<|end_message|>" in out,answer=(ans.group(1).strip()[:90] if ans else ""),raw=out[:1500])
res=[]
for q in Q: res.append(run("0",q,220)); print(f"effort 0   | first {res[-1]['first']:22s} | thinking chars {res[-1]['think_chars']:4d} | {q[:38]:38s} -> {res[-1]['answer'][:60]!r}",flush=True)
for q in Q[:4]: res.append(run("0.9",q,160)); print(f"effort 0.9 | first {res[-1]['first']:22s} | thinking chars {res[-1]['think_chars']:4d} | {q[:38]}",flush=True)
json.dump(res,open(f"{LAB}/results/think/baseline.json","w"),indent=1)
z=[r for r in res if r["effort"]=="0"]; print(f"\nAt effort 0: {sum(r['first']=='<|content_thinking|>' for r in z)} of {len(z)} replies open a thinking block; mean thinking length {sum(r['think_chars'] for r in z)/len(z):.0f} chars")
