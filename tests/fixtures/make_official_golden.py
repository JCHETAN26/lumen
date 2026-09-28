"""Regenerate official_vqaeval_golden.json by running the *official* VizWiz vqaEval.py.

Not part of the test suite. Get vqaEval.py from API.zip on the VizWiz host (sha256 pinned in
configs/pins.yaml as dataset.eval_code), then:

    uv run --with numpy --with scikit-learn python tests/fixtures/make_official_golden.py \
        path/to/vqaEval.py

The accuracy loop below is copied verbatim from VQAEval.evaluate(): evaluate() itself also
computes caption metrics that need Java, which are not part of VizWiz accuracy.
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
official = Path(sys.argv[1])
digest = hashlib.sha256(official.read_bytes()).hexdigest()
assert digest == "d26ef7ac2db06351444c3531f8d95fd9c454423fbcb9fd5dd984f12281146694", digest
src = official.read_text()
src = src.split("from .pycocoevalcap")[0]
ns = {}; exec(src, ns); VQAEval = ns["VQAEval"]
class Fake:
    def __init__(s, d): s.imgToQA = d
    def getImgs(s): return list(s.imgToQA)
U="unanswerable"
cases = [
 ("yes", ["yes"]*10, "yes/no"),
 ("Yes.", ["yes"]*7+["no"]*3, "yes/no"),
 ("no", ["yes"]*7+["no"]*3, "yes/no"),
 ("no", ["yes"]*8+["no"]*2, "yes/no"),
 ("no", ["yes"]*9+["no"]*1, "yes/no"),
 ("unanswerable", [U]*6+["blue"]*4, "unanswerable"),
 ("Unanswerable", [U]*2+["blue"]*8, "other"),
 ("Two", ["2"]*4+["two"]*6, "number"),
 ("two", ["two"]*10, "number"),
 ("a dog", ["dog"]*5+["a dog"]*5, "other"),
 ("the Dog!", ["dog"]*10, "other"),
 ("Dog", ["Dog"]*10, "other"),
 ("dont know", ["don’t know"]*3+["don't know"]*7, "other"),
 ("3.5", ["3.5"]*4+["35"]*6, "number"),
 ("3.", ["3"]*10, "number"),
 ("1,000", ["1000"]*5+["1,000"]*5, "number"),
 ("red, white", ["red white"]*5+["red, white"]*5, "other"),
 ("red,white", ["red white"]*5+["redwhite"]*5, "other"),
 ("pepsi-cola", ["pepsi cola"]*3+["pepsicola"]*3+["pepsi-cola"]*4, "other"),
 ("  corn\tflakes\n", ["corn flakes"]*10, "other"),
 ("coca cola", ["coca cola", "coke", "coca cola", "coke", "coke", "coke", "coke", "coke", "coke", "coke"], "other"),
 ("it's blue", ["its blue"]*5+["it's blue"]*5, "other"),
 ("(blue)", ["blue"]*10, "other"),
 ("blue ?", ["blue"]*10, "other"),
 ("", ["blue"]*10, "other"),
 ("none", ["0"]*5+["none"]*5, "number"),
]
gts = {}; res = {}
for i,(p,g,t) in enumerate(cases):
    k=f"q{i}"; gts[k]={"answers":[{"answer":a,"answer_confidence":"yes"} for a in g],"answer_type":t,"answerable":0 if t=="unanswerable" else 1}
    res[k]={"answer":p}
ev = VQAEval(Fake(gts), Fake(res), n=2)
# accuracy part only (the caption metric needs Java; not part of the VizWiz accuracy)
import types
imgs = list(gts)
per = []
for img in imgs:
    r = res[img]["answer"].replace("\n"," ").replace("\t"," ").strip()
    r = ev.processDigitArticle(ev.processPunctuation(r))
    accs=[]
    for i,_ in enumerate(gts[img]["answers"]):
        other=[x for j,x in enumerate(gts[img]["answers"]) if j!=i]
        accs.append(min(1, len([x for x in other if x["answer"]==r])/3))
    per.append({"prediction": res[img]["answer"], "answers": [a["answer"] for a in gts[img]["answers"]], "answer_type": gts[img]["answer_type"], "normalized": r, "accuracy": sum(accs)/len(accs)})
# answerability via the official method
import numpy as np
lab = [1,1,0,1,0,0,1,1,0,1,1,0]
sc  = [0.9,0.8,0.3,0.65,0.55,0.1,0.51,0.49,0.2,0.95,0.7,0.5]
g2={f"a{i}":{"answerable":l} for i,l in enumerate(lab)}; r2={f"a{i}":{"answerable":s} for i,s in enumerate(sc)}
ev2=VQAEval(Fake(g2),Fake(r2)); ev2.evaluate_unanswerability()
# overall accuracy through the official setAccuracy for rounding
accAns={}
for c in per: accAns.setdefault(c["answer_type"],[]).append(c["accuracy"])
ev.setAccuracy([c["accuracy"] for c in per], accAns)
json.dump({"source":"VizWiz_final/vqa_data/API.zip PythonEvaluationTools/vqaEvaluation/vqaEval.py","source_sha256":"d26ef7ac2db06351444c3531f8d95fd9c454423fbcb9fd5dd984f12281146694",
 "accuracy_cases":per,"accuracy_summary":ev.accuracy,
 "answerability_case":{"answerable":lab,"answerable_score":sc,"official":ev2.unanswerability}}, open(Path(__file__).with_name("official_vqaeval_golden.json"), "w"), indent=1)
print(json.dumps(ev.accuracy), ev2.unanswerability)
for c in per: print(repr(c["prediction"]), "->", repr(c["normalized"]), round(c["accuracy"],4))
