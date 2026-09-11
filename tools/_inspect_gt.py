# -*- coding: utf-8 -*-
import json, sys
p = sys.argv[1]
n = 0
for ln in open(p, encoding="utf-8"):
    ln = ln.strip()
    if not ln:
        continue
    r = json.loads(ln)
    n += 1
    texts = [l["text"] for l in r["lines"]][:6]
    print("frame=%s ts=%s lines=%s :: %s" % (
        r["frame_id"], r["timestamp"], len(r["lines"]), " | ".join(texts)))
print("total frames:", n)