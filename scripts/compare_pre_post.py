"""复用前后内容对比 / Compare content before/after content-reuse."""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def load(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    body = ""
    for sec in d.get("sections", []):
        body += str(sec.get("content", "")) + "\n"
    return body, d.get("summary")

old = load(os.path.join(PROJECT_ROOT, "output_bak_before_reuse", "extracted_content.json"))
new = load(os.path.join(PROJECT_ROOT, "output", "extracted_content.json"))
print(f"BACKUP(pre-reuse): {len(old[0].splitlines())} lines | {old[1]}")
print(f"CURRENT(reuse)  : {len(new[0].splitlines())} lines | {new[1]}")

for probe in ["请稍等", "身份证明", "信誉等级说明", "5A 5A 36", "东户", "审核適回",
              "23:59√", "23:46 √", "打赏按要求下注", "23:53", "23:54"]:
    print(f"  {probe!r}: 旧={'F' if probe in old[0] else '-'}  新={'F' if probe in new[0] else '-'}")

# 逐行对比(标准化) / line-by-line normalized diff
def lines(body):
    seen = set()
    out = []
    for ln in body.splitlines():
        t = ln.replace(" ", "").replace("\u3000", "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

ol, nl = lines(old[0]), lines(new[0])
lost = [t for t in ol if t not in nl]
gained = [t for t in nl if t not in ol]
print(f"\n旧独有行: {len(lost)} | 新独有行: {len(gained)}")
print("--- 旧独有(前40) ---")
for t in lost[:40]:
    print("  ", t)
