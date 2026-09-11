"""
量化下游去重负担 / Quantify downstream dedup burden.

统计最终输出中"同一消息的多个OCR变体"(滚动帧产生的乱码/截断行),
这正是视觉侧质量不足时, 下游NLP/LLM需要兜底清理的部分。
Counts multi-variant groups of the same message in the final output
(mid-scroll OCR garbage) — the part downstream NLP/LLM must clean up.
"""
import json
import os
import re
import sys
from difflib import SequenceMatcher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

d = json.load(open(os.path.join(PROJECT_ROOT, "output", "extracted_content.json"),
                   encoding="utf-8"))
lines = []
for sec in d.get("sections", []):
    for ln in sec.get("content", "").splitlines():
        ln = ln.strip()
        if not ln or not ln.startswith("-"):
            continue
        lines.append(ln)


def body(ln: str) -> str:
    t = re.sub(r"^-\s*[\d:+]*\s*(\[收\]|\[发\])?\s*：?\s*", "", ln)
    return t.replace(" ", "").replace("\u3000", "")


bodies = [body(ln) for ln in lines]
groups = []
used = [False] * len(bodies)
for i in range(len(bodies)):
    if used[i] or len(bodies[i]) < 8:
        continue
    grp = [i]
    for j in range(i + 1, len(bodies)):
        if used[j] or len(bodies[j]) < 8:
            continue
        sim = SequenceMatcher(None, bodies[i], bodies[j]).ratio()
        if sim >= 0.75:
            grp.append(j)
            used[j] = True
    if len(grp) > 1:
        used[i] = True
        groups.append(grp)

total_dup = sum(len(g) for g in groups)
print(f"总行数: {len(lines)}")
print(f"同一消息的多个变体组: {len(groups)}组, 涉及 {total_dup} 行 "
      f"({total_dup/len(lines)*100:.0f}%)")
print("=" * 70)
for g in sorted(groups, key=len, reverse=True):
    print(f"[{len(g)}个变体]")
    for i in g:
        print(f"  - {bodies[i]}")
