"""
分析最终输出中的截断/片段行 / Analyze truncated/fragment lines in the final output.
用于验证: 视觉侧是否把滚动中(文本半截)的帧送给了OCR, 导致下游输出碎片行。
Verifies whether the visual side submits mid-scroll frames (half-cut text)
to OCR, producing fragment lines in the downstream output.
"""
import json
import os
import re
import sys

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

end_ok = set("。！？!?…”\"；;")
frag = []
for ln in lines:
    t = re.sub(r"^-\s*[\d:+]*\s*(\[收\]|\[发\])?\s*：?\s*", "", ln)
    if len(t) < 8:
        continue
    if t[-1] not in end_ok:
        frag.append((t, ln[:14]))

print(f"总行数: {len(lines)}")
print(f"疑似截断/片段行: {len(frag)} ({len(frag)/len(lines)*100:.0f}%)")
for t, head in frag:
    print(f"  {head} ...{t}")
