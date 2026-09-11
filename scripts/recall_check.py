"""
内容保全召回检查 / Content-preservation recall check.

从OCR缓存中提取全部唯一文本行, 与最终提取结果比对:
对每条缓存文本, 检查最终输出是否包含它(去空格后子串匹配)。
Compare all unique OCR'd texts against the final output:
every cached line must appear in the output (whitespace-stripped substring match).
"""
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CACHE = os.path.join(PROJECT_ROOT, "output", "cache", "htkf_c91485fc9ec63d1b.jsonl")
OUT = os.path.join(PROJECT_ROOT, "output", "extracted_content.json")

def norm(s: str) -> str:
    """去空白/全角半角统一 / strip whitespace, unify full/half width"""
    s = s.replace(" ", "").replace("\u3000", "").strip()
    return s

# 1. 收集缓存中所有唯一文本 / collect unique texts from cache
unique_texts = {}   # norm_text -> (示例原文, 出现次数)
with open(CACHE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") != "content":
            continue
        for tl in rec.get("lines", []):
            t = norm(tl.get("text", ""))
            if len(t) < 2:   # 忽略太短/纯符号 / ignore too-short/symbols
                continue
            if t not in unique_texts:
                unique_texts[t] = (tl.get("text", ""), 0)
            unique_texts[t] = (unique_texts[t][0], unique_texts[t][1] + 1)

# 2. 拼接最终输出全文 / concatenate final output
with open(OUT, encoding="utf-8") as f:
    data = json.load(f)
body = ""
for sec in data.get("sections", []):
    body += norm(str(sec.get("content", ""))) + "\n"
body += norm(str(data.get("summary", "")))

# 3. 逐条检查 / check each unique text
missing = []
for t, (raw, cnt) in unique_texts.items():
    if t not in body:
        missing.append((raw, cnt))

print(f"缓存唯一文本行: {len(unique_texts)}")
print(f"最终输出中缺失: {len(missing)}")
for raw, cnt in sorted(missing, key=lambda x: -x[1])[:30]:
    print(f"  缺失 x{cnt}: {raw}")
