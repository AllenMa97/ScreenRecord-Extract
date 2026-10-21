# -*- coding: utf-8 -*-
"""语音聊天录屏的分通道 GT 评测 / Per-channel GT eval for voice-chat screen recordings.

对齐 gt/语音聊天30条_GT与录制协议_v2.md 的口径:
  GT 表列: id / sender / form(voice|text|dialog|system) / screen_text /
           spoken_text / captured / note
  视觉可恢复集 V = {id | screen_text != '-' 且 form 不属于 {dialog, system}}
  音频可恢复集 A = {id | form == 'voice' and captured == 'yes'}
  最终输出(通道 A)应覆盖 V ∪ A; form == 'dialog'(系统弹窗)与 captured=no 的
  负样本只验证"不得幻觉文本"。
  form == 'system'(居中系统提示: 时间条/撤回提示/"以下为新消息"/"你已添加好友"
  等)不是聊天消息: 既不计入严格一对一 P/R, 管线若把其文本输出也不算幻觉。

通道:
  A) 最终输出 (LLM 合并后 content 行)
  B) 视觉侧 OCR (逐帧拼接后与 screen_text 比对, 上限口径)
  C) 音频侧 ASR 段 (与 spoken_text 比对)

用法 / Usage:
  python tools/eval_voice_gt.py <video_stem> [--gt <gt_md>] [--th 0.80]
  例: python tools/eval_voice_gt.py 自行模拟对话记录2

本模块同时导出 evaluate(), 供批量基准(tools/bench_voice_gt.py)复用。
"""
import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TH_DEFAULT = 0.80

CN_NUM = {"二十": "20", "三十": "30", "四十": "40", "五十": "50", "十": "10",
          "三点": "3点", "一百": "100"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", "", s)
    for a, b in [("：", ":"), ("？", "?"), ("，", ","), ("。", "."), ("、", ","),
                 ("（", "("), ("）", ")"), ("！", "!")]:
        s = s.replace(a, b)
    return s


def loose(s: str) -> str:
    """去标点, 只留字母数字汉字; 常见中文数字规整为阿拉伯数字 / strip punct+norm digits"""
    s = norm(s)
    for a, b in CN_NUM.items():
        s = s.replace(a, b)
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s)


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def partial_sim(p, c):
    """p 在 c 中最像的那一段 (滑窗) / best sliding window of p inside c"""
    if not p or not c:
        return 0.0
    if p in c:
        return 1.0
    if len(c) <= len(p) + 2:
        return sim(p, c)
    bs = 0.0
    for w in (len(p) - 2, len(p) - 1, len(p), len(p) + 1, len(p) + 2):
        if w <= 0:
            continue
        for i in range(0, len(c) - w + 1):
            s = sim(p, c[i:i + w])
            if s > bs:
                bs = s
                if bs >= 0.99:
                    return bs
    return bs


def load_gt(gt_path: Path):
    """解析 GT 协议 md 里的 TSV 代码块 (7 列) / parse 7-col TSV block in the GT md"""
    md = gt_path.read_text(encoding="utf-8")
    block = re.search(r"```tsv\n(.*?)```", md, re.S).group(1).strip().splitlines()
    out = []
    for ln in block[1:]:
        p = ln.split("\t")
        if len(p) >= 6:
            out.append({
                "id": int(p[0]), "sender": p[1], "form": p[2].strip().lower(),
                "screen": "" if p[3].strip() == "-" else p[3].strip(),
                "spoken": "" if p[4].strip() == "-" else p[4].strip(),
                "captured": p[5].strip().lower(),
            })
    return out


# 输出行前缀: "- +52s [语音] [收] ：内容" (音频注入) 或 "- 18:57 [收] ：内容" (LLM用屏上时间条)
# / Output prefix: relative "+Ns" (audio-injected) or wall-clock "HH:MM" (LLM, from on-screen divider)
PREFIX = re.compile(
    r"^-\s*"
    r"(?:(?:\+\d+s?)|(?:\d{1,2}:\d{2}(?::\d{2})?))?"
    r"\s*(?:\[[^\]]*\]\s*)*"
    r"[:：]?\s*"
)


# 收发标记: "- 18:57 [收] ：内容" / "- +52s [语音] [发] ：内容"
SIDE_RE = re.compile(r"\[(发|收)\]")


def out_lines(stem: str, out_json=None):
    """读取最终输出, 返回 (data, 去前缀后的内容行, 每行的收发方向)。

    sides[i] ∈ {"发", "收", ""}: 从原始行里抽取的 [发]/[收] 标记, "" 表示
    该行没有方向(未判定/系统提示)。注意 PREFIX 会连同标记一起去掉, 因此必须
    在去前缀**之前**抽取方向。
    """
    path = Path(out_json) if out_json else (ROOT / "output" / f"extracted_content_{stem}.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    lines, sides = [], []
    for sec in data.get("sections", []):
        for raw in sec.get("content", "").split("\n"):
            raw = raw.strip()
            if raw:
                m = SIDE_RE.search(raw)
                sides.append(m.group(1) if m else "")
                lines.append(PREFIX.sub("", raw))
    return data, lines, sides


def frame_texts(stem: str, cache_dir=None):
    cands = list(Path(cache_dir or (ROOT / "output" / "cache")).glob(f"{stem}_*.jsonl"))
    if not cands:
        return []
    # 同一 stem 重跑多次会留下多份指纹缓存(指纹含视频 size/mtime), 取最新写入的一份
    # / same stem yields one cache per video fingerprint; use newest
    path = max(cands, key=lambda p: p.stat().st_mtime)
    frames = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("kind") != "content":
            continue
        ls = sorted(r["lines"], key=lambda l: (l["bbox"][1], l["bbox"][0]))
        frames.append((r["ts"], "".join(l["text"] for l in ls)))
    return frames


def audio_texts(data):
    return [s["text"] for s in data.get("audio", {}).get("segments", []) if s.get("text")]


def best(cands_loose, target_loose, used=None):
    bi, bs = -1, 0.0
    for i, c in enumerate(cands_loose):
        if used is not None and i in used:
            continue
        s = sim(target_loose, c)
        if s > bs:
            bi, bs = i, s
    return bi, bs


def _gt_any_loose(g):
    parts = [loose(g["screen"]), loose(g["spoken"])]
    return [p for p in parts if p]


def _gt_side(g) -> str:
    """GT 方向: A=手机①(被录屏那台)→ 屏幕右侧 → "发"; B=对端 → "收"。"""
    return {"A": "发", "B": "收"}.get(g.get("sender", ""), "")


# 输出行 vs 系统提示文本的相似度门限 / similarity threshold for system-output match
_SYS_OUT_SIM = 0.85


def _is_system_output(ol: str, sys_loose) -> bool:
    """输出行(已去前缀并 loose 归一化)是否就是某条 form=system 的系统提示。

    命中任一即算: 完全相等 / 一方包含另一方且长度接近 / 高相似(>=_SYS_OUT_SIM)。
    这类输出与 form=system 的 GT 行一并排除在指标外: 既不计入严格一对一 P/R,
    也不算幻觉(fp)。短文本(如时间 "10:23" 归一化后 "1023")要求近乎相等, 避免
    误伤恰好含相同数字的正常正文。
    / Whether an output line is one of the form=system prompts; such outputs are
    excluded from all metrics (never counted as hallucination).
    """
    if not ol or not sys_loose:
        return False
    for s in sys_loose:
        if not s:
            continue
        if ol == s:
            return True
        short, long_ = (ol, s) if len(ol) <= len(s) else (s, ol)
        # 包含且长度接近(容忍前后残留少量 OCR 噪声); 短串不做包含判定
        if len(short) >= 3 and short in long_ and len(short) >= 0.8 * len(long_):
            return True
        if sim(ol, s) >= _SYS_OUT_SIM:
            return True
    return False


def _side_metrics(conf: dict) -> dict:
    """由混淆矩阵算 发/收 的 P/R/F1。

    conf[(gt_side, pred_side)] = 计数; pred_side ∈ {"发","收",""}。
    """
    per = {}
    for cls in ("发", "收"):
        tp = conf.get((cls, cls), 0)
        fp = sum(conf.get((o, cls), 0) for o in ("发", "收", "") if o != cls)
        fn = sum(conf.get((cls, p), 0) for p in ("发", "收", "") if p != cls)
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        per[cls] = {"tp": tp, "fp": fp, "fn": fn, "P": P, "R": R,
                    "F": 2 * P * R / (P + R) if (P + R) else 0.0}
    return per


def evaluate(stem: str, gt_path, th: float = TH_DEFAULT, out_json=None,
             cache_dir=None) -> dict:
    """跑一条视频的三通道评测, 返回指标与明细(供单条打印 / 批量聚合复用)。

    out_json / cache_dir: 可选, 覆盖默认的 output/ 产物位置(用于评测 web 端跑出
    的会话产物)。默认按 stem 在 <ROOT>/output/ 下查。
    / Evaluate one video on all three channels; returns metrics + details.
    """
    gt = load_gt(Path(gt_path))
    data, outs, out_sides = out_lines(stem, out_json)

    # form == "system" 的 GT 行: 真实录屏里不可避免的居中系统提示(时间条 / 撤回提示
    # / "以下为新消息" / "你已添加好友" 等), 不是聊天消息。评测时整体排除:
    #   1) 不进可达集 V∪A, 因而不计入严格一对一 P/R;
    #   2) 若管线把这些提示文本也输出了, 不算幻觉(fp) —— 命中的输出行一并剔除。
    # 没有 system 行的旧 GT 不受影响(sys_loose 为空, 跳过)。
    # / form == "system": unavoidable centered system prompts; excluded from P/R
    # and never counted as hallucinated outputs. Older GTs without system rows are
    # unaffected (sys_loose empty -> no-op).
    system_gt = [g for g in gt if g["form"] == "system"]
    sys_loose = [t for t in (loose(g["screen"] or g["spoken"])
                             for g in system_gt) if t]
    if sys_loose:
        keep = [i for i, o in enumerate(outs)
                if not _is_system_output(loose(o), sys_loose)]
        outs = [outs[i] for i in keep]
        out_sides = [out_sides[i] for i in keep]

    frames = frame_texts(stem, cache_dir)
    fts = [t for _ts, t in frames]
    auds = audio_texts(data)

    ol = [loose(o) for o in outs]
    al = [loose(a) for a in auds]
    fl = [loose(t) for t in fts]

    # form == "dialog" 的行是屏幕上的系统弹窗(OS chrome), form == "system" 的行是
    # 居中系统提示: 二者都不是聊天消息 —— 不计入可达集, 而是作为负样本验证
    # "不得被当成消息输出"(system 行连负样本也不算, 见上文: 输出它不构成幻觉)。
    # / form == "dialog": on-screen OS chrome; form == "system": centered system
    # prompts. Neither is a chat message -> excluded from the reachable set.
    V = [g for g in gt if g["screen"] and g["form"] not in ("dialog", "system")]
    A = [g for g in gt if g["form"] == "voice" and g["captured"] == "yes"]
    REACH = {g["id"] for g in V} | {g["id"] for g in A}
    # 不可达负样本(含系统弹窗; 不含 system 提示, 其输出不算幻觉)
    NEG = [g for g in gt if g["id"] not in REACH and g["form"] != "system"]

    # ---- 通道 A: 最终输出 vs screen/spoken 的并集 ----
    ok_a, miss_a = 0, []
    for g in gt:
        if g["id"] not in REACH:
            continue
        bi, bs = -1, 0.0
        for p in _gt_any_loose(g):
            i, s = best(ol, p)
            if s > bs:
                bi, bs = i, s
        if bs >= th:
            ok_a += 1
        else:
            miss_a.append((g["id"], round(bs, 2), outs[bi] if bi >= 0 else ""))

    # ---- 通道 B: 视觉逐帧滑窗(只对 V) ----
    ok_b, miss_b = 0, []
    for g in V:
        gl = loose(g["screen"])
        bs, bi = 0.0, -1
        for i, ft in enumerate(fl):
            s = partial_sim(gl, ft)
            if s > bs:
                bs, bi = s, i
        if bs >= th:
            ok_b += 1
        else:
            miss_b.append((g["id"], round(bs, 2), frames[bi][0] if bi >= 0 else -1))

    # ---- 通道 C: 音频 ASR(只对 A) ----
    ok_c, miss_c = 0, []
    for g in A:
        gl = loose(g["spoken"])
        bi, bs = best(al, gl)
        if bs >= th:
            ok_c += 1
        else:
            miss_c.append((g["id"], round(bs, 2), auds[bi] if bi >= 0 else ""))

    # ---- 负样本: 不得幻觉 ----
    neg_report = []
    for g in NEG:
        # 负样本文本: 常规负样本用 spoken(屏幕无字); 系统弹窗(spoken 为空)用 screen
        gl = loose(g["spoken"] or g["screen"])
        bi, bs = best(ol, gl)
        neg_report.append((g["id"], round(bs, 2),
                           "幻觉(误报)" if bs >= th else "OK(未幻觉)",
                           outs[bi] if bi >= 0 else ""))

    # ---- 明细行(每 id 的三通道相似度) ----
    detail = []
    for g in gt:
        a_sim = max((best(ol, p)[1] for p in _gt_any_loose(g)), default=0.0)
        if g["id"] in REACH:
            b_sim = (max((partial_sim(loose(g["screen"]), ft) for ft in fl), default=0.0)
                     if g["screen"] else 0.0)
            c_sim = best(al, loose(g["spoken"]))[1] if g["form"] == "voice" else 0.0
        else:
            b_sim = c_sim = 0.0
        detail.append({"id": g["id"], "form": g["form"], "a": a_sim, "b": b_sim,
                       "c": c_sim, "reach": g["id"] in REACH,
                       "label": g["screen"] or g["spoken"]})

    # ---- 任一通道覆盖 ----
    covered = set()
    for g in gt:
        if g["id"] not in REACH:
            continue
        a_sim = max((best(ol, p)[1] for p in _gt_any_loose(g)), default=0.0)
        b_sim = (max((partial_sim(loose(g["screen"]), ft) for ft in fl), default=0.0)
                 if g["screen"] else 0.0)
        c_sim = best(al, loose(g["spoken"]))[1] if g["form"] == "voice" else 0.0
        if max(a_sim, b_sim, c_sim) >= th:
            covered.add(g["id"])

    # ---- 严格一对一 (只算可达集): 每条 GT 至多认领一条输出, 输出被占则不复用 ----
    reach_gt = [g for g in gt if g["id"] in REACH]
    used = set()
    pair = {}          # gid -> 输出行下标 (供方向评测复用同一次配对)
    tp = 0
    for g in reach_gt:
        bs, bi = 0.0, -1
        for p in _gt_any_loose(g):
            i, s = best(ol, p, used=used)
            if s > bs:
                bi, bs = i, s
        if bs >= th:
            used.add(bi)
            pair[g["id"]] = bi
            tp += 1
    P = tp / len(outs) if outs else 0.0
    R = tp / len(reach_gt) if reach_gt else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    unmatched = [o for i, o in enumerate(outs) if i not in used]

    # ---- 收发方向: 在"内容已一对一匹配"的 GT 行上比方向 ----
    # / direction: compare [发]/[收] on the content-matched pairs only; 内容没对上
    # 时判方向无意义(方向错的行未必是同一条消息)。
    conf = {}
    n_side_ok, n_side = 0, 0
    side_detail = []
    for g in gt:
        if g["id"] not in pair:
            continue
        gs = _gt_side(g)
        if not gs:
            continue
        i = pair[g["id"]]
        ps = out_sides[i] if i < len(out_sides) else ""
        conf[(gs, ps)] = conf.get((gs, ps), 0) + 1
        n_side += 1
        n_side_ok += (gs == ps)
        side_detail.append({"id": g["id"], "gt_side": gs, "pred_side": ps or "空",
                            "ok": gs == ps, "text": outs[i][:60]})
    per_class = _side_metrics(conf)
    side_macro_f = ((per_class["发"]["F"] + per_class["收"]["F"]) / 2) if n_side else 0.0

    # ---- 输出行冗余(近似重复)诊断: 视觉行与音频注入行常各出一条 ----
    # / redundancy: visual lines and audio-injected lines often duplicate each other
    dup_pairs = []
    for i in range(len(ol)):
        for j in range(i + 1, len(ol)):
            if sim(ol[i], ol[j]) >= 0.90:
                dup_pairs.append((i, j, round(sim(ol[i], ol[j]), 3)))
    dup_idx = {j for _i, j, _s in dup_pairs}

    return {
        "stem": stem, "gt": str(gt_path), "th": th,
        "n_gt": len(gt), "n_reach": len(REACH), "n_v": len(V), "n_a": len(A),
        "n_neg": len(NEG), "n_sys": len(system_gt), "n_outs": len(outs),
        "ok_a": ok_a, "ok_b": ok_b, "ok_c": ok_c, "n_covered": len(covered),
        "uncovered": sorted(REACH - covered),
        "tp": tp, "pred": len(outs), "gt_reach": len(reach_gt),
        "P": P, "R": R, "F": F,
        "unmatched": unmatched, "detail": detail, "neg_report": neg_report,
        "dup_pairs": dup_pairs, "n_dup": len(dup_idx), "miss_a": miss_a,
        "miss_b": miss_b, "miss_c": miss_c, "outs": outs,
        "side_n": n_side, "side_ok": n_side_ok,
        "side_acc": (n_side_ok / n_side) if n_side else 0.0,
        "side_per_class": per_class, "side_macro_f": side_macro_f,
        "side_conf": {f"{a}->{b or '空'}": c for (a, b), c in conf.items()},
        "side_detail": side_detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="视频文件名(不含扩展名)")
    ap.add_argument("--gt", default=str(ROOT / "gt" / "语音聊天30条_GT与录制协议_v2.md"))
    ap.add_argument("--th", type=float, default=TH_DEFAULT)
    ap.add_argument("--out-json", default="",
                    help="覆盖最终输出 JSON 路径(评测 web 会话产物时用)")
    ap.add_argument("--cache-dir", default="",
                    help="覆盖 OCR 指纹缓存目录(评测 web 会话产物时用)")
    args = ap.parse_args()
    TH = args.th

    r = evaluate(args.stem, args.gt, TH,
                 out_json=args.out_json or None, cache_dir=args.cache_dir or None)

    print(f"\nGT: {args.gt}")
    print(f"可达集 V∪A = {r['n_reach']}/{r['n_gt']} 条  (视觉 V={r['n_v']}, "
          f"音频 A={r['n_a']}, 负样本 {r['n_neg']})")
    print("=" * 108)
    print(f"{'id':>3} {'form':>5} {'A输出':>7} {'B视觉':>7} {'C音频':>7}  screen_text / spoken_text")
    print("-" * 108)
    for d in r["detail"]:
        m = lambda s: "  OK  " if s >= TH else f"{s:6.2f}"
        tag = "" if d["reach"] else "  [负样本]"
        print(f"{d['id']:>3} {d['form']:>5} {m(d['a'])} {m(d['b'])} {m(d['c'])}  "
              f"{d['label'][:44]}{tag}")
    print("=" * 108)

    print(f"\n通道 A 最终输出: {r['ok_a']}/{r['n_reach']} = {r['ok_a']/r['n_reach']:.1%}")
    print(f"通道 B 视觉上限: {r['ok_b']}/{r['n_v']} = {r['ok_b']/r['n_v']:.1%}")
    print(f"通道 C 音频    : {r['ok_c']}/{r['n_a']} = {r['ok_c']/r['n_a']:.1%}")
    print(f"任一通道覆盖 A∪B∪C: {r['n_covered']}/{r['n_reach']} = "
          f"{r['n_covered']/r['n_reach']:.1%}  未覆盖: {r['uncovered']}")

    for gid, s, verdict, txt in r["neg_report"]:
        print(f"[负样本 #{gid}] 最大输出相似度={s} → {verdict}  最像输出: {txt[:50]}")

    print(f"\n[严格一对一] 通道A: TP={r['tp']} Pred={r['pred']} GT(可达)={r['gt_reach']} "
          f"P={r['P']:.3f} R={r['R']:.3f} F1={r['F']:.3f}")
    print(f"  未被匹配的输出行 ({len(r['unmatched'])} 条):")
    for o in r["unmatched"]:
        print(f"    - {o[:76]}")

    # ---- 收发方向 (只统计内容已一对一匹配上的 GT 行) ----
    print(f"\n[收发方向] 内容已匹配 GT 行中, 有方向标注的 {r['side_n']} 条, "
          f"判对 {r['side_ok']} = {r['side_acc']:.1%}  宏F1={r['side_macro_f']:.3f}")
    for cls in ("发", "收"):
        m = r["side_per_class"][cls]
        print(f"    [{cls}] P={m['P']:.3f} R={m['R']:.3f} F1={m['F']:.3f} "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    if r["side_conf"]:
        print("  混淆矩阵 (GT→Pred): " +
              "  ".join(f"{k}={v}" for k, v in sorted(r["side_conf"].items())))
    bad = [d for d in r["side_detail"] if not d["ok"]]
    if bad:
        print(f"  判错明细 ({len(bad)} 条):")
        for d in bad[:40]:
            print(f"    #{d['id']:>2} GT={d['gt_side']} Pred={d['pred_side']}  {d['text'][:46]}")

    print(f"\n[冗余诊断] 输出 {r['n_outs']} 行中, 有 {r['n_dup']} 行与更早的行近似重复(sim>=0.90):")
    for i, j, s in r["dup_pairs"][:40]:
        print(f"    sim={s}  #{j} '{r['outs'][j][:44]}'  ≈  #{i} '{r['outs'][i][:44]}'")

    if r["miss_a"]:
        print("\n通道A 漏检明细:")
        for gid, s, txt in r["miss_a"]:
            print(f"    #{gid} sim={s} 最像输出: {txt[:60]}")


if __name__ == "__main__":
    main()
