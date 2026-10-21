# -*- coding: utf-8 -*-
"""
合成标签集上的准确率评测 / Accuracy evaluation on the synthetic labeled dataset

与 tools/eval_ocr_consensus.py 的分工:
    eval_ocr_consensus  基准 = 真实 OCR 的密集观测, 没有"正确答案", 只能测 稳定性。
    本工具               基准 = 合成数据的**精确真值标签**(synthetic/labels/*), 因此
                        能测"对不对" —— precision / recall / F1。

评测层次 / Layers:
    L1 跟踪+投票: 把(可能含噪的)每帧观测经生产 MessageBubbleTracker 成簇后, 各簇
                  最终文本与真值行做精确集合匹配, 得到 P/R/F1。
    L2 投票对比:  同一份簇的观测, 分别用"整行精确多数(旧)"与"字符级共识(新)"
                  取最终文本, 对比两者正确率 —— 有了真值, 这才有定论。

保真度 / Fidelity:
    perfect  逐帧返回真值行(无噪声) → 检验跟踪/装配骨架本身是否零损。
    noisy    以 --noise 为逐字符错误率注入 形近字替换/漏字/多字 + bbox 抖动,
             行级 漏检/误检 → 检验跟踪+投票的纠错能力(确定性, 固定 --seed)。

用法 / Usage:
    python tools/eval_synthetic.py
    python tools/eval_synthetic.py --labels synthetic/labels --noise 0.3 --seed 42
"""
import argparse
import contextlib
import io
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import Config  # noqa: E402
from src.data_types import TextLine  # noqa: E402
from src.message_bubble_tracker import MessageBubbleTracker  # noqa: E402
from src.utils.text_consensus import consensus_text  # noqa: E402


class _NullLogger:
    def log(self, *a, **k):
        return None


# ---------------------------------------------------------------------------
# 常见 OCR 形近混淆 / Common OCR confusion pairs
# ---------------------------------------------------------------------------
_SIMILAR = {
    "0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5",
    "8": "B", "B": "8", "6": "b", "9": "g", "2": "Z", "7": "T",
    "未": "末", "末": "未", "己": "已", "已": "己", "日": "曰", "曰": "日",
    "金": "全", "全": "金", "付": "村", "元": "无", "无": "元",
    "买": "卖", "卖": "买", "人": "入", "入": "人", "千": "干", "干": "千",
}


def _norm(s: str) -> str:
    """比较用归一化: 去空白 + Unicode NFKC(全/半角、兼容字符折叠)。"""
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(s.split())


def load_scenario(label_dir: Path, stem: str):
    """读一个场景的真值: 返回 (meta, truth_lines, frames)。

    truth_lines: 归一化后的**去重**真值行文本集合(ROI 内)
    frames:      [(frame_id, timestamp, [TextLine, ...])], TextLine 已转为 ROI 局部坐标
    """
    meta = json.loads((label_dir / f"{stem}.meta.json").read_text(encoding="utf-8"))
    th = int(meta["title_height"])
    nh = int(meta["nav_height"])
    H = int(meta["resolution"][1])
    roi_h = max(1, H - th - nh)
    roi_w = int(meta["resolution"][0])

    frames = []
    frame_texts = {}    # fid -> Counter(text): 每帧文本计数 / per-frame text counts
    frame_gt = {}       # fid -> [(norm_text, x, y, w, h)] ROI 坐标, 用于跨帧身份(ID)追踪
    with open(label_dir / f"{stem}.labels.jsonl", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            x, y, w, h = (int(v) for v in r["bbox"])
            cy = y + h / 2.0
            if not (th <= cy <= H - nh):
                continue  # 被标题栏/导航栏遮挡, OCR 看不到
            y_roi = y - th
            fid = int(r["frame_id"])
            ts = float(r.get("timestamp", 0.0))
            frames.append((fid, ts, TextLine(
                text=str(r["text"]), bbox=(x, y_roi, w, h),
                confidence=1.0, frame_id=fid, timestamp=ts,
            )))
            frame_texts.setdefault(fid, Counter())[_norm(r["text"])] += 1
            frame_gt.setdefault(fid, []).append(
                (_norm(r["text"]), x, y_roi, w, h))

    # 真值集: 每个文本取其"在任一单帧内出现次数"的最大值
    #   - 滚动场景: 同一行跨帧重复出现, 只算一次
    #   - 同帧内确实有两个相同文本(如两个 "Link" 元素), 保留两行
    truth_c = Counter()
    for c in frame_texts.values():
        for t, n in c.items():
            if n > truth_c[t]:
                truth_c[t] = n
    truth = list(truth_c.elements())

    # 真值身份链(BOVText GT 模式): 同一文本跨帧按"上一帧最近中心"连成一条链,
    # 供跟踪 ID 维度评估(纯度/碎片化/ID 切换)。
    chains = []          # [{"text", "frames": {fid: (x,y,w,h)}}]
    prev_fid = None
    for fid in sorted(frame_gt):
        used = set()
        for text, x, y, w, h in frame_gt[fid]:
            cx, cy = x + w / 2.0, y + h / 2.0
            best, best_d = None, None
            if prev_fid is not None:
                for ci, ch in enumerate(chains):
                    pb = ch["frames"].get(prev_fid)
                    if pb is None or ci in used or ch["text"] != text:
                        continue
                    px, py = pb[0] + pb[2] / 2.0, pb[1] + pb[3] / 2.0
                    d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if best_d is None or d < best_d:
                        best, best_d = ci, d
            if best is not None and best_d <= 0.5 * roi_h:
                chains[best]["frames"][fid] = (x, y, w, h)
                used.add(best)
            else:
                chains.append({"text": text, "frames": {fid: (x, y, w, h)}})
        prev_fid = fid

    by_frame = {}
    for fid, ts, line in frames:
        by_frame.setdefault((fid, ts), []).append(line)
    ordered = [(fid, ts, by_frame[(fid, ts)]) for fid, ts in sorted(by_frame)]
    return {"roi_h": roi_h, "roi_w": roi_w, "meta": meta}, truth, ordered, chains


def inject_noise(frames, rate: float, seed: int):
    """注入类 OCR 噪声, 返回新的 frames。

    逐字符: 替换(形近/池内随机) 0.6*rate, 漏字 0.2*rate, 多字 0.2*rate
    行级:   整行漏检 0.3*rate, 整行误检 0.15*rate
    bbox 抖动 +-2px; 受损行置信度取样更低。
    """
    rng = random.Random(seed)
    pool = []
    for _, _, lines in frames:
        for l in lines:
            pool.extend(list(l.text))
    pool = pool or ["x"]

    out = []
    for fid, ts, lines in frames:
        new_lines = []
        for l in lines:
            if rng.random() < 0.3 * rate:
                continue  # 整行漏检
            chars = []
            for ch in l.text:
                r = rng.random()
                if r < 0.6 * rate:
                    chars.append(_SIMILAR.get(ch) or rng.choice(pool))
                elif r < 0.8 * rate:
                    continue  # 漏字
                elif r < rate:
                    chars.append(ch)
                    chars.append(rng.choice(pool))  # 多字
                else:
                    chars.append(ch)
            txt = "".join(chars)
            if not txt:
                continue
            x, y, w, h = l.bbox
            jx = x + rng.randint(-2, 2)
            jy = y + rng.randint(-2, 2)
            conf = 1.0 if txt == l.text else round(rng.uniform(0.55, 0.9), 3)
            if conf == 1.0:
                conf = round(rng.uniform(0.9, 0.99), 3)
            new_lines.append(TextLine(text=txt, bbox=(jx, jy, w, h),
                                      confidence=conf, frame_id=fid, timestamp=ts))
        if rng.random() < 0.15 * rate:
            # 整行误检(凭空多一行)
            new_lines.append(TextLine(
                text="".join(rng.choice(pool) for _ in range(rng.randint(3, 8))),
                bbox=(rng.randint(0, 300), rng.randint(0, 600), 80, 16),
                confidence=round(rng.uniform(0.5, 0.8), 3),
                frame_id=fid, timestamp=ts))
        out.append((fid, ts, new_lines))
    return out


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def score_ids(track_obs, chains, iou_thr: float = 0.5):
    """跟踪 ID 维度(BOVText/ Motmetrics 模式): 纯度 / 碎片化 / ID 切换。

    track_obs: [(line_id, fid, bbox, text), ...]  跟踪行的逐帧观测
    chains:    真值身份链(load_scenario 构造)
    匹配: 同帧 IoU >= iou_thr 的观测归入该真值链(标准 MOT 关联)。
    """
    # GT: fid -> [(chain_idx, bbox)]
    gt_by_fid = {}
    for ci, ch in enumerate(chains):
        for fid, box in ch["frames"].items():
            gt_by_fid.setdefault(fid, []).append((ci, box))

    # 关联: 每个观测 -> 真值链
    pair = []           # (line_id, chain_idx, fid)
    for line_id, fid, bbox, _text in track_obs:
        best, best_iou = None, 0.0
        for ci, gbox in gt_by_fid.get(fid, []):
            v = _iou(bbox, gbox)
            if v > best_iou:
                best, best_iou = ci, v
        if best is not None and best_iou >= iou_thr:
            pair.append((line_id, best, fid))
    if not pair:
        return {"purity": 0.0, "frag": 0.0, "idsw": 0,
                "matched": 0, "gt_obs": sum(len(v) for v in gt_by_fid.values())}

    # 纯度: 各跟踪行观测里占比最大的真值链的份额(按观测数加权)
    per_track = {}
    for lid, ci, _f in pair:
        per_track.setdefault(lid, Counter())[ci] += 1
    tot = sum(sum(c.values()) for c in per_track.values())
    purity = sum(c.most_common(1)[0][1] for c in per_track.values()) / max(tot, 1)

    # 碎片化: 每条真值链被多少个不同跟踪行覆盖(平均多出的行数)
    per_chain = {}
    for lid, ci, _f in pair:
        per_chain.setdefault(ci, set()).add(lid)
    frag = (sum(len(v) - 1 for v in per_chain.values()) / len(per_chain)
            if per_chain else 0.0)

    # ID 切换(标准 MOT 口径): 逐真值链按帧序看其关联到的跟踪行是否变化
    chain_seq = {}
    for lid, ci, fid in pair:
        chain_seq.setdefault(ci, {}).setdefault(fid, set()).add(lid)
    idsw = 0
    for fid_map in chain_seq.values():
        prev = None
        for fid in sorted(fid_map):
            lids = fid_map[fid]
            if prev is None:
                prev = min(lids)
            elif prev in lids:
                continue
            else:
                idsw += 1
                prev = min(lids)

    return {"purity": purity, "frag": frag, "idsw": idsw, "matched": len(pair),
            "gt_obs": sum(len(v) for v in gt_by_fid.values())}


def run_tracker(frames, tracker_cfg, roi_h, roi_w):
    """跑生产跟踪器, 返回每个跟踪行在两法下的最终文本 + 观测统计 + 逐帧观测。"""
    tracker = MessageBubbleTracker(dict(tracker_cfg), _NullLogger())
    tracker.set_roi_dimensions(roi_h, roi_w)
    with contextlib.redirect_stdout(io.StringIO()):
        for fid, ts, lines in frames:
            tracker.update(fid, ts, lines, allow_new=True)

    rows = []
    track_obs = []
    for tl in tracker.get_all_tracked():
        texts = [o.text for o in tl.observations]
        weights = [max(0.1, float(o.confidence or 0.0)) for o in tl.observations]
        if not texts:
            continue
        rows.append({
            "obs": len(texts),
            "multi_variant": len(set(texts)) >= 2,
            "exact": Counter(texts).most_common(1)[0][0],
            "consensus": consensus_text(texts, weights),
        })
        for o in tl.observations:
            track_obs.append((tl.line_id, o.frame_id, o.bbox, o.text))
    return rows, track_obs


def score(rows, truth):
    """多重集匹配打分: P/R/F1(exact 与 consensus 各一套) + 多变体子集正确率。

    用多重集(Counter)而非集合: 同一文本可能出现多次(如两个 "Link" 元素),
    集合会漏计重复导致精度虚高, 多重集按出现次数配平。
    """
    truth_c = Counter(truth)
    res = {}
    for key in ("exact", "consensus"):
        got_c = Counter(_norm(r[key]) for r in rows)
        matched = sum((got_c & truth_c).values())
        p = matched / max(sum(got_c.values()), 1)
        r_ = matched / max(sum(truth_c.values()), 1)
        f1 = 2 * p * r_ / max(p + r_, 1e-9)
        res[key] = {"P": p, "R": r_, "F1": f1, "assembled": sum(got_c.values())}
    mv = [r for r in rows if r["multi_variant"]]
    res["mv_n"] = len(mv)
    res["mv_exact_ok"] = sum(1 for r in mv if truth_c[_norm(r["exact"])] > 0)
    res["mv_cons_ok"] = sum(1 for r in mv if truth_c[_norm(r["consensus"])] > 0)
    res["truth_n"] = sum(truth_c.values())
    res["tracked_n"] = len(rows)
    return res


def report(stem, fidelity, s):
    print(f"\n=== {stem}  [fidelity={fidelity}] ===")
    print(f"  真值行 = {s['truth_n']}   跟踪行 = {s['tracked_n']}")
    for key, label in (("exact", "整行多数(旧)"), ("consensus", "字符共识(新)")):
        m = s[key]
        print(f"  {label}: P={m['P']:.1%}  R={m['R']:.1%}  F1={m['F1']:.1%}"
              f"   (装配行 {m['assembled']})")
    if s["mv_n"]:
        print(f"  多变体跟踪行 = {s['mv_n']}"
              f"   正确: exact {s['mv_exact_ok']}/{s['mv_n']}"
              f"    consensus {s['mv_cons_ok']}/{s['mv_n']}")
    ids = s.get("ids")
    if ids:
        print(f"  跟踪ID(BOVText式): 纯度={ids['purity']:.1%}"
              f"  碎片化={ids['frag']:.2f}  ID切换={ids['idsw']}"
              f"   (关联 {ids['matched']}/{ids['gt_obs']} 观测)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "synthetic" / "labels"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--noise", type=float, default=0.3,
                    help="noisy 保真度下的逐字符错误率")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("stems", nargs="*", help="场景 stem; 缺省=全部")
    args = ap.parse_args()

    cfg = Config(args.config)
    tracker_cfg = cfg.get("text_tracker") or {}
    label_dir = Path(args.labels)
    stems = args.stems or sorted(p.name[:-len(".labels.jsonl")]
                                 for p in label_dir.glob("*.labels.jsonl"))
    if not stems:
        print(f"未找到标签: {label_dir}/*.labels.jsonl")
        return

    agg = {}
    for stem in stems:
        meta, truth, frames, chains = load_scenario(label_dir, stem)
        for fidelity in ("perfect", "noisy"):
            use = frames if fidelity == "perfect" else \
                inject_noise(frames, args.noise, args.seed)
            rows, track_obs = run_tracker(use, tracker_cfg,
                                          meta["roi_h"], meta["roi_w"])
            s = score(rows, truth)
            s["ids"] = score_ids(track_obs, chains)
            report(stem, fidelity, s)
            a = agg.setdefault(fidelity, {"matched_e": 0, "matched_c": 0,
                                          "assembled_e": 0, "assembled_c": 0,
                                          "truth": 0, "idm": 0, "idtot": 0,
                                          "idfrag": 0, "idn": 0, "idsw": 0,
                                          "gt_obs": 0})
            a["truth"] += s["truth_n"]
            a["assembled_e"] += s["exact"]["assembled"]
            a["assembled_c"] += s["consensus"]["assembled"]
            a["matched_e"] += round(s["exact"]["P"] * s["exact"]["assembled"])
            a["matched_c"] += round(s["consensus"]["P"] * s["consensus"]["assembled"])
            ids = s["ids"]
            a["idm"] += ids["matched"]
            a["idtot"] += ids["gt_obs"]
            a["idfrag"] += ids["frag"]
            a["idn"] += 1
            a["idsw"] += ids["idsw"]
            a["gt_obs"] += ids["gt_obs"]

    print("\n" + "=" * 56)
    print("汇总 / TOTAL (跨场景宏平均, 近似)")
    for fidelity, a in agg.items():
        print(f"  [{fidelity}] 真值总行 = {a['truth']}")
        for key, lab in (("e", "整行多数(旧)"), ("c", "字符共识(新)")):
            p = a[f"matched_{key}"] / max(a[f"assembled_{key}"], 1)
            r_ = a[f"matched_{key}"] / max(a["truth"], 1)
            f1 = 2 * p * r_ / max(p + r_, 1e-9)
            print(f"    {lab}: P={p:.1%}  R={r_:.1%}  F1={f1:.1%}")
        if a["idn"]:
            print(f"    跟踪ID(BOVText式): 关联率={a['idm'] / max(a['gt_obs'], 1):.1%}"
                  f"  平均碎片={a['idfrag'] / a['idn']:.2f}"
                  f"  ID切换总数={a['idsw']}")


if __name__ == "__main__":
    main()
