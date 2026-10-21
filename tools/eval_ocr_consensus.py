# -*- coding: utf-8 -*-
"""
OCR 多帧共识评测 harness (离线, 无需视频/API, 可 CI 运行)

用途 / Purpose:
    以 ground_truth/<stem>.jsonl(每 0.5s 一次的密集 OCR, bbox 已是 ROI 局部坐标)
    为基准, 量化"同一文本行被反复 OCR"时的识别抖动, 以及字符级共识投票(#12)
    相对"整行精确多数"在 收敛率 与 留一稳定性 上的增益。

方法 / Method:
    用**生产同款**的文本行跟踪器(MessageBubbleTracker: 空间IoU/中心距 + 文本相似
    度 + 时间戳隔离 + 过期)把 GT 行按帧序成簇 —— 直接复现共识在生产中生效的口径。
    纯文本聚类会把"语义不同但字形相近"的行(如 3000/1000 元、认证一/二/三)误并,
    因此必须带空间约束。
    每个簇(≥2 个观测)同时算:
        exact_vote = 整行精确字符串多数(旧行为)
        consensus  = 字符级共识(新行为)
      并度量:
        收敛率(share)   = 与结果完全相同的观测占比(越高越稳定)
        留一稳定性(LOO) = 逐个剔除一个观测后结果不变的比例(越高越鲁棒)

用法 / Usage:
    python tools/eval_ocr_consensus.py            # 评测全部 ground truth
    python tools/eval_ocr_consensus.py zy xxr     # 只评测指定 stem
"""
import argparse
import contextlib
import io
import json
import sys
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


def load_frames(gt_path: Path):
    """读 GT, 返回 [(frame_id, timestamp, [TextLine, ...]), ...](按帧序)。"""
    frames = []
    with open(gt_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            fid = int(row["frame_id"])
            ts = float(row.get("timestamp", 0.0))
            lines = []
            for li in row.get("lines", []):
                bbox = li.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                lines.append(TextLine(
                    text=str(li.get("text", "")),
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    confidence=float(li.get("conf", 1.0) or 1.0),
                    frame_id=fid, timestamp=ts,
                ))
            frames.append((fid, ts, lines))
    frames.sort(key=lambda x: x[0])
    return frames


def _roi_height_estimate(frames):
    """ROI 高度估计: 取所有行 y+h 的最大值(ROI 已裁掉状态栏, 其高度即内容区高度)。"""
    mx = 0
    for _, _, lines in frames:
        for l in lines:
            mx = max(mx, l.bbox[1] + l.bbox[3])
    return max(mx, 1)


def build_clusters(frames, tracker_cfg):
    """用生产跟踪器把 GT 行成簇, 返回 (clusters, total_lines)。

    clusters: [(texts, weights), ...](仅 ≥2 观测的跟踪行)
    total_lines: GT 中所有观测行总数(含重复), 用于口径完整的统计。

    注: 跟踪器的 _log_vote 会向 stdout 打印 [投票] 事件, 这里临时重定向屏蔽,
    避免污染评测报告。
    """
    tracker = MessageBubbleTracker(dict(tracker_cfg), _NullLogger())
    tracker.set_roi_dimensions(_roi_height_estimate(frames), 0)
    with contextlib.redirect_stdout(io.StringIO()):
        for fid, ts, lines in frames:
            tracker.update(fid, ts, lines, allow_new=True)

    total_lines = sum(len(lines) for _, _, lines in frames)
    clusters = []
    for tl in tracker.get_all_tracked():
        obs = tl.observations
        if len(obs) >= 2:
            clusters.append((
                [o.text for o in obs],
                [max(0.1, float(o.confidence or 0.0)) for o in obs],
            ))
    return clusters, total_lines


def exact_majority(texts):
    """旧行为: 整行精确字符串多数。"""
    return Counter(texts).most_common(1)[0][0]


def eval_clusters(clusters):
    """评估所有簇。

    除整体口径外, 额外单列"多变体簇"(簇内不同文本 ≥2)的对比 —— 只有这类簇
    投票才会真正起作用, 整体口径会被大量"单变体簇"(两法都原样返回)稀释。
    并统计 consensus 输出"与任一观测都不同"的簇数(重建串): 这是本算法的设计
    能力(把各帧分别错在不同位置的字符拼回正确串), 非缺陷, 仅用于确认该路径命中。
    """
    stats = {"clusters": 0, "obs": 0, "variants_sum": 0, "single_variant": 0,
             "exact_share": 0.0, "cons_share": 0.0,
             "exact_loo_hit": 0, "exact_loo_tot": 0,
             "cons_loo_hit": 0, "cons_loo_tot": 0,
             "diff_clusters": 0, "chimera": 0,
             "mv_clusters": 0, "mv_obs": 0,
             "mv_exact_share": 0.0, "mv_cons_share": 0.0,
             "mv_exact_loo_hit": 0, "mv_exact_loo_tot": 0,
             "mv_cons_loo_hit": 0, "mv_cons_loo_tot": 0,
             "mv_diff": 0, "mv_chimera": 0}
    examples = []
    for texts, weights in clusters:
        n = len(texts)
        uniq = set(texts)
        ex = exact_majority(texts)
        cons = consensus_text(texts, weights)
        multi = len(uniq) >= 2

        stats["clusters"] += 1
        stats["obs"] += n
        stats["variants_sum"] += len(uniq)
        if not multi:
            stats["single_variant"] += 1
        stats["exact_share"] += sum(1 for t in texts if t == ex) / n
        stats["cons_share"] += sum(1 for t in texts if t == cons) / n
        if cons not in uniq:
            stats["chimera"] += 1

        if multi:
            stats["mv_clusters"] += 1
            stats["mv_obs"] += n
            stats["mv_exact_share"] += sum(1 for t in texts if t == ex) / n
            stats["mv_cons_share"] += sum(1 for t in texts if t == cons) / n
            if cons not in uniq:
                stats["mv_chimera"] += 1
            if cons != ex:
                stats["mv_diff"] += 1

        for k in range(n):
            sub_t = texts[:k] + texts[k + 1:]
            sub_w = weights[:k] + weights[k + 1:]
            stats["exact_loo_tot"] += 1
            stats["cons_loo_tot"] += 1
            e_hit = exact_majority(sub_t) == ex
            c_hit = consensus_text(sub_t, sub_w) == cons
            if e_hit:
                stats["exact_loo_hit"] += 1
            if c_hit:
                stats["cons_loo_hit"] += 1
            if multi:
                stats["mv_exact_loo_tot"] += 1
                stats["mv_cons_loo_tot"] += 1
                if e_hit:
                    stats["mv_exact_loo_hit"] += 1
                if c_hit:
                    stats["mv_cons_loo_hit"] += 1

        if cons != ex:
            stats["diff_clusters"] += 1
            if len(examples) < 8:
                examples.append((Counter(texts), ex, cons))
    return stats, examples


def report(stem, n_frames, total_lines, clusters, stats, examples):
    c = max(stats["clusters"], 1)
    print(f"\n=== {stem} ===")
    print(f"  GT 帧数 = {n_frames}   观测文本行(含重复) = {total_lines}")
    print(f"  跟踪簇(≥2观测) = {stats['clusters']}   簇内观测 = {stats['obs']}")
    if not stats["clusters"]:
        print("  (无可评估簇)")
        return
    print(f"  平均变体数/簇  = {stats['variants_sum'] / c:.2f}"
          f"   单变体簇 = {stats['single_variant']} / {stats['clusters']}"
          f" ({stats['single_variant'] / c:.0%})")
    print(f"  收敛率   exact = {stats['exact_share'] / c:.1%}"
          f"    consensus = {stats['cons_share'] / c:.1%}")
    el = stats["exact_loo_hit"] / max(stats["exact_loo_tot"], 1)
    cl = stats["cons_loo_hit"] / max(stats["cons_loo_tot"], 1)
    print(f"  留一稳定性 exact = {el:.1%}    consensus = {cl:.1%}")
    print(f"  两法结论不同的簇 = {stats['diff_clusters']} / {stats['clusters']}"
          f"   consensus 重建串 = {stats['chimera']}")
    mc = max(stats["mv_clusters"], 1)
    if stats["mv_clusters"]:
        print(f"  [多变体簇] 簇 = {stats['mv_clusters']} (观测 {stats['mv_obs']})")
        print(f"    收敛率   exact = {stats['mv_exact_share'] / mc:.1%}"
              f"    consensus = {stats['mv_cons_share'] / mc:.1%}")
        mel = stats["mv_exact_loo_hit"] / max(stats["mv_exact_loo_tot"], 1)
        mcl = stats["mv_cons_loo_hit"] / max(stats["mv_cons_loo_tot"], 1)
        print(f"    留一稳定性 exact = {mel:.1%}    consensus = {mcl:.1%}")
        print(f"    两法结论不同 = {stats['mv_diff']} / {stats['mv_clusters']}"
              f"   consensus 重建串 = {stats['mv_chimera']}")
    if examples:
        print("  样例(exact -> consensus):")
        for counter, ex, cons in examples:
            variants = ", ".join(f"{t!r}x{n}" for t, n in counter.most_common(4))
            print(f"    {variants}")
            print(f"      exact={ex!r}  ->  consensus={cons!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*",
                    help="ground_truth/<stem>.jsonl 的 stem; 缺省=全部")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = Config(args.config)
    tracker_cfg = cfg.get("text_tracker") or {}

    gt_dir = ROOT / "ground_truth"
    stems = args.stems or sorted(p.stem for p in gt_dir.glob("*.jsonl"))
    if not stems:
        print("未找到 ground_truth/*.jsonl")
        return

    total = {"clusters": 0, "obs": 0, "variants_sum": 0, "single_variant": 0,
             "exact_share": 0.0, "cons_share": 0.0,
             "exact_loo_hit": 0, "exact_loo_tot": 0,
             "cons_loo_hit": 0, "cons_loo_tot": 0,
             "diff_clusters": 0, "chimera": 0,
             "mv_clusters": 0, "mv_obs": 0,
             "mv_exact_share": 0.0, "mv_cons_share": 0.0,
             "mv_exact_loo_hit": 0, "mv_exact_loo_tot": 0,
             "mv_cons_loo_hit": 0, "mv_cons_loo_tot": 0,
             "mv_diff": 0, "mv_chimera": 0}
    for stem in stems:
        gt_path = gt_dir / f"{stem}.jsonl"
        if not gt_path.exists():
            print(f"[跳过] 无 {gt_path.name}")
            continue
        frames = load_frames(gt_path)
        clusters, total_lines = build_clusters(frames, tracker_cfg)
        stats, examples = eval_clusters(clusters)
        report(stem, len(frames), total_lines, clusters, stats, examples)
        for k in total:
            total[k] += stats.get(k, 0)

    if total["clusters"]:
        c = total["clusters"]
        print("\n" + "=" * 52)
        print("汇总 / TOTAL")
        print(f"  跟踪簇 = {total['clusters']}   观测 = {total['obs']}"
              f"   单变体簇 = {total['single_variant']} / {total['clusters']}"
              f" ({total['single_variant'] / c:.0%})")
        print(f"  收敛率   exact = {total['exact_share'] / c:.1%}"
              f"    consensus = {total['cons_share'] / c:.1%}")
        print(f"  留一稳定性 exact = "
              f"{total['exact_loo_hit'] / max(total['exact_loo_tot'], 1):.1%}"
              f"    consensus = "
              f"{total['cons_loo_hit'] / max(total['cons_loo_tot'], 1):.1%}")
        print(f"  两法结论不同的簇 = {total['diff_clusters']} / {total['clusters']}"
              f"   consensus 重建串 = {total['chimera']}")
        if total["mv_clusters"]:
            mc = total["mv_clusters"]
            print(f"  [多变体簇] 簇 = {total['mv_clusters']}"
                  f" (观测 {total['mv_obs']})")
            print(f"    收敛率   exact = {total['mv_exact_share'] / mc:.1%}"
                  f"    consensus = {total['mv_cons_share'] / mc:.1%}")
            print(f"    留一稳定性 exact = "
                  f"{total['mv_exact_loo_hit'] / max(total['mv_exact_loo_tot'], 1):.1%}"
                  f"    consensus = "
                  f"{total['mv_cons_loo_hit'] / max(total['mv_cons_loo_tot'], 1):.1%}")
            print(f"    两法结论不同 = {total['mv_diff']} / {total['mv_clusters']}"
                  f"   consensus 重建串 = {total['mv_chimera']}")


if __name__ == "__main__":
    main()
