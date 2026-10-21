# -*- coding: utf-8 -*-
"""语音聊天录屏的批量基准 / Batch benchmark for voice-chat screen recordings.

一次性把**所有已登记场景**串起来: (可选)批量生成合成视频 → 跑提取管线 →
逐条 GT 评测 → 聚合成 P/R/F1 总账。用于回归: 改动管线后重跑本脚本, 与
`--baseline` 保存的基线对比, 一眼看出哪个场景退化。

场景登记 / Scene registry:
    每个场景是 (视频名, GT 源文件) 二元组; GT 源文件是 tools/gen_voice_chat_video.py
    能解析的 7 列 TSV(或 form=dialog) 协议 md。视频落在 <out>/videos/<视频名>.mp4,
    评测用的 GT 取 <out>/gt/<视频名>.md(生成器落盘的同一份)。新增场景时在 SCENES
    里登记一行即可。

用法 / Usage:
    python tools/bench_voice_gt.py                     # 评测已生成的场景(缺件则跳过)
    python tools/bench_voice_gt.py --gen               # 先生成缺件/全量视频
    python tools/bench_voice_gt.py --run               # 先跑提取管线
    python tools/bench_voice_gt.py --gen --run         # 生成 + 跑管线 + 评测
    python tools/bench_voice_gt.py --save baseline.json          # 存基线
    python tools/bench_voice_gt.py --baseline baseline.json     # 与基线对比
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))       # tools/ 下的 eval_voice_gt
from eval_voice_gt import evaluate, ROOT                        # noqa: E402

# ---------------------------------------------------------------------------
# 场景登记: (视频名, GT 源文件) / scene registry: (video name, GT source file)
# ---------------------------------------------------------------------------
SCENES = [
    ("v2_synth",        "gt/语音聊天30条_GT与录制协议_v2.md"),
    ("synth_text_long", "gt/语音聊天_场景_纯文本与超长语音.md"),
    ("synth_dialog",    "gt/语音聊天_场景_系统弹窗.md"),
    ("synth_edge",      "gt/语音聊天_场景_边界与负样本.md"),
    ("v6_catcare",      "gt/语音聊天30条_GT与录制协议_v6_catcare.md"),
    ("v7_realistic",    "gt/语音聊天30条_GT与录制协议_v7_realistic.md"),
]
OUT_DEFAULT = ROOT / "synthetic" / "voice_chat"
PY = sys.executable                     # 必须是当前解释器(conda gnt 环境)


def gen_video(name: str, gt_src: str, out_dir: Path, fps: int) -> int:
    cmd = [PY, str(ROOT / "tests" / "gen_voice_chat_video.py"),
           "--gt", str(ROOT / gt_src), "--name", name,
           "--fps", str(fps), "--out", str(out_dir)]
    print(f"[gen] {name}  <- {gt_src}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def run_pipeline(video: Path) -> int:
    cmd = [PY, str(ROOT / "main.py"), "-v", str(video)]
    print(f"[run] {video.name}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--gen", action="store_true", help="先生成视频(缺件或全量)")
    ap.add_argument("--only-missing", action="store_true",
                    help="--gen 时只生成缺件的场景")
    ap.add_argument("--run", action="store_true", help="先生成提取结果")
    ap.add_argument("--th", type=float, default=0.80)
    ap.add_argument("--save", default="", help="把本次结果写入该 JSON 作为基线")
    ap.add_argument("--baseline", default="", help="与该 JSON 基线对比")
    args = ap.parse_args()

    out_dir = Path(args.out)
    rows = []
    for name, gt_src in SCENES:
        video = out_dir / "videos" / f"{name}.mp4"
        gt_eval = out_dir / "gt" / f"{name}.md"

        if args.gen and not (args.only_missing and video.exists()):
            if gen_video(name, gt_src, out_dir, args.fps) != 0:
                print(f"  !! 生成失败, 跳过 / gen failed, skip: {name}")
                continue
        if not video.exists():
            print(f"[skip] {name}: 缺视频 {video} (先跑 --gen)")
            continue
        if args.run:
            run_pipeline(video)

        stem = name
        if not (ROOT / "output" / f"extracted_content_{stem}.json").exists():
            print(f"[skip] {name}: 缺提取结果 output/extracted_content_{stem}.json (先跑 --run)")
            continue
        gt_used = gt_eval if gt_eval.exists() else (ROOT / gt_src)
        r = evaluate(stem, gt_used, args.th)
        hallu = sum(1 for _i, _s, v, _t in r["neg_report"] if v.startswith("幻觉"))
        rows.append({
            "name": name, "th": args.th,
            "n_gt": r["n_gt"], "n_reach": r["n_reach"], "n_neg": r["n_neg"],
            "pred": r["pred"], "tp": r["tp"], "gt_reach": r["gt_reach"],
            "P": r["P"], "R": r["R"], "F": r["F"],
            "ok_a": r["ok_a"], "ok_b": r["ok_b"], "ok_c": r["ok_c"],
            "n_v": r["n_v"], "n_a": r["n_a"], "covered": r["n_covered"],
            "uncovered": r["uncovered"], "hallucinated_neg": hallu,
            "redundant_lines": r["n_dup"], "unmatched": r["unmatched"],
            "side_n": r["side_n"], "side_ok": r["side_ok"],
            "side_acc": r["side_acc"], "side_macro_f": r["side_macro_f"],
        })

    if not rows:
        print("没有任何可评测的场景 / nothing evaluated")
        return

    # ---- 逐场景表 ----
    print("\n" + "=" * 100)
    print("语音聊天合成基准 / voice-chat synthetic benchmark")
    print("=" * 100)
    print(f"{'scene':<18} {'P':>7} {'R':>7} {'F1':>7} {'Pred':>5} {'GT':>4} "
          f"{'A':>7} {'B':>7} {'C':>7} {'方向Acc':>8} {'方向F1':>7} {'负样本幻':>8} {'冗余':>5}")
    print("-" * 118)
    for r in rows:
        a = f"{r['ok_a']}/{r['n_reach']}"
        b = f"{r['ok_b']}/{r['n_v']}"
        c = f"{r['ok_c']}/{r['n_a']}"
        sa = f"{r['side_acc']:.3f}" if r["side_n"] else "  -  "
        sf = f"{r['side_macro_f']:.3f}" if r["side_n"] else "  -  "
        print(f"{r['name']:<18} {r['P']:>7.3f} {r['R']:>7.3f} {r['F']:>7.3f} "
              f"{r['pred']:>5} {r['gt_reach']:>4} {a:>7} {b:>7} {c:>7} "
              f"{sa:>8} {sf:>7} {r['hallucinated_neg']:>8} {r['redundant_lines']:>5}")

    # ---- 聚合(微平均: 把各场景的 TP/Pred/GT 直接相加) ----
    tp = sum(r["tp"] for r in rows)
    pred = sum(r["pred"] for r in rows)
    gtr = sum(r["gt_reach"] for r in rows)
    P = tp / pred if pred else 0.0
    R = tp / gtr if gtr else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    hallu = sum(r["hallucinated_neg"] for r in rows)
    side_n = sum(r["side_n"] for r in rows)
    side_ok = sum(r["side_ok"] for r in rows)
    side_acc = side_ok / side_n if side_n else 0.0
    side_macro = (sum(r["side_macro_f"] for r in rows if r["side_n"]) /
                  max(1, sum(1 for r in rows if r["side_n"])))
    print("-" * 118)
    print(f"{'ALL(微平均)':<18} {P:>7.3f} {R:>7.3f} {F:>7.3f} "
          f"{pred:>5} {gtr:>4}  TP={tp}  负样本幻觉={hallu}  "
          f"方向Acc={side_acc:.3f} ({side_ok}/{side_n})  方向宏F1={side_macro:.3f}")
    macro_f = sum(r["F"] for r in rows) / len(rows)
    print(f"{'ALL(宏平均F1)':<18} {macro_f:>7.3f}   场景数={len(rows)}")
    for r in rows:
        if r["uncovered"]:
            print(f"  ! {r['name']} 未覆盖 GT id: {r['uncovered']}")

    # ---- 基线对比(回归) ----
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        bmap = {b["name"]: b for b in base.get("scenes", [])}
        print("\n与基线对比 / vs baseline:")
        for r in rows:
            b = bmap.get(r["name"])
            if not b:
                print(f"  {r['name']:<18} (基线无此场景)")
                continue
            d_f = r["F"] - b["F"]
            d_p = r["P"] - b["P"]
            d_r = r["R"] - b["R"]
            d_sa = r["side_acc"] - b.get("side_acc", 0.0)
            flag = "" if d_f >= -1e-9 else "  <== 退化"
            sflag = "" if d_sa >= -1e-9 else "  <== 方向退化"
            print(f"  {r['name']:<18} ΔF1={d_f:+.3f} ΔP={d_p:+.3f} ΔR={d_r:+.3f} "
                  f"Δ方向Acc={d_sa:+.3f}{flag}{sflag}")
        dF = F - base.get("micro", {}).get("F", 0.0)
        dSA = side_acc - base.get("micro", {}).get("side_acc", 0.0)
        print(f"  {'ALL(微平均)':<18} ΔF1={dF:+.3f}  Δ方向Acc={dSA:+.3f}")

    if args.save:
        payload = {"th": args.th, "scenes": rows,
                   "micro": {"tp": tp, "pred": pred, "gt": gtr, "P": P, "R": R, "F": F,
                             "side_ok": side_ok, "side_n": side_n,
                             "side_acc": side_acc, "side_macro_f": side_macro},
                   "macro_F": macro_f, "hallucinated_neg": hallu}
        Path(args.save).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n基线已写入 / baseline saved: {args.save}")


if __name__ == "__main__":
    main()
