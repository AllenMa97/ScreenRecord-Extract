# -*- coding: utf-8 -*-
"""合成带标签数据集生成器 / Synthetic labeled-dataset generator

用途 / Purpose:
    复用 tests/gen_long_video.py 的渲染引擎, 用小规模、确定性的对话脚本渲染出
    "合成录屏", 并同步落盘**精确标签**, 从而得到带 ground truth 的数据集。
    这是当前唯一能同时给出"内容答案 + bbox + 装配/去重结构答案"的来源。

产出 / Output:
    synthetic/videos/<video_id>.mp4          合成的聊天录屏
    synthetic/labels/<video_id>.labels.jsonl  每帧每行文本(真值): text + bbox[全帧坐标]
    synthetic/labels/<video_id>.meta.json     真值消息表 / 滚动序列 / 帧类型 / ROI 提示

坐标契约 / Coordinate contract:
    labels.jsonl 的 bbox 是**全帧坐标**。pipeline 送 OCR 的是 ROI 裁剪帧, 换算见
    meta.json: ROI = (0, title_height, W, H - title_height - nav_height)。

用法 / Usage:
    python tests/gen_labeled_sample.py
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_long_video as G  # noqa: E402

OUT_VIDEO = os.path.join(ROOT, "synthetic", "videos")
OUT_LABEL = os.path.join(ROOT, "synthetic", "labels")
RESOLUTION = (540, 960)


def pick_style(app_value: str = "CUSTOM_APP"):
    """选一个字体可用的样式(项目字体目录里只有 msyh/simhei/simkai/Droid/Roboto)。"""
    styles = G.generate_all_styles()
    for s in styles:
        if s.app_type.value == app_value:
            return s
    return styles[0]


# ---------------------------------------------------------------------------
# 场景脚本 / Scenario scripts
# ---------------------------------------------------------------------------
def scenario_basic():
    """短对话, 内容基本不滚动(用于测文本装配/去重的最基本正确性)。"""
    return [
        G.ChatMessage("peer", "在吗？", True),
        G.ChatMessage("me", "在的，什么事", False),
        G.ChatMessage("peer", "帮我看看这个链接靠不靠谱", True),
        G.ChatMessage("me", "好，我看看", False),
        G.ChatMessage("peer", "说是充值返现，年利率12%", True),
        G.ChatMessage("me", "这大概率是诈骗，别点", False),
        G.ChatMessage("peer", "好的，谢谢提醒", True),
    ]


def scenario_scroll():
    """长对话, 强制产生滚动(用于测视觉去重的召回/冗余)。"""
    msgs = []
    for i in range(30):
        recv = (i % 2 == 0)
        msgs.append(G.ChatMessage(
            "peer" if recv else "me",
            f"第{i + 1}条消息：请确认订单号 SN{i + 1:04d} 的收款状态。",
            recv,
        ))
    return msgs


def scenario_mixed():
    """混合类型: 文本 + 系统提示 + 语音转写 + 长文本换行。"""
    return [
        G.ChatMessage("system", "以下为对方发起的转账请求", True, msg_type="system"),
        G.ChatMessage("peer", "我先转你 5000 元定金", True),
        G.ChatMessage("peer", "语音", True, msg_type="voice",
                      voice_duration=4.0, voice_transcribed="你先把保证金交了我们就发货"),
        G.ChatMessage("me", "为什么要先交保证金？这不是正规流程吧", False),
        G.ChatMessage("peer", "这是公司规定，很多人都是这样操作的，你放心，"
                             "我们这边做这个业务已经很多年了，从来没有出过问题，"
                             "你只要按流程走就一定没问题。", True),
        G.ChatMessage("me", "我还是觉得不对劲", False),
    ]


SCENARIOS = [
    ("syn_basic", scenario_basic, 5, 10),
    ("syn_scroll", scenario_scroll, 10, 10),
    ("syn_mixed", scenario_mixed, 6, 10),
]


def render_one(video_id, msgs, duration, fps, style, loader, profile="normal_user"):
    task = G.VideoTask(
        video_id=video_id, style=style, device=list(G.DeviceType)[0],
        fps=fps, duration_sec=duration, behavior_profile=profile,
        case_id="syn", resolution=RESOLUTION,
    )
    renderer = G.FrameRenderer(RESOLUTION[0], RESOLUTION[1], asset_loader=loader)
    return G.render_single_video(task, msgs, OUT_VIDEO, renderer, label_dir=OUT_LABEL)


def main():
    os.makedirs(OUT_VIDEO, exist_ok=True)
    os.makedirs(OUT_LABEL, exist_ok=True)

    loader = None
    try:
        loader = G.IndexedAssetLoader()
        loader.add_avatar_pool(os.path.join(ROOT, "tests", "avatars"))
    except Exception as e:
        print(f"[WARN] 素材加载失败, 使用占位图: {e}")

    style = pick_style()
    print(f"[INFO] 使用样式: {style.app_type.value} "
          f"title_h={style.title_height} nav_h={style.nav_height} font={os.path.basename(style.font_path)}")

    for video_id, script, duration, fps in SCENARIOS:
        msgs = script()
        path = render_one(video_id, msgs, duration, fps, style, loader)
        labels_path = os.path.join(OUT_LABEL, f"{video_id}.labels.jsonl")
        n_frames = 0
        n_lines = 0
        seen_texts = set()
        if os.path.exists(labels_path):
            with open(labels_path, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    r = json.loads(ln)
                    n_frames = max(n_frames, int(r["frame_id"]) + 1)
                    n_lines += 1
                    seen_texts.add(r["text"])
        print(f"[DONE] {video_id}: 视频={os.path.basename(path)} "
              f"帧≈{n_frames} 标签行={n_lines} 去重文本={len(seen_texts)}")

    print("\n[样本标签示例] syn_basic 前 6 条:")
    sample = os.path.join(OUT_LABEL, "syn_basic.labels.jsonl")
    if os.path.exists(sample):
        with open(sample, encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if i >= 6:
                    break
                print("  ", ln.strip())


if __name__ == "__main__":
    main()
