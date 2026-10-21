# -*- coding: utf-8 -*-
"""合成"语音聊天录屏"生成器(AI 口播) / Synthetic voice-chat recording generator

为什么 / Why:
    人工录制一条语音聊天录屏(逐条发、每条间隔 ≥3s、还要外放)成本极高且不可复现。
    本脚本用 TTS(edge-tts, AI 口播) + 现有渲染引擎合成同形态录屏:
      - 画面: 消息按时间**逐条追加**, 视图贴在底部(真实 App 的追加式滚动), 末尾
              可选一段"滚到顶再滚回"的演示(考验视觉去重);
      - 声音: 每条语音用 TTS 生成人声, 按"气泡出现时刻"混入 16kHz 单声道音轨,
              与画面一起封装成 mp4(PyAV 自带完整 ffmpeg, 不依赖系统 ffmpeg);
      - 真值: 同步落盘 GT TSV(与 gt/语音聊天30条_GT与录制协议_v2.md 同 7 列格式,
              可直接被 tools/eval_voice_gt.py 解析) + 逐帧行标签 + meta。

产出 / Output:
    <out>/videos/<video_id>.mp4
    <out>/labels/<video_id>.labels.jsonl   每帧每行文本(text + bbox[全帧坐标])
    <out>/labels/<video_id>.meta.json      真值消息 / 出现时刻 / 语音发音时刻 / ROI
    <out>/gt/<video_id>.md                 含 ```tsv 机读块(7 列)

用法 / Usage:
    python tests/gen_voice_chat_video.py                     # v2 全量 30 条, 1 条视频
    python tests/gen_voice_chat_video.py --ids 1-8 --fps 10  # 只生成前 8 条(快速冒烟)
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import av  # noqa: E402

import gen_long_video as G  # noqa: E402
from src.audio.audio_loader import decode_audio  # noqa: E402

# 9:20 竖屏, 对齐真实录屏样本(自行模拟对话记录*.mp4 = 432x960 = 9:20);
# 旧值 540x960(9:16) 比真实机型明显偏宽。
# / 9:20 portrait, matching the real recordings (432x960); the old 540x960 was 9:16.
RESOLUTION = (540, 1200)
SAMPLE_RATE = 16000
VOICE_A = "zh-CN-YunxiNeural"        # 手机①(A) 的说话人 / speaker of phone ①
VOICE_B = "zh-CN-XiaoxiaoNeural"     # 手机②(B) 的说话人 / speaker of phone ②

GAP_S = 3.0            # 相邻两条消息的间隔(录制协议要求 ≥3s) / gap between messages
VOICE_LEAD_S = 0.3     # 气泡出现 → 声音开始播放的延迟 / bubble -> audio lead
HOLD_VOICE_NOAUDIO_S = 2.0   # 未采集到声音的语音气泡停留时长 / silent bubble hold
HOLD_TEXT_S = 1.2      # 文本消息阅读停留 / text reading hold
HOLD_SYSTEM_S = 1.5    # 系统提示行停留时长 / system-line hold
HOLD_DIALOG_S = 5.0    # 系统弹窗停留时长 / system-dialog overlay hold
TAIL_S = 4.0           # 结尾静止(录完后再停留几秒, 让最后一条也有足够的静止帧)
                       # / tail freeze: hold a few seconds so the last message also
                       # gets enough settled frames to be read
SCROLL_UP_S, SCROLL_HOLD_S, SCROLL_BACK_S = 1.2, 0.8, 1.2

# 内容底边与导航栏之间的留白(px)。真实 App 底部是"输入框+安全区"(比本渲染的导航栏
# 高), 管线据此检出的 ROI 底边会比导航栏上沿再高出一段; 若不留白, 刚追加的最后一条
# 消息会整条落在 ROI 之外 → 既不被 OCR, 也不改变 ROI 内像素(去重判为重复而跳过),
# 导致末尾消息在视觉通路丢失。留出这段空白后, 最新一条始终在 ROI 内可见。
# / Bottom inset: the pipeline's ROI bottom sits above the nav bar, so without this
# the newest bubble would fall outside the ROI and be silently dropped.
BOTTOM_INSET = 80

# 系统弹窗按钮文案(两枚): 用于验证管线"弹窗正文+按钮同帧成簇一并丢弃"的过滤
# / System-dialog button labels, used to exercise the OS-chrome filter.
DIALOG_BUTTONS = ("取消", "开启")


@dataclass
class Item:
    """一条待发送的消息(同时是视觉真值与音频真值)。

    kind == "dialog" 时不是聊天消息, 而是叠加在画面上的**系统弹窗**(OS chrome),
    用于验证管线会把"弹窗正文 + 按钮同帧成簇"整体过滤掉、不当作消息输出。
    / kind == "dialog" is an OS dialog overlay (not a chat message), used to
    exercise the pipeline's system-chrome filter.
    """
    idx: int
    sender: str          # "A"(手机①, 我方发送) / "B"(手机②, 对端); dialog/system 行填 "-"
    kind: str            # "voice" | "text" | "sticker" | "transfer" | "dialog" | "system"
    screen: str          # 屏幕文字("-" = 该语音未转文字, 屏幕只有气泡); dialog=弹窗正文
    spoken: str          # 语音内容(text/sticker/transfer 形态时为空)
    captured: bool       # 声音是否进入录屏音轨(False = 故意不进 → 不可达负样本)
    note: str = ""


# ---------------------------------------------------------------------------
# 场景: 读 gt/*_v2.md 的 TSV 机读块 / Scenario from the GT protocol's TSV block
# ---------------------------------------------------------------------------
def load_v2_items(gt_md: str) -> List[Item]:
    md = open(gt_md, encoding="utf-8").read()
    block = re.search(r"```tsv\n(.*?)```", md, re.S).group(1).strip().splitlines()
    items = []
    for ln in block[1:]:
        p = ln.split("\t")
        if len(p) < 6:
            continue
        items.append(Item(
            idx=int(p[0]), sender=p[1].strip(), kind=p[2].strip(),
            screen=p[3].strip(), spoken=p[4].strip(),
            captured=p[5].strip().lower() == "yes",
            note=p[6].strip() if len(p) > 6 else "",
        ))
    return items


def parse_ids(spec: str, n: int) -> List[int]:
    """"1-8,12" → [1..8, 12]; 空 → 全部。"""
    if not spec:
        return list(range(1, n + 1))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [i for i in out if 1 <= i <= n]


# ---------------------------------------------------------------------------
# TTS / edge-tts
# ---------------------------------------------------------------------------
def tts_clip(text: str, voice: str, cache_dir: str) -> str:
    """合成一条语音并缓存, 返回 mp3 路径 / synthesize (cached) and return path."""
    import edge_tts

    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(f"{voice}|{text}".encode("utf-8")).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{key}.mp3")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    async def _run():
        await edge_tts.Communicate(text, voice).save(path)

    asyncio.run(_run())
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"TTS 合成失败 / TTS failed: {text[:24]}")
    return path


# ---------------------------------------------------------------------------
# 时间轴 / Timeline
# ---------------------------------------------------------------------------
def build_timeline(items: Sequence[Item], tts_dir: str,
                   scroll_after: int = 0) -> Tuple[List[float], Dict[int, float],
                                                    Dict[int, str], List[float],
                                                    List[Tuple[float, float, str]]]:
    """把消息序列铺成时间轴。

    返回 (每条**聊天消息**的出现时刻 appear[i], 每条语音的发音时刻 audio_at[idx],
          每条语音的剪辑路径 clip_of[idx], [scroll_win_start, scroll_win_end],
          系统弹窗窗口列表 [(start_s, end_s, 正文)] )。
    scroll_after 表示"第 k 条消息之后插入滚到顶再滚回的演示"(0=不插入)。
    """
    appear: List[float] = []
    audio_at: Dict[int, float] = {}
    clip_of: Dict[int, str] = {}
    scroll_win = [0.0, 0.0]
    dialogs: List[Tuple[float, float, str]] = []

    t = 0.0
    for pos, it in enumerate(items):
        if pos == scroll_after:
            scroll_win = [t, t + SCROLL_UP_S + SCROLL_HOLD_S + SCROLL_BACK_S]
            t = scroll_win[1]
        t += GAP_S
        if it.kind == "dialog":
            # 系统弹窗: 不产生聊天消息, 只在画面叠加一层 OS 弹窗, 持续 HOLD_DIALOG_S
            dialogs.append((t, t + HOLD_DIALOG_S, it.screen))
            t += HOLD_DIALOG_S
            continue
        appear.append(t)
        if it.kind == "voice" and it.captured:
            clip = tts_clip(it.spoken or it.screen, VOICE_A if it.sender == "A"
                            else VOICE_B, tts_dir)
            dur = float(decode_audio(clip, SAMPLE_RATE).audio_duration)
            clip_of[it.idx] = clip
            audio_at[it.idx] = t + VOICE_LEAD_S
            t = t + VOICE_LEAD_S + dur
        elif it.kind == "voice":
            t += HOLD_VOICE_NOAUDIO_S
        elif it.kind == "system":
            t += HOLD_SYSTEM_S
        else:
            t += HOLD_TEXT_S
    return appear, audio_at, clip_of, scroll_win, dialogs


def build_audio(items: Sequence[Item], audio_at: Dict[int, float],
                clip_of: Dict[int, str], total_s: float) -> np.ndarray:
    """把各条语音按发音时刻混入 16kHz 单声道 PCM / mix clips onto a silent canvas."""
    pcm = np.zeros(int(round(total_s * SAMPLE_RATE)) + SAMPLE_RATE, dtype=np.float64)
    for it in items:
        if it.idx not in clip_of:
            continue
        seg = decode_audio(clip_of[it.idx], SAMPLE_RATE).samples
        i0 = int(round(audio_at[it.idx] * SAMPLE_RATE))
        i1 = min(len(pcm), i0 + len(seg))
        if i1 <= i0:
            continue
        pcm[i0:i1] += seg[:i1 - i0]
    peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
    if peak > 0:
        pcm *= (0.6 / peak)      # 留足余量, 避免削顶 / headroom to avoid clipping
    return pcm


# ---------------------------------------------------------------------------
# 画面 / Frames
# ---------------------------------------------------------------------------
# 转账金额与状态: 屏幕文案形如 "¥128.00 已收款 微信转账" / screen_text of a transfer card
_TRANSFER_AMOUNT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
_TRANSFER_STATUS = ("已收款", "等待对方收款", "待确认", "已退还")


def _parse_transfer(screen: str) -> Tuple[float, str]:
    """从屏幕文案里抠出金额与状态 / parse amount + status out of the screen text."""
    m = _TRANSFER_AMOUNT_RE.search(screen)
    amount = float(m.group(1)) if m else 0.0
    status = next((s for s in _TRANSFER_STATUS if s in screen), "已收款")
    return amount, status


def to_chat_message(it: Item, dur: float) -> G.ChatMessage:
    is_recv = (it.sender == "B")
    sid = "peer" if is_recv else "me"
    if it.kind == "system":
        # 系统提示行(时间戳/撤回/加好友等): 居中灰字, 无气泡(负样本之外的真实噪声)
        # / System line: centred grey text, no bubble.
        return G.ChatMessage(sender_id="sys", text=it.screen, is_received=False,
                             msg_type="system")
    if it.kind == "voice":
        return G.ChatMessage(
            sender_id=sid, text="",
            is_received=is_recv, msg_type="voice",
            voice_duration=round(dur, 1),
            voice_transcribed=("" if it.screen == "-" else it.screen),
            has_waveform=True,
        )
    if it.kind == "sticker":
        # 表情包: 只有图形, 屏幕上没有任何可提取文字(负样本口径) / sticker: no text at all
        return G.ChatMessage(sender_id=sid, text="", is_received=is_recv,
                             msg_type="sticker", sticker_name="微笑")
    if it.kind == "transfer":
        amount, status = _parse_transfer(it.screen)
        return G.ChatMessage(sender_id=sid, text="", is_received=is_recv,
                             msg_type="transfer", transfer_amount=amount,
                             transfer_status=status)
    return G.ChatMessage(sender_id=sid, text=it.screen, is_received=is_recv,
                         msg_type="text")


def _layout_params(renderer: G.FrameRenderer, style: G.AppStyle):
    base_font_size = int(16 * style.font_scale / 0.4)
    chat_font = renderer._get_font(style, base_font_size)
    line_height = int(base_font_size * style.line_spacing)
    return chat_font, line_height, int(renderer.W * style.bubble_max_w_ratio), 8


def bottom_scroll_y(msgs: List[G.ChatMessage], renderer: G.FrameRenderer,
                    style: G.AppStyle) -> int:
    """追加式滚动: 视图贴在底部(最新一条紧贴输入区上方; 内容不足一屏时同样贴底)。

    返回值可为负 —— 表示内容整体下移、底边对齐 H-nav-BOTTOM_INSET(真实 App 行为),
    而不是把少量消息顶到标题栏下方。render_chat_frame 用 y_cursor = top - scroll_y,
    故负值即"内容下移"。
    / Append-style bottom-anchored scroll. A negative value means content is pushed
    down so its bottom edge sits at H-nav-BOTTOM_INSET (what real apps do), instead
    of pinning few messages right under the title bar.
    """
    if not msgs:
        return 0
    chat_font, line_height, max_bubble_w, margin_y = _layout_params(renderer, style)
    total = sum(renderer._estimate_msg_height(m, chat_font, max_bubble_w,
                                             line_height, margin_y, style) for m in msgs)
    total += renderer._divider_height(style, line_height, margin_y)
    view = renderer.H - style.nav_height - style.top_offset - BOTTOM_INSET
    return int(round(total - view))


def scroll_y_at(ts: float, win: Sequence[float], y_bottom: int) -> int:
    """滚到顶再滚回: 线性插值 bottom→0→bottom。"""
    s0, s1 = win[0], win[1]
    if s1 <= s0 or not (s0 <= ts < s1):
        return y_bottom
    if ts < s0 + SCROLL_UP_S:
        k = (ts - s0) / SCROLL_UP_S
    elif ts < s0 + SCROLL_UP_S + SCROLL_HOLD_S:
        k = 1.0
    else:
        k = 1.0 - (ts - s0 - SCROLL_UP_S - SCROLL_HOLD_S) / max(1e-6, SCROLL_BACK_S)
    k = min(1.0, max(0.0, k))
    return int(round(y_bottom * (1.0 - k)))


def draw_system_dialog(renderer: "G.FrameRenderer", style: "G.AppStyle",
                       img_bgr: np.ndarray, text: str) -> np.ndarray:
    """在聊天帧上叠加一层系统弹窗(白色圆角对话框 + 正文 + 两枚按钮)。

    弹窗/按钮不是聊天消息, 用于验证管线把它整体当作 OS chrome 过滤掉。
    注意: 这里**不加整屏半透明遮罩** —— 遮罩会把墙纸亮度压到聊天场景判据
    [230,244] 之外, 导致整帧被当成"非聊天界面"丢弃, 反而测不到弹窗文本过滤。
    / Overlay an OS dialog (white rounded card + body + two buttons) on the chat
    frame to exercise the pipeline's system-chrome filter. No full-screen scrim:
    dimming the wallpaper would drop the whole frame as "non-chat" and we would
    never exercise the dialog-text filter.
    """
    from PIL import Image, ImageDraw

    W, H = renderer.W, renderer.H
    pil = Image.fromarray(img_bgr[:, :, ::-1]).convert("RGBA")   # BGR -> RGB

    box_w, box_h = int(W * 0.86), 200
    x0, y0 = (W - box_w) // 2, (H - box_h) // 2
    # 正文用较小字号 + 较宽卡片, 使整句落在**一行**内: OCR 若把正文拆成两三行,
    # 每行单独看都匹配不上弹窗判据(_SYS_DIALOG_RE 要求"是否开启…模式"同现),
    # 就测不到过滤逻辑。
    # Smaller font / wider card keeps the body on ONE line so the dialog-body filter
    # (which needs "是否开启…模式" in the same line) is actually exercised.
    body_font = renderer._get_font(style, 22)
    btn_font = renderer._get_font(style, 24)

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=20,
                         fill=(255, 255, 255, 255),
                         outline=(180, 180, 180, 255), width=2)
    pad = 24
    ty = y0 + pad
    for ln in renderer._wrap_pil(text, body_font, box_w - 2 * pad):
        w = body_font.getbbox(ln)[2] - body_font.getbbox(ln)[0]
        od.text(((W - w) // 2, ty), ln, font=body_font, fill=(34, 34, 34, 255))
        ty += 34
    # 两枚按钮(右对齐): "取消" "开启" —— 按钮文案单独看与正文同形, 靠同帧成簇过滤
    by = y0 + box_h - pad - 34
    bx = x0 + box_w - pad
    for label in reversed(DIALOG_BUTTONS):
        w = btn_font.getbbox(label)[2] - btn_font.getbbox(label)[0]
        bx -= w + 16
        od.text((bx, by), label, font=btn_font, fill=(24, 110, 220, 255))
        bx -= 40
    return np.array(Image.alpha_composite(pil, ov).convert("RGB"))[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# 封装 / Muxing (PyAV, 自带完整 ffmpeg)
# ---------------------------------------------------------------------------
def write_mp4(path: str, fps: int, n_frames: int, frame_fn, pcm: np.ndarray) -> None:
    container = av.open(path, "w", format="mp4")
    vstream = container.add_stream("mpeg4", rate=fps)
    vstream.width, vstream.height = RESOLUTION
    vstream.pix_fmt = "yuv420p"
    vstream.bit_rate = 8_000_000

    astream = container.add_stream("aac", rate=SAMPLE_RATE)
    astream.layout = "mono"
    astream.bit_rate = 64_000

    for f in range(n_frames):
        vf = av.VideoFrame.from_ndarray(frame_fn(f), format="bgr24")
        vf.pts, vf.time_base = f, Fraction(1, fps)
        for pkt in vstream.encode(vf):
            container.mux(pkt)

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
    for i in range(0, len(pcm), SAMPLE_RATE):
        chunk = (pcm[i:i + SAMPLE_RATE] * 32767).astype(np.int16).reshape(1, -1)
        af = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
        af.sample_rate = SAMPLE_RATE
        af.pts, af.time_base = i, Fraction(1, SAMPLE_RATE)
        for rf in resampler.resample(af):
            for pkt in astream.encode(rf):
                container.mux(pkt)
    for rf in resampler.resample(None):
        for pkt in astream.encode(rf):
            container.mux(pkt)
    for pkt in vstream.encode(None):
        container.mux(pkt)
    for pkt in astream.encode(None):
        container.mux(pkt)
    container.close()


# ---------------------------------------------------------------------------
# 主流程 / Main
# ---------------------------------------------------------------------------
def render_one(video_id: str, items: List[Item], out_dir: str, fps: int,
               scroll_after: int, use_image_wallpaper: bool = True,
               wallpaper_image: str = "") -> str:
    # 用 WeChat 浅色样式: 墙纸亮度 ~237, 落在管线 chat_scene 判定带 [230,244] 内
    # (深色样式/纯白底会被判成"非聊天界面"而整帧丢弃 OCR 行)。
    # 默认铺"纹理壁纸图"(wallpaper_image), use_image_wallpaper=False 回退纯平底色。
    # / Light WeChat style whose wallpaper (~237) sits in the pipeline's chat-scene
    # brightness band; image wallpaper by default, flat colour as a fallback.
    style = G.make_wechat_realistic_style(use_image_wallpaper, wallpaper_image)
    renderer = G.FrameRenderer(RESOLUTION[0], RESOLUTION[1])
    tts_dir = os.path.join(out_dir, "tts_cache")

    appear, audio_at, clip_of, scroll_win, dialogs = build_timeline(
        items, tts_dir, scroll_after)
    chat_items = [it for it in items if it.kind != "dialog"]
    durs = {it.idx: (float(decode_audio(clip_of[it.idx], SAMPLE_RATE).audio_duration)
                     if it.idx in clip_of else 0.0) for it in items}
    end_s = appear[-1] + durs.get(chat_items[-1].idx, HOLD_TEXT_S) + VOICE_LEAD_S
    if dialogs:
        end_s = max(end_s, max(d[1] for d in dialogs))
    total_s = end_s + TAIL_S
    n_frames = int(round(total_s * fps))

    msgs = [to_chat_message(it, durs[it.idx]) for it in chat_items]
    pcm = build_audio(items, audio_at, clip_of, total_s)

    label_dir = os.path.join(out_dir, "labels")
    video_dir = os.path.join(out_dir, "videos")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    out_path = os.path.join(video_dir, f"{video_id}.mp4")

    # 画面在 (已追加条数, 滚动位置) 不变时完全一致 → 缓存复用, 渲染量从"帧数"降到"事件数"
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    records: List[dict] = []

    def frame_fn(f: int) -> np.ndarray:
        ts = f / float(fps)
        k = bisect_right(appear, ts + 1e-9)
        sub = msgs[:k]
        y_bottom = bottom_scroll_y(sub, renderer, style)
        y = scroll_y_at(ts, scroll_win, y_bottom)
        key = (k, y)
        if key not in cache:
            renderer.label_recorder = G.LabelRecorder(f, ts)
            cache[key] = renderer.render_chat_frame(sub, y, style)
            for r in renderer.label_recorder.records:
                cy = r["bbox"][1] + r["bbox"][3] / 2.0
                if style.top_offset <= cy <= (RESOLUTION[1] - style.nav_height):
                    records.append(r)
            renderer.label_recorder = None
        img = cache[key]
        # 系统弹窗窗口内叠加 OS 弹窗(不改缓存基准图, 返回副本)
        dlg = next((d for d in dialogs if d[0] <= ts < d[1]), None)
        if dlg is not None:
            img = draw_system_dialog(renderer, style, img, dlg[2])
        return img

    write_mp4(out_path, fps, n_frames, frame_fn, pcm)

    with open(os.path.join(label_dir, f"{video_id}.labels.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "video_id": video_id, "fps": fps, "resolution": list(RESOLUTION),
        "duration_sec": round(total_s, 2), "frame_count": n_frames,
        "title_height": style.top_offset,
        "nav_height": style.nav_height,
        # 视觉样式留痕: 壁纸图路径(空=纯平底色) + 气泡配色(RGB)
        # / Visual-style trace: wallpaper image ("" = flat) + bubble palette in RGB.
        "wallpaper_image": style.wallpaper_image,
        "bubble_palette": {
            "sent": list(style.sent_bubble[::-1]),
            "recv": list(style.recv_bubble[::-1]),
            "sent_text": list((style.sent_text_color or style.text_color)[::-1]),
        },
        "scroll_window": [round(v, 2) for v in scroll_win],
        "messages": [
            {"idx": it.idx, "sender": it.sender, "kind": it.kind,
             "screen": it.screen, "spoken": it.spoken, "captured": it.captured,
             "appear_sec": round(appear[i], 2),
             "audio_sec": round(audio_at.get(it.idx, -1.0), 2),
             "voice_duration": round(durs[it.idx], 2)}
            for i, it in enumerate(chat_items)
        ],
        "system_dialogs": [
            {"text": txt, "start_sec": round(a, 2), "end_sec": round(b, 2)}
            for a, b, txt in dialogs
        ],
    }
    with open(os.path.join(label_dir, f"{video_id}.meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    write_gt_md(os.path.join(out_dir, "gt", f"{video_id}.md"), video_id, items)
    return out_path


def write_gt_md(path: str, video_id: str, items: List[Item]) -> None:
    """落盘与 v2 协议同 7 列格式的 GT(含 ```tsv 机读块)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"# 合成语音聊天录屏 GT — {video_id}", "",
             "由 tests/gen_voice_chat_video.py 生成; 列同 v2 协议。", "",
             "```tsv", "id\tsender\tform\tscreen_text\tspoken_text\tcaptured\tnote"]
    for it in items:
        lines.append("\t".join([
            str(it.idx), it.sender, it.kind, it.screen or "-",
            it.spoken or "-", "yes" if it.captured else "no",
            it.note or {"text": "纯文本", "dialog": "系统弹窗(应被过滤)",
                        "sticker": "表情包(无文本)",
                        "system": "系统提示(居中灰字, 无气泡)",
                        "transfer": "转账/钱包(仅视觉)"}.get(it.kind, "常规"),
        ]))
    lines += ["```", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "synthetic", "voice_chat"))
    ap.add_argument("--gt", default=os.path.join(
        ROOT, "gt", "语音聊天30条_GT与录制协议_v2.md"))
    ap.add_argument("--ids", default="", help="只取这些序号, 如 1-8,12")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--name", default="")
    ap.add_argument("--no-scroll-demo", action="store_true",
                    help="末尾不插入滚到顶再滚回的演示")
    ap.add_argument("--scroll-after", type=int, default=28,
                    help="在第 N 条之后插入'滚到顶再滚回'演示(仅全量时生效)")
    ap.add_argument("--flat-wallpaper", action="store_true",
                    help="用原来的纯平底色墙纸(默认铺纹理壁纸图)")
    ap.add_argument("--wallpaper-image", default="",
                    help="自定义壁纸图路径(默认 tests/assets/wallpaper_example.png)")
    args = ap.parse_args()

    all_items = load_v2_items(args.gt)
    keep = parse_ids(args.ids, len(all_items))
    items = [it for it in all_items if it.idx in keep]
    for pos, it in enumerate(items, start=1):
        it.idx = pos        # 重新编号, 保证 GT 的 id 连续
    if not items:
        print("没有选中任何消息 / nothing selected")
        return

    full = (len(keep) == len(all_items))
    scroll_after = args.scroll_after if (full and not args.no_scroll_demo) else 0
    name = args.name or ("v2_synth" if full else f"v2_synth_x{len(items)}")
    out = render_one(name, items, args.out, args.fps, scroll_after,
                     use_image_wallpaper=not args.flat_wallpaper,
                     wallpaper_image=args.wallpaper_image)
    print(f"视频 / video : {out}")
    print(f"GT          : {os.path.join(args.out, 'gt', name + '.md')}")
    print(f"msgs={len(items)}  voice={sum(1 for i in items if i.kind == 'voice')}  "
          f"system={sum(1 for i in items if i.kind == 'system')}")


if __name__ == "__main__":
    main()
