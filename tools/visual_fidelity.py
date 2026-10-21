# -*- coding: utf-8 -*-
"""视觉拟真量化 / Visual-fidelity quantification.

对比「真实 IM 录屏 / 旧合成渲染 / 新拟真渲染」三者在配色与纹理上的客观差距。
Compare real IM screen recordings vs legacy synth rendering vs new realistic
rendering, on measurable colour/texture statistics.

量化口径 / Metrics (每帧 -> 跨帧取中位数):
  wallpaper_rgb    墙纸主色(中性像素的中位数 RGB) / wallpaper base colour
  band_ratio       墙纸像素落在亮度带 [230,244] 的占比(= 管线 chat_scene 判据)
  wp_texture       最平坦 40x40 块的逐通道 std(墙纸纹理强度)
  sent_bubble      「绿色」发送气泡像素均值 RGB(G 明显高于 R/B)
  recv_bubble      「近白」接收气泡像素占比与均值 RGB
  sat_ratio        高饱和(彩色)像素占比(整体色彩丰富度)

用法 / Usage:
  python tools/visual_fidelity.py --n 12 \
      --video real="D:/聊天记录视频/自行模拟对话记录2.mp4" \
      --video old_synth="synthetic/voice_chat/videos/v3_synth.mp4" \
      --video new_synth="synthetic/voice_chat/videos/v5_renovation.mp4"
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import av
except Exception as e:  # pragma: no cover
    print(f"[fatal] 需要 PyAV / PyAV required: {e}")
    sys.exit(1)


def sample_frames(path: str, n: int):
    """均匀抽取 n 帧 / uniformly sample n frames."""
    container = av.open(path)
    stream = container.streams.video[0]
    total = stream.frames or 0
    frames = []
    if total > 0:
        want = set(int(i * total / n) for i in range(n))
        idx = 0
        for frame in container.decode(stream):
            if idx in want:
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= n:
                    break
            idx += 1
    else:
        buf = []
        for frame in container.decode(stream):
            buf.append(frame.to_ndarray(format="rgb24"))
        if buf:
            step = max(1, len(buf) // n)
            frames = buf[::step][:n]
    container.close()
    return frames


def _flattest_patch_std(crop: np.ndarray, blk: int = 40, stride: int = 20):
    """在内容区找「墙纸色」的最平坦 blk×blk 块, 返回其逐通道 std(纹理强度).

    Among blk×blk patches whose median colour is neutral (channel spread<=8) and
    brightness in [225,250] — i.e. wallpaper, not avatars/stickers — pick the one
    with the smallest luma range; its per-channel std proxies wallpaper texture.
    """
    H, W, _ = crop.shape
    if H < blk or W < blk:
        return 0.0, crop.reshape(-1, 3)
    best = None
    best_px = None
    for y in range(0, H - blk + 1, stride):
        for x in range(0, W - blk + 1, stride):
            p = crop[y:y + blk, x:x + blk]
            med = np.median(p.reshape(-1, 3), axis=0)
            if (med.max() - med.min()) > 8 or not (225 <= med.mean() <= 250):
                continue                       # 非墙纸(头像/贴纸/气泡) / not wallpaper
            l = p.mean(axis=2)
            rng = float(l.max() - l.min())
            if best is None or rng < best:
                best = rng
                best_px = p
    if best_px is None:                        # 兜底: 全中性块 / fallback
        return 0.0, crop.reshape(-1, 3)
    std = float(np.mean(np.std(best_px.reshape(-1, 3), axis=0)))
    return std, best_px.reshape(-1, 3)


def frame_stats(rgb: np.ndarray):
    """单帧统计 / per-frame statistics."""
    H, W, _ = rgb.shape
    # 内容视口: 去掉顶部状态栏+标题栏(约 12% 高)与底部输入条(约 9%) / crop chrome
    top = int(H * 0.12)
    bot = int(H * 0.91)
    crop = rgb[top:bot, :, :].astype(np.float32)

    r, g, b = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    spread = crop.max(axis=2) - crop.min(axis=2)
    neutral = spread <= 8                      # 中性色(灰/白/黑) / neutral colours
    lum_all = crop.mean(axis=2)

    stats = {}
    # 亮度分箱(全内容区) / brightness bins over the content area
    stats["frac_dark"] = float((lum_all < 120).mean())          # 文字/深色
    stats["frac_mid"] = float(((lum_all >= 120) & (lum_all < 230)).mean())
    stats["frac_band"] = float(((lum_all >= 230) & (lum_all <= 244)).mean())
    stats["frac_bright"] = float((lum_all > 244).mean())        # 近白(接收气泡/亮底)

    if neutral.sum() > 0:
        npx = crop[neutral]
        wp = np.median(npx, axis=0)
        stats["wallpaper_rgb"] = wp
        lum = npx.mean(axis=1)
        stats["band_ratio"] = float(((lum >= 230) & (lum <= 244)).mean())
    else:
        stats["wallpaper_rgb"] = np.array([0.0, 0.0, 0.0])
        stats["band_ratio"] = 0.0

    # 墙纸纹理: 最平坦块 / wallpaper texture via flattest patch
    tstd, fpx = _flattest_patch_std(crop)
    stats["wp_texture"] = tstd
    stats["wp_rgb_flat"] = np.median(fpx, axis=0) if len(fpx) else stats["wallpaper_rgb"]
    stats["wp_noise_flat"] = tstd
    stats["wp_noise_std"] = tstd

    # 发送气泡: 绿色(G 明显高于 R 与 B) / outgoing green bubble
    green = (g > r + 20) & (g > b + 20) & (g > 120)
    if green.sum() > 0:
        stats["sent_bubble"] = crop[green].mean(axis=0)
        stats["sent_frac"] = float(green.mean())
    else:
        stats["sent_bubble"] = np.array([np.nan] * 3)
        stats["sent_frac"] = 0.0

    # 接收气泡: 近白 / incoming near-white bubble
    white = (r >= 250) & (g >= 250) & (b >= 250)
    stats["recv_frac"] = float(white.mean())

    # 彩色像素占比 / saturated (colourful) pixel ratio
    stats["sat_ratio"] = float((spread > 40).mean())
    return stats


def agg(video_path: str, n: int):
    frames = sample_frames(video_path, n)
    if not frames:
        return None
    per = [frame_stats(f) for f in frames]
    out = {"n_frames": len(per)}
    for key in ("wallpaper_rgb", "wp_rgb_flat", "sent_bubble"):
        arr = np.array([p[key] for p in per], dtype=np.float32)
        out[key] = np.nanmedian(arr, axis=0)
    for key in ("band_ratio", "wp_texture", "wp_noise_std", "wp_noise_flat", "sent_frac",
                "recv_frac", "sat_ratio", "frac_dark", "frac_mid", "frac_band",
                "frac_bright"):
        out[key] = float(np.median([p[key] for p in per]))
    out["resolution"] = (frames[0].shape[1], frames[0].shape[0])
    return out


def fmt_rgb(v):
    if v is None or not np.all(np.isfinite(v)):
        return "   -   "
    return f"({v[0]:3.0f},{v[1]:3.0f},{v[2]:3.0f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="每视频抽帧数 / frames per video")
    ap.add_argument("--video", action="append", default=[],
                    metavar="LABEL=PATH", help="可重复 / repeatable")
    args = ap.parse_args()

    if not args.video:
        ap.error("至少给一个 --video LABEL=PATH / need at least one --video")

    results = {}
    for spec in args.video:
        if "=" not in spec:
            ap.error(f"--video 需 LABEL=PATH 形式 / bad spec: {spec}")
        label, path = spec.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / path
        if not p.exists():
            print(f"[skip] {label}: 文件不存在 / not found: {p}")
            continue
        r = agg(str(p), args.n)
        if r is None:
            print(f"[skip] {label}: 无可用帧 / no frames")
            continue
        results[label] = r

    if not results:
        print("没有可对比的视频 / nothing to compare")
        return

    print("\n" + "=" * 96)
    print("视觉拟真量化 / visual fidelity (跨帧中位数 / median over frames)")
    print("=" * 96)
    print("[表1 配色与纹理 / colours & texture]")
    print(f"{'scene':<12} {'res':>11} {'wallpaper':>15} {'wp(flat)':>15} "
          f"{'band%':>7} {'texture':>8} {'sent_bubble':>15} {'sent%':>6} {'recv%':>6}")
    print("-" * 96)
    for label, r in results.items():
        res = f"{r['resolution'][0]}x{r['resolution'][1]}"
        print(f"{label:<12} {res:>11} {fmt_rgb(r['wallpaper_rgb']):>15} "
              f"{fmt_rgb(r['wp_rgb_flat']):>15} {r['band_ratio']*100:>6.1f}% "
              f"{r['wp_texture']:>8.2f} {fmt_rgb(r['sent_bubble']):>15} "
              f"{r['sent_frac']*100:>6.2f} {r['recv_frac']*100:>6.2f}")
    print("-" * 96)
    print("注: band% = 中性像素落亮度带[230,244]占比(管线 chat_scene 判据); "
          "texture = 最平坦 40x40 块的逐通道 std(墙纸纹理)")

    print("\n[表2 内容区亮度分布 / brightness distribution over content area]")
    print(f"{'scene':<12} {'<120(文字)':>10} {'120-230(中)':>11} "
          f"{'230-244(带)':>11} {'>244(近白)':>11} {'sat(彩色)':>10}")
    print("-" * 96)
    for label, r in results.items():
        print(f"{label:<12} {r['frac_dark']*100:>9.1f}% {r['frac_mid']*100:>10.1f}% "
              f"{r['frac_band']*100:>10.1f}% {r['frac_bright']*100:>10.1f}% "
              f"{r['sat_ratio']*100:>9.1f}%")
    print("-" * 96)

    # 相对真实样本的差距 / gap vs the real recording
    keys = list(results.keys())
    real_keys = [k for k in keys if "real" in k.lower()]
    if real_keys:
        ref = results[real_keys[0]]
        print(f"\n相对参考样本 [{real_keys[0]}] 的墙纸色差(欧氏) / wallpaper ΔRGB vs {real_keys[0]}:")
        for label, r in results.items():
            if label == real_keys[0]:
                continue
            d = float(np.linalg.norm(r["wp_rgb_flat"] - ref["wp_rgb_flat"]))
            print(f"  {label:<12} Δ墙纸RGB={d:6.2f}  "
                  f"Δband={r['band_ratio']-ref['band_ratio']:+.3f}  "
                  f"Δ纹理={r['wp_noise_flat']-ref['wp_noise_flat']:+.2f}  "
                  f"Δ气泡色={float(np.linalg.norm(r['sent_bubble']-ref['sent_bubble'])):5.1f}  "
                  f"Δ近白={r['frac_bright']-ref['frac_bright']:+.3f}")


if __name__ == "__main__":
    main()
