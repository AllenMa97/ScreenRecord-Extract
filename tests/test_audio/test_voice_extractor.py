"""
逐条语音抽取器 冒烟测试 (voice extractor smoke test).

用合成 PCM(正弦/静音)模拟音轨, 构造 VoiceEvent 切片,
验证: 切片时长正确 / wav 落盘存在 / manifest 写入 / 空音轨返回空列表。

不依赖真实视频, 无网络。直接 `python tests/test_audio/test_voice_extractor.py` 运行。
"""
import os
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.audio.data_types import VoiceEvent
from src.audio.voice_extractor import VoiceExtractor, write_manifest

SR = 16000


def _synthetic_samples(dur_s: float, sr: int = SR) -> np.ndarray:
    t = np.arange(0, dur_s * sr, dtype=np.float64) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


def _fake_decode(monkeypatch):
    """打桩 decode_audio/has_audio_track: 返回 4 秒合成音轨。"""
    import src.audio.audio_loader as loader
    import src.audio.voice_extractor as ve
    samples = _synthetic_samples(4.0)
    monkeypatch(loader, "has_audio_track", lambda path: True)

    def fake_decode(path, target_rate=16000):
        return type("D", (), {
            "samples": samples, "sample_rate": target_rate,
            "has_audio": True, "audio_duration": len(samples) / target_rate})()
    monkeypatch(loader, "decode_audio", fake_decode)
    monkeypatch(ve, "decode_audio", fake_decode)
    monkeypatch(ve, "has_audio_track", lambda path: True)


def _monkeypatch(obj, name, fn):
    setattr(obj, name, fn)


_ok = True


def check(name, cond, detail=""):
    global _ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        _ok = False


def main():
    monkeypatch = _monkeypatch
    _fake_decode(monkeypatch)

    events = [
        VoiceEvent(timestamp=0.5, end_time=1.5, side="发"),
        VoiceEvent(timestamp=2.0, end_time=3.0, side="收"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        ex = VoiceExtractor(base_url="", api_key="")  # transcribe 关闭(无 ASR)
        out = ex.extract("dummy.mp4", events, out_dir=tmp, transcribe=False,
                         side_meta={"发": "victim", "收": "suspect"})

        check("抽取 2 条语音", len(out) == 2, f"got {len(out)}")

        for r in out:
            check("wav 已落盘", os.path.exists(r.path), r.path)
            check("时长约 1s", abs(r.duration - 1.0) < 0.02, f"{r.duration}")
            check("文件名含序号", f"voice_{r.index:03d}" in os.path.basename(r.path))
        check("分类映射", out[0].category == "victim" and out[1].category == "suspect",
              f"{out[0].category}/{out[1].category}")

        man = os.path.join(tmp, "manifest.jsonl")
        write_manifest(out, man)
        check("manifest 写入", os.path.exists(man) and os.path.getsize(man) > 0)

    # 空音轨 / 无事件 -> 空
    check("空事件返回空", len(VoiceExtractor(base_url="").extract(
        "x.mp4", [], out_dir=tempfile.mkdtemp(), transcribe=False)) == 0)

    print("\nALL PASS" if _ok else "\nSOME FAILED")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()