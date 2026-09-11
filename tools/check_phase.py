# -*- coding: utf-8 -*-
"""检查相位估计是否真正激活/稳定, 以及它对同内容对齐的影响.
Usage: python tools/check_phase.py <stem>"""
import sys
from pathlib import Path
import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import Config
from src.row_fingerprint import RowFingerprintAligner
from src.utils.behavior_logger import BehaviorLogger

class NL:
    def log(self, *a, **k): return None

def roi(img,cfg):
    h,w=img.shape[:2]
    t=int(cfg.get("roi","margin_top",60) or 60); b=int(cfg.get("roi","margin_bottom",120) or 120)
    return (0,min(t,h-1),w,max(t+1,h-b)-min(t,h-1))

def main():
    stem=sys.argv[1]
    cfg=Config(str(ROOT/"config.yaml"))
    vd=cfg.get("visual_dedup"); rf=dict(vd); rf["compute_width"]=vd.get("compute_width",160)
    al=RowFingerprintAligner(rf,NL())
    cont=av.open(str(ROOT/"videos"/f"{stem}.mp4"))
    # 取若干间隔帧, 打印相位与首帧序列的匹配情况
    phases=[]
    samples=[i for i in range(0,400,30)]
    first_seq=None
    idx=0
    for fr in cont.decode(video=0):
        if idx in samples:
            img=fr.to_ndarray(format="bgr24")
            x,y,w,h=roi(img,cfg); im=img[y:y+h,x:x+w]
            seq=al._extract_sequence(im)
            phases.append((idx, al._last_phase, len(seq), seq[:2] if seq else []))
            if first_seq is None:
                first_seq=seq
        idx+=1
        if idx>max(samples): break
    print("帧idx / 相位 / 行数(前2指纹):")
    for p in phases:
        print("  ",p)
    # 检查相位分布
    ph=[p[1] for p in phases]
    print("相位分布 min/max/mean:", min(ph),max(ph),sum(ph)/len(ph))

if __name__=="__main__":
    main()