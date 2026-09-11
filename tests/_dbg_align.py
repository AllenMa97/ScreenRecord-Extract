# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.row_fingerprint import RowFingerprintAligner

class NL:
    def log(self, *a, **k): pass

def cfg():
    return {"rf_strip_height":16,"rf_dhash_size":8,"rf_gap_penalty":-0.3,
            "rf_dup_ratio":0.8,"rf_new_ratio":0.4,"rf_noise_floor":14,
            "rf_history_len":3,"rf_rigid_roll_ratio":0.7,"rf_rigid_over_ratio":0.5,
            "compute_width":160}

def content(k):
    import random
    return (random.Random(f"content-{k}").getrandbits(32) << 32) | 0x00000000000015

def ctx(idx,k):
    return (content(k) & 0xFFFFFFFF00000000) | (idx & 0xFF)

a = RowFingerprintAligner(cfg(), NL())
prev2 = [ctx(0,21),ctx(0,22),ctx(0,23),ctx(0,24),ctx(0,25)]
cur2  = [ctx(1,22),ctx(1,23),ctx(1,24),ctx(1,25),ctx(0,26),ctx(0,27)]
r,d,n = a._align(prev2, cur2)
print("sc2 ratio=%.2f disp=%d new=%s main_ratio=%.2f over=%.2f rigid=%s"%(
    r,d,n,a._last_main_ratio,a._last_over_ratio,a._last_rigid_roll))

# show per-pair hd for the intra-frame matching
prev3 = [content(31),content(32),content(33),content(34)]
cur3  = [content(51),content(52),content(53),content(54)]
a3 = RowFingerprintAligner(cfg(), NL())
r3,d3,n3 = a3._align(prev3,cur3)
print("sc3 ratio=%.2f disp=%d new=%s rigid=%s over=%.2f main=%.2f"%(
    r3,d3,n3,a3._last_rigid_roll,a3._last_over_ratio,a3._last_main_ratio))