# -*- coding: utf-8 -*-
"""把裁剪气泡拼成联系表一张大图, 便于肉眼核查。"""
import sys, os, glob
import cv2, numpy as np

def _imread(p):
    data = np.fromfile(p, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

outdir = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "b"
imgs = sorted(glob.glob(os.path.join(outdir, prefix + "*.jpg")))
cols = 3
th = 160; tw = 260
rows = (len(imgs) + cols - 1) // cols
sheet = np.full((rows*(th+30), cols*(tw+10), 3), 245, np.uint8)
for i, p in enumerate(imgs):
    img = _imread(p)
    if img is None: continue
    ih, iw = img.shape[:2]
    scale = min(tw/iw, th/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    img = cv2.resize(img, (max(nw,1), max(nh,1)))
    r, c = divmod(i, cols)
    y = r*(th+30); x = c*(tw+10)
    sheet[y:y+nh, x:x+nw] = img[:nh, :nw]
    cv2.putText(sheet, os.path.basename(p), (x, y+nh+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,50,50), 1)
out = os.path.join(outdir, "_contact.jpg")
cv2.imencode(".jpg", sheet)[1].tofile(out)
print(out, len(imgs), "crops")