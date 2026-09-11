"""
一键准备测试所需字体
- Windows 自带字体：直接从 C:\Windows\Fonts 复制
- App 专属开源替代字体：自动从 GitHub 下载
"""
import os, shutil, urllib.request

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
WIN_FONTS = r"C:\Windows\Fonts"
os.makedirs(FONT_DIR, exist_ok=True)

# (目标文件名, Windows自带源文件名 or None, GitHub备用URL)
FONT_REGISTRY = [
    # Windows 自带中文字体（直接复制）
    ("msyh.ttc",            "msyh.ttc",            None),  # 微软雅黑 - 通用中文
    ("simhei.ttf",          "simhei.ttf",          None),  # 黑体 - SMS/系统默认
    ("simkai.ttf",          "simkai.ttf",          None),  # 楷体 - 备用
    
    # App 专属字体开源替代（GitHub 下载）
    ("PingFangSC-Regular.ttf", None, 
     "https://github.com/adobe-fonts/source-han-sans/raw/release/Variable/TTF/Subset/SourceHanSansCN-Regular.ttf"),
    ("DroidSansFallback.ttf", None,
     "https://github.com/notofonts/noto-cjk/raw/main/Sans/Variable/TTF/NotoSansCJKsc-VF.ttf"),
    ("Roboto-Regular.ttf", None,
     "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf"),
    ("AlibabaPuHuiTi-Regular.ttf", None,
     "https://github.com/adobe-fonts/source-han-sans/raw/release/Variable/TTF/Subset/SourceHanSansCN-Regular.ttf"),
    ("Lato-Regular.ttf", None,
     "https://github.com/google/fonts/raw/main/ofl/lato/Lato%5Bwght%5D.ttf"),
    ("SanFrancisco-Regular.ttf", None,  # SF Pro 无开源版，用 Inter 替代
     "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Regular.ttf"),
]

print(f"[SETUP] 字体目录: {FONT_DIR}")
for target_name, win_src, github_url in FONT_REGISTRY:
    dest = os.path.join(FONT_DIR, target_name)
    if os.path.exists(dest):
        print(f"  [SKIP] {target_name} 已存在")
        continue
    
    # 优先从 Windows 字体目录复制
    if win_src:
        src = os.path.join(WIN_FONTS, win_src)
        if os.path.exists(src):
            shutil.copy2(src, dest)
            print(f"  [COPY] {target_name} <- Windows Fonts")
            continue
    
    # 从 GitHub 下载
    if github_url:
        try:
            print(f"  [DOWN] {target_name} ...", end=" ", flush=True)
            urllib.request.urlretrieve(github_url, dest)
            size_kb = os.path.getsize(dest) // 1024
            print(f"OK ({size_kb}KB)")
        except Exception as e:
            print(f"FAIL: {e}")
    else:
        print(f"  [WARN] {target_name} 无可用来源，将使用 fallback")

print(f"\n[DONE] 字体准备完成！共 {len(os.listdir(FONT_DIR))} 个文件")