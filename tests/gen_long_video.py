"""
==============================================================================
工业级全覆盖测试视频生成器 (V2.0)
- 支持正交实验组合 (Orthogonal Combinations)
- 支持断点续连 (Checkpoint Resume)
- 支持多进程并发 (Multiprocessing)
==============================================================================
"""

import os
import re
import json
import random
import glob
import time
import math
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np

# random.seed(42)


# ==============================================================================
# 真实素材加载器 / Real Asset Loader
# ==============================================================================
class IndexedAssetLoader:
    """从预生成的 assets_index.json 加载真实素材路径"""
    def __init__(self, index_path: str = r"D:\手机录屏内容提取\tests\assets_index.json"):
        self.pools: Dict[str, List[str]] = {}
        self.avatar_cache = {} # 新增：用于缓存 ID 对应的头像

        # 严格定义图片的扩展名（排除音频和视频）
        self.image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        self.voice_exts = {".mp3", ".wav", ".flac", ".aac", ".silk", ".amr"}
        self.video_exts = {".mp4", ".mov", ".avi", ".mkv"}

        index_loaded = False
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                    
                for cat, paths in index.items():
                    valid_paths = []
                    for p in paths:
                        if not os.path.isfile(p): continue
                        ext = os.path.splitext(p)[1].lower()
                        
                        # 严格过滤：按类别限制扩展名 / Strict filtering by category
                        if cat in ('photos', 'stickers', 'docs') and ext not in self.image_exts:
                            continue
                        elif cat == 'voices' and ext not in self.voice_exts:
                            continue
                        elif cat == 'videos' and ext not in self.video_exts:
                            continue
                            
                        valid_paths.append(p)
                        
                    if valid_paths:
                        self.pools[cat] = valid_paths
                index_loaded = True
            except Exception as e:
                print(f"  [ASSETS] 警告: 加载索引文件失败 - {e}，将尝试扫描本地目录...")

        # 【兜底机制】如果索引没加载成功，直接从同级目录扫描真实素材
        if not index_loaded:
            base_dir = os.path.dirname(index_path)
            print(f"  [ASSETS] 正在扫描本地目录: {base_dir}")
            self.pools = {"photos": [], "voices": [], "videos": [], "stickers": [], "docs": []}
            
            for root, _, files in os.walk(base_dir):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    full_path = os.path.join(root, fname)
                    if ext in self.image_exts:
                        self.pools['photos'].append(full_path)
                    elif ext in self.voice_exts:
                        self.pools['voices'].append(full_path)
                    elif ext in self.video_exts:
                        self.pools['videos'].append(full_path)
                        
        print(f"  [ASSETS] 素材加载完成: {sum(len(v) for v in self.pools.values())} 个文件")
        for k, v in self.pools.items(): 
            if v: print(f"    - {k}: {len(v)}")

    def get_random(self, category: str) -> Optional[str]:
        """随机获取一个指定类别的文件路径 / Get a random file path from category"""
        pool = self.pools.get(category, [])
        return random.choice(pool) if pool else None

    def get_deterministic(self, category: str, seed: int) -> Optional[str]:
        """确定性获取：相同 seed 返回相同文件 / Deterministic: same seed -> same file

        用于视频渲染，确保同一消息在每帧渲染时使用相同的素材，避免闪烁
        Used in video rendering to ensure same asset per message across frames, preventing flicker
        """
        pool = self.pools.get(category, [])
        if not pool:
            return None
        return pool[seed % len(pool)]

    def add_avatar_pool(self, avatar_dir: str):
        """将外部头像文件夹加入素材池"""
        if not os.path.exists(avatar_dir):
            print(f"  [ASSETS] 头像目录不存在: {avatar_dir}")
            return
        exts = {".jpg", ".jpeg", ".png"}
        avatars = [os.path.join(avatar_dir, f) for f in os.listdir(avatar_dir)
                   if os.path.splitext(f)[1].lower() in exts]
        self.pools['avatars'] = avatars
        print(f"  [ASSETS] avatars: {len(avatars)} 个可用头像")

    def get_avatar(self, sender_id: str) -> Optional[str]:
        """根据发送者ID获取固定头像"""
        if sender_id in self.avatar_cache:
            return self.avatar_cache[sender_id]
        
        # 如果没有缓存，随机取一个并缓存起来
        path = self.get_random('avatars')
        if path:
            self.avatar_cache[sender_id] = path
        return path

# ==============================================================================
# PART 1: 枚举与配置 / Enums & Configurations
# ==============================================================================

class AppType(Enum):
    WECHAT = "WeChat"
    QQ = "QQ"
    TELEGRAM = "Telegram"
    SMS = "SMS"
    DINGTALK = "DingTalk"
    FEISHU = "Feishu"
    ENTERPRISE_WECHAT = "EntWeChat"
    LINE = "Line"
    WHATSAPP = "WhatsApp"
    KAKAO = "KakaoTalk"
    WEB_BROWSER = "WebChat"
    CUSTOM_APP = "CustomApp"

class DisplayMode(Enum):
    NORMAL = "Normal"
    DARK_MODE = "DarkMode"
    LARGE_FONT = "LargeFont"
    SMALL_FONT = "SmallFont"
    HIGH_CONTRAST = "HighContrast"
    NIGHT_SHIFT = "NightShift"

class DeviceType(Enum):
    STANDARD_9_19 = "Standard_9_19"
    NOTCH = "Notch"
    PUNCH_HOLE = "PunchHole"
    FOLDABLE = "Foldable"
    CRACKED = "CrackedScreen"
    TABLET = "Tablet"

class BehaviorType(Enum):
    SLOW_SCROLL = "slow_scroll"
    FAST_SCROLL = "fast_scroll"
    VARIABLE_SCROLL = "variable_scroll"
    RECIPROCATING = "reciprocating"
    OVERSCROLL_BOUNCE = "overscroll_bounce"
    STATIC_READING = "static_reading"
    PINCH_ZOOM = "pinch_zoom"
    AVATAR_CLICK = "avatar_click"
    VIEW_IMAGE = "view_image"
    VIEW_VIDEO = "view_video"
    VOICE_PLAYBACK = "voice_playback"
    LONG_PRESS_MENU = "long_press_menu"
    KEYBOARD_TYPING = "keyboard_typing"
    EXIT_TO_HOME = "exit_to_home"
    INCOMING_CALL = "incoming_call"
    NOTIFICATION_POPUP = "notification_popup"
    APP_SWITCH = "app_switch"
    LOCK_UNLOCK = "lock_unlock"
    SEARCH_IN_CHAT = "search_in_chat"
    PULL_REFRESH = "pull_refresh"
    RED_PACKET = "red_packet"
    TRANSFER_SCREEN = "transfer_screen"
    MESSAGE_RECALL = "message_recall"
    GROUP_ANNOUNCE = "group_announce"
    SPLASH_SCREEN = "splash_screen"
    LOADING_SPINNER = "loading_spinner"
    PERMISSION_DIALOG = "permission_dialog"
    LOW_BATTERY_WARN = "low_battery_warn"
    FAKE_SCREENSHOT_IN_CHAT = "fake_screenshot"
    SPLIT_SCREEN = "split_screen"


# ==============================================================================
# PART 2: 样式定义 / Style Definitions
# ==============================================================================

@dataclass
class AppStyle:
    app_type: AppType
    display_mode: DisplayMode
    bg_color: Tuple[int, int, int]
    title_bar_color: Tuple[int, int, int]
    nav_bar_color: Tuple[int, int, int]
    sent_bubble: Tuple[int, int, int]
    recv_bubble: Tuple[int, int, int]
    text_color: Tuple[int, int, int]
    secondary_text: Tuple[int, int, int]
    font_scale: float          # 保留作为字号基准系数
    bubble_radius: int
    title_height: int
    nav_height: int
    has_avatar: bool = True
    # === 新增字体配置 ===
    font_path: str = ""        # TTF/OTF 字体文件路径
    font_bold_path: str = ""   # 粗体字体路径（可选）
    line_spacing: float = 1.2  # 行高系数
    # === 拟真 UI 参数 / Realistic-UI parameters ===
    # 默认值 == 旧渲染行为(不画状态栏/输入条, 标题白字, 气泡撑满, 无纹理),
    # 仅 WeChat 浅色样式通过 overrides 打开, 其余样式零影响。
    # Defaults reproduce the legacy look; only the light WeChat style overrides them.
    status_bar_height: int = 0                       # 顶部系统状态栏高度(px, 0=不画)
    status_text_color: Tuple[int, int, int] = (60, 60, 60)
    status_time: str = ""                            # 状态栏时间文本(空=不画)
    title_text_color: Tuple[int, int, int] = (255, 255, 255)   # 标题栏文字颜色
    chat_title: str = ""                             # 标题栏文本(空=用 App 名)
    input_bar_color: Optional[Tuple[int, int, int]] = None  # 底部输入条底色(None=nav_bar_color)
    input_field_color: Tuple[int, int, int] = (255, 255, 255)
    input_hint: str = ""                             # 输入区提示文字(空=不画)
    input_hint_color: Tuple[int, int, int] = (170, 170, 170)
    bubble_max_w_ratio: float = 0.7                  # 气泡最大宽度占屏比
    bubble_pad_x: int = 10                           # 气泡内水平内边距
    bubble_pad_y: int = 8                            # 气泡内垂直内边距
    bubble_tail: bool = False                        # 气泡小尾巴
    fit_bubble_to_text: bool = True                  # 气泡宽度贴合文字(短气泡不再撑满)
    wallpaper_noise_std: float = 0.0                 # 墙纸极淡纹理(高斯噪声 σ)
    wallpaper_image: str = ""                        # 墙纸图片路径(空=纯平底色, 非空=铺图)
    sent_text_color: Optional[Tuple[int, int, int]] = None  # 发送侧气泡文字色(None=text_color)
    avatar_size: int = 36                            # 头像边长
    voice_bar_height: int = 44                       # 语音气泡高度
    divider_text: str = ""                           # 时间分隔条文本(空=不画)
    divider_text_color: Tuple[int, int, int] = (150, 150, 150)

    @property
    def top_offset(self) -> int:
        """内容视口上沿 = 系统状态栏 + 标题栏 / Top of the content viewport."""
        return self.status_bar_height + self.title_height


def generate_all_styles() -> List[AppStyle]:
    base_styles = {
        AppType.WECHAT: dict(bg=(237,237,237), title=(60,60,60), nav=(50,50,50), sent=(148,200,100), recv=(255,255,255), text=(30,30,30), sec=(150,150,150), font=0.40, radius=6, title_h=50, nav_h=70),
        AppType.QQ: dict(bg=(240,245,255), title=(0,122,255), nav=(240,245,255), sent=(0,122,255), recv=(255,255,255), text=(30,30,30), sec=(130,130,130), font=0.42, radius=10, title_h=48, nav_h=65),
        AppType.TELEGRAM: dict(bg=(15,25,35), title=(20,35,50), nav=(15,25,35), sent=(40,120,180), recv=(40,50,65), text=(240,240,240), sec=(140,150,160), font=0.40, radius=4, title_h=52, nav_h=60),
        AppType.SMS: dict(bg=(255,255,255), title=(248,248,248), nav=(255,255,255), sent=(52,199,89), recv=(230,230,235), text=(0,0,0), sec=(140,140,145), font=0.44, radius=14, title_h=44, nav_h=50),
        AppType.DINGTALK: dict(bg=(245,246,248), title=(48,97,246), nav=(255,255,255), sent=(48,97,246), recv=(255,255,255), text=(40,40,40), sec=(150,150,150), font=0.40, radius=8, title_h=50, nav_h=68),
        AppType.FEISHU: dict(bg=(248,249,250), title=(255,255,255), nav=(248,249,250), sent=(60,130,246), recv=(255,255,255), text=(31,35,41), sec=(148,152,158), font=0.41, radius=8, title_h=48, nav_h=62),
        AppType.ENTERPRISE_WECHAT: dict(bg=(240,240,240), title=(30,120,200), nav=(45,45,45), sent=(80,180,80), recv=(255,255,255), text=(35,35,35), sec=(145,145,145), font=0.40, radius=6, title_h=50, nav_h=70),
        AppType.LINE: dict(bg=(140,200,255), title=(24,140,60), nav=(255,255,255), sent=(140,220,130), recv=(255,255,255), text=(30,30,30), sec=(140,140,140), font=0.42, radius=12, title_h=46, nav_h=64),
        AppType.WHATSAPP: dict(bg=(236,229,221), title=(0,117,95), nav=(240,240,240), sent=(213,245,193), recv=(255,255,255), text=(20,20,20), sec=(130,130,130), font=0.41, radius=8, title_h=52, nav_h=58),
        AppType.KAKAO: dict(bg=(178,214,233), title=(60,60,60), nav=(250,250,250), sent=(255,235,80), recv=(255,255,255), text=(30,30,30), sec=(140,140,140), font=0.42, radius=10, title_h=48, nav_h=66),
        AppType.WEB_BROWSER: dict(bg=(255,255,255), title=(240,240,240), nav=(250,250,250), sent=(220,235,255), recv=(245,245,245), text=(25,25,25), sec=(135,135,135), font=0.38, radius=4, title_h=56, nav_h=54),
        AppType.CUSTOM_APP: dict(bg=(40,40,50), title=(60,60,80), nav=(30,30,40), sent=(200,80,80), recv=(60,60,80), text=(230,230,230), sec=(150,150,160), font=0.39, radius=6, title_h=54, nav_h=72),
    }

    FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")

    # ✅ 每个 App 对应的字体文件（与 setup_fonts.py 中的目标文件名一致）
    font_map = {
        AppType.WECHAT:            "PingFangSC-Regular.ttf",      # 思源黑体替代
        AppType.QQ:                "DroidSansFallback.ttf",       # Noto Sans CJK 替代
        AppType.TELEGRAM:          "Roboto-Regular.ttf",
        AppType.SMS:               "simhei.ttf",                  # Windows 自带黑体
        AppType.DINGTALK:          "AlibabaPuHuiTi-Regular.ttf",  # 思源黑体替代
        AppType.FEISHU:            "Lato-Regular.ttf",
        AppType.ENTERPRISE_WECHAT: "PingFangSC-Regular.ttf",
        AppType.LINE:              "DroidSansFallback.ttf",
        AppType.WHATSAPP:          "Roboto-Regular.ttf",
        AppType.KAKAO:             "DroidSansFallback.ttf",
        AppType.WEB_BROWSER:       "msyh.ttc",                    # Windows 自带微软雅黑
        AppType.CUSTOM_APP:        "msyh.ttc",
    }

    def _make_style(app_type, display_mode, bg, title, nav, sent, recv, text, sec, 
                    font_scale, radius, title_h, nav_h, overrides=None):
        """统一构建 AppStyle，自动注入 font_path。

        overrides: 可选 dict, 覆盖任意 AppStyle 字段(用于 WeChat 拟真 UI)。
        / overrides: optional dict overriding any AppStyle field.
        """
        fname = font_map.get(app_type, "msyh.ttc")
        fpath = os.path.join(FONT_DIR, fname)
        kw = dict(
            app_type=app_type, display_mode=display_mode,
            bg_color=bg, title_bar_color=title, nav_bar_color=nav,
            sent_bubble=sent, recv_bubble=recv, text_color=text, secondary_text=sec,
            font_scale=font_scale, bubble_radius=radius,
            title_height=title_h, nav_height=nav_h,
            font_path=fpath,           # ✅ 关键：注入字体路径
            line_spacing=1.25,         # ✅ 新增行高系数
        )
        if overrides:
            kw.update(overrides)
        return AppStyle(**kw)

    # WeChat 浅色"拟真"UI: 参数取自真实微信录屏测量(432x960 参考帧, 按宽缩放 1.25)
    # / Light-WeChat realistic UI, parameters measured from the real recording.
    WECHAT_REAL_UI = dict(
        bg_color=(237, 237, 237),          # 墙纸 237 (落在管线聊天判据带 [230,244])
        title_bar_color=(237, 237, 237),   # 顶部栏与墙纸同色(真实微信浅色为 #EDEDED)
        nav_bar_color=(247, 247, 247),     # 底部输入条浅色 (#F7F7F7)
        # 非绿配色 / Non-green palette:
        #   发 = 深蓝 #4C6EF5(白字), 收 = 近白 #FAFAFA(深灰字, 与浅色壁纸可区分)。
        #   按 BGR 存, 渲染时 [::-1] 转 RGB; 文字色另配以保证对比度(蓝底用白字)。
        #   Sent = deep blue #4C6EF5 (white text); recv = near-white #FAFAFA
        #   (dark text). Stored BGR, flipped to RGB at draw time.
        sent_bubble=(245, 110, 76),        # BGR -> RGB #4C6EF5 (深蓝)
        recv_bubble=(250, 250, 250),       # BGR -> RGB #FAFAFA (近白)
        text_color=(40, 40, 40),
        sent_text_color=(255, 255, 255),   # 发送侧(蓝底)白字, 对比度 ~4.4:1
        secondary_text=(150, 150, 150),
        font_scale=0.60,                   # -> 正文字号 24px @540W (真实 ~20px @432W)
        bubble_radius=10,
        title_height=58,                   # 标题栏(真实 ~47px @432W)
        nav_height=88,                     # 输入条(真实 ~70px @432W)
        line_spacing=1.20,
        status_bar_height=58,              # 系统状态栏(真实 ~48px @432W)
        status_time="18:53",
        status_text_color=(60, 60, 60),
        title_text_color=(48, 48, 48),
        chat_title="林小雨",                # 标题栏显示聊天对象名(真实行为)
        input_bar_color=(247, 247, 247),
        input_field_color=(255, 255, 255),
        input_hint="按住说话",
        input_hint_color=(178, 178, 178),
        bubble_max_w_ratio=0.68,
        bubble_pad_x=14,
        bubble_pad_y=10,
        bubble_tail=True,
        fit_bubble_to_text=True,
        # 真实微信浅色墙纸实测为**纯平 #EDEDED**(最平坦 40x40 块 std≈0.0-0.3),
        # 加纹理反而制造了真实样本里没有的颗粒 → 置 0。
        # / The real light-mode wallpaper is a flat #EDEDED (flattest 40x40 patch
        # std ~0.0-0.3); adding a texture invents grain the real samples lack.
        # 注意: 本处保持"纯平底色"默认(旧行为零影响); 需要"一张图"壁纸的场景
        # 用 make_wechat_realistic_style() 打开 wallpaper_image。
        # / Flat base colour stays the default here; image-wallpaper scenarios opt in
        # through make_wechat_realistic_style().
        wallpaper_noise_std=0.0,
        wallpaper_image="",
        avatar_size=52,                    # 真实头像 ~44px @432W
        voice_bar_height=44,
        divider_text="",              # 顶部时间分隔条(真实微信有)
        divider_text_color=(150, 150, 150),
    )

    all_styles = []

    # 基础样式
    for app_type, p in base_styles.items():
        ov = WECHAT_REAL_UI if app_type is AppType.WECHAT else None
        all_styles.append(_make_style(
            app_type, DisplayMode.NORMAL,
            p['bg'], p['title'], p['nav'], p['sent'], p['recv'],
            p['text'], p['sec'], p['font'], p['radius'], p['title_h'], p['nav_h'],
            overrides=ov
        ))

    # 深色模式变体
    for app_type in [AppType.WECHAT, AppType.TELEGRAM, AppType.QQ, AppType.WHATSAPP, AppType.SMS, AppType.CUSTOM_APP]:
        p = base_styles[app_type]
        all_styles.append(_make_style(
            app_type, DisplayMode.DARK_MODE,
            (20,20,25), (30,30,35), (25,25,30), (60,130,70), (50,50,55),
            (230,230,230), (130,130,135), p['font'], p['radius'], p['title_h'], p['nav_h']
        ))

    # 大字体变体
    for app_type in [AppType.WECHAT, AppType.SMS, AppType.QQ, AppType.DINGTALK]:
        p = base_styles[app_type]
        all_styles.append(_make_style(
            app_type, DisplayMode.LARGE_FONT,
            p['bg'], p['title'], p['nav'], p['sent'], p['recv'],
            p['text'], p['sec'], p['font'] * 1.8, p['radius'],
            p['title_h'] + 10, p['nav_h'] + 10
        ))

    # 小字体变体
    for app_type in [AppType.WECHAT, AppType.TELEGRAM, AppType.FEISHU]:
        p = base_styles[app_type]
        all_styles.append(_make_style(
            app_type, DisplayMode.SMALL_FONT,
            p['bg'], p['title'], p['nav'], p['sent'], p['recv'],
            p['text'], p['sec'], p['font'] * 0.7, max(2, p['radius']-2),
            p['title_h']-6, p['nav_h']-8
        ))

    # 高对比度
    for app_type in [AppType.WECHAT, AppType.SMS]:
        all_styles.append(_make_style(
            app_type, DisplayMode.HIGH_CONTRAST,
            (0,0,0), (255,255,255), (0,0,0), (255,255,0), (0,0,200),
            (255,255,255), (200,200,200), 0.45, 0, 50, 70
        ))

    # 夜间模式(黄屏)
    for app_type in [AppType.WECHAT, AppType.QQ]:
        p = base_styles[app_type]
        all_styles.append(_make_style(
            app_type, DisplayMode.NIGHT_SHIFT,
            (min(255,p['bg'][0]+30), min(255,p['bg'][1]+10), max(0,p['bg'][2]-40)),
            p['title'], p['nav'],
            (min(255,p['sent'][0]+20), min(255,p['sent'][1]+5), max(0,p['sent'][2]-30)),
            (min(255,p['recv'][0]+15), min(255,p['recv'][1]+5), max(0,p['recv'][2]-25)),
            p['text'], p['sec'], p['font'], p['radius'], p['title_h'], p['nav_h']
        ))

    print(f"  样式总数: {len(all_styles)}")
    return all_styles


# ==============================================================================
# PART 2b: 壁纸图片 / Wallpaper image
# ==============================================================================
# 真实录屏里聊天区背景是**一张墙纸图**(不是纯平色块)。本模块用程序化方式生成一张
# 浅色纹理壁纸(渐变 + 噪点 + 几何图案)并落盘; 同时保留原来的"纯平底色"模式, 通过
# make_wechat_realistic_style(use_image_wallpaper=False) 或 CLI 开关回退。
# / Real recordings show an image wallpaper, not a flat colour. We render one
# procedurally and keep the flat-colour mode as a switchable fallback.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(_SCRIPT_DIR, "assets")
DEFAULT_WALLPAPER_IMAGE = os.path.join(ASSET_DIR, "wallpaper_example.png")

WALLPAPER_MIN_VALUE = 230     # 与 src/visual/chat_scene 的墙纸亮度带 [230,244] 对齐
WALLPAPER_MAX_VALUE = 243     # 上限留 1 级余量, 不贴 244 边界
SYSTEM_FONT_RATIO = 0.78      # 系统提示字号 = 正文字号 * 该系数(小号)


def generate_wallpaper_image(path: str, width: int = 540, height: int = 1200,
                             seed: int = 20240516) -> str:
    """程序化生成一张"浅色纹理壁纸图"并落盘, 返回路径。

    设计约束 / Design constraints:
      - 每个像素 R=G=B(中性) 且亮度落在 [230,243]: 保证 src/visual/chat_scene.py 的
        "墙纸占比"判据(中性色 + 亮度在 [230,244])仍把整帧判为聊天界面, 不会被当
        非聊天帧丢掉;
      - 叠加 垂直/对角渐变 + 大圆/斜条纹/圆角矩形几何图案 + 细噪点 → 肉眼可见的纹理
        (不再是纯平色块);
      - 固定随机种子 → 结果可复现。
    / Neutral (R=G=B) light wallpaper in [230,243] so the pipeline's chat-scene test
    still passes, while carrying a visible gradient + geometry + grain texture.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    # 1) 渐变: 垂直 + 对角 / gradient
    canvas = (234.0 + 4.0 * (yy / max(height - 1, 1))
              + 2.0 * (xx / max(width - 1, 1)))

    # 2) 几何图案: 柔和大圆 / soft large circles
    for _ in range(6):
        cx, cy = rng.randint(0, width), rng.randint(0, height)
        r = rng.uniform(width * 0.18, width * 0.5)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        canvas += (rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 5.0)
                   * np.clip(1.0 - d / r, 0.0, 1.0))

    # 3) 几何图案: 斜条纹 / diagonal stripes
    canvas += 1.6 * np.sin((xx + yy) / 26.0)

    # 4) 几何图案: 圆角矩形 / rounded rectangles
    for _ in range(4):
        x0, y0 = rng.randint(0, width), rng.randint(0, height)
        rw = rng.randint(width // 6, max(width // 6 + 1, width // 2))
        rh = rng.randint(height // 10, max(height // 10 + 1, height // 3))
        inside = ((xx >= x0) & (xx < x0 + rw) & (yy >= y0) & (yy < y0 + rh))
        canvas += rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 4.0) * inside.astype(np.float32)

    # 5) 细噪点 / fine grain
    canvas += rng.normal(0.0, 1.8, size=(height, width)).astype(np.float32)

    gray = np.clip(canvas, WALLPAPER_MIN_VALUE, WALLPAPER_MAX_VALUE).astype(np.uint8)
    img = np.stack([gray, gray, gray], axis=-1)
    Image.fromarray(img, "RGB").save(path)
    return path


def ensure_wallpaper_image(path: str = "", width: int = 540, height: int = 1200) -> str:
    """壁纸图不存在时才生成 / generate the wallpaper image only when missing."""
    path = path or DEFAULT_WALLPAPER_IMAGE
    if not os.path.isfile(path):
        generate_wallpaper_image(path, width, height)
    return path


def make_wechat_realistic_style(use_image_wallpaper: bool = True,
                                wallpaper_image: str = "") -> AppStyle:
    """构造"拟真微信(浅色)"样式 / Build the realistic light-WeChat style.

    use_image_wallpaper=True(默认) → 聊天区铺一张纹理壁纸图;
    False → 回退到原来的纯平底色模式(便于对照/回退)。
    / True (default) uses the textured wallpaper image; False keeps the flat colour.
    """
    style = next(s for s in generate_all_styles()
                 if s.app_type is AppType.WECHAT
                 and s.display_mode is DisplayMode.NORMAL)
    if use_image_wallpaper:
        style.wallpaper_image = ensure_wallpaper_image(wallpaper_image)
    return style


# ==============================================================================
# PART 3: 数据加载 / Data Loading
# ==============================================================================

@dataclass
class ChatMessage:
    sender_id: str
    text: str
    is_received: bool
    timestamp: float = 0.0
    msg_type: str = "text"          # text/image/voice/video/transfer/screenshot/sticker/system
    
    # === 新增多媒体元信息 ===
    media_width: int = 0            # 图片/视频/截图的显示宽度
    media_height: int = 0           # 图片/视频/截图的显示高度
    voice_duration: float = 0.0     # 语音时长（秒），0 表示未标注
    voice_transcribed: str = ""     # 语音转文字结果，空字符串表示未转写
    has_waveform: bool = False      # 是否显示语音波形条
    video_cover_color: Tuple[int,int,int] = (0,0,0)  # 视频封面主色
    sticker_name: str = ""          # 表情包标识名
    transfer_amount: float = 0.0    # 转账金额
    transfer_status: str = ""       # 已收款/待确认/已退还
    screenshot_app: str = ""        # 截图来源App标识
    is_video_playing: bool = False  # 当前帧是否处于视频播放状态

class MediaAssetGenerator:
    """生成各类多媒体消息的模拟视觉素材"""
    
    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
    
    def gen_image_placeholder(self, w: int, h: int, seed: int = 0) -> np.ndarray:
        """生成模拟照片/图片（带渐变+噪点，不像纯色块）"""
        key = f"img_{w}_{h}_{seed}"
        if key in self._cache:
            return self._cache[key]
        
        rng = np.random.RandomState(seed)
        # 随机渐变色背景
        c1 = rng.randint(80, 220, size=3)
        c2 = rng.randint(80, 220, size=3)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            t = y / max(h - 1, 1)
            img[y] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        # 加噪点模拟真实照片质感
        noise = rng.randint(-15, 15, (h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # 中心图标提示
        cv2.putText(img, "IMG", (w//2-20, h//2+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255,180), 1)
        
        self._cache[key] = img
        return img
    
    def gen_voice_bubble(self, duration: float, has_waveform: bool,
                         transcribed: str, font: ImageFont.FreeTypeFont,
                         bubble_w: int, bubble_h: int, is_recv: bool,
                         ink: Tuple[int, int, int] = (70, 70, 70)) -> np.ndarray:
        """语音消息气泡内容(RGBA): 喇叭图标 + 时长数字。

        真实微信语音气泡 = [喇叭图标] + 时长秒数(如 5\\"), **没有**波形条; 气泡宽度
        随时长增长。旧实现画了一个灰色方块"图标"+随机波形条, 且因通道颠倒导致内容
        半透明, 一眼可辨为合成。本实现改为图标+时长, 返回 RGBA(alpha=0 表示透明),
        由调用方直接按 alpha 合成。
        ink = 图标/文字颜色(RGB), 需与气泡底色对比(深色气泡传白字)。
        / Real WeChat voice bubble = speaker icon + duration badge (no waveform).
        Returns RGBA; caller composites with the alpha channel. `ink` is the icon/text
        colour (RGB) and must contrast the bubble fill (white on dark bubbles).
        """
        h, w = bubble_h, bubble_w
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        ink = tuple(int(v) for v in ink) + (255,)
        cy = h // 2

        if is_recv:      # 图标贴左缘, 声波朝右
            d.polygon([(14, cy - 6), (22, cy - 6), (30, cy - 13), (30, cy + 13),
                       (22, cy + 6), (14, cy + 6)], fill=ink)
            for r in (9, 14, 19):
                d.arc([30 - r, cy - r, 30 + r, cy + r], -55, 55, fill=ink, width=2)
            tx = 56
        else:            # 图标贴右缘, 声波朝左
            d.polygon([(w - 14, cy - 6), (w - 22, cy - 6), (w - 30, cy - 13),
                       (w - 30, cy + 13), (w - 22, cy + 6), (w - 14, cy + 6)], fill=ink)
            for r in (9, 14, 19):
                d.arc([w - 30 - r, cy - r, w - 30 + r, cy + r], 125, 235, fill=ink, width=2)
            tx = None    # 时长右对齐

        if duration > 0:
            dur_text = f"{duration:.0f}\""
            bb = font.getbbox(dur_text)
            tw = bb[2] - bb[0]
            ty = cy - (bb[3] - bb[1]) // 2 - bb[1]
            if tx is None:
                tx = w - 56 - tw
            d.text((tx, ty), dur_text, font=font, fill=ink)

        return np.array(img)          # RGBA

    def gen_video_cover(self, w: int, h: int, cover_color: Tuple[int,int,int]) -> np.ndarray:
        """生成视频封面 + 播放按钮"""
        img = np.full((h, w, 3), cover_color, dtype=np.uint8)
        # 半透明黑色遮罩
        overlay = (img * 0.6).astype(np.uint8)
        # 播放三角形
        cx, cy = w // 2, h // 2
        pts = np.array([[cx-12, cy-16], [cx-12, cy+16], [cx+16, cy]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (255, 255, 255))
        cv2.circle(overlay, (cx, cy), 24, (255, 255, 255), 2)
        # 混合
        mask = np.zeros_like(img, dtype=np.float32)
        cv2.circle(mask, (cx, cy), 28, (1, 1, 1), -1)
        result = (img * (1 - mask) + overlay * mask).astype(np.uint8)
        # 底部进度条
        cv2.rectangle(result, (0, h-4), (w, h), (255, 255, 255), -1)
        cv2.rectangle(result, (0, h-4), (w//3, h), (255, 80, 80), -1)
        return result
    
    def gen_transfer_card(self, amount: float, status: str,
                          is_recv: bool, font: ImageFont.FreeTypeFont) -> np.ndarray:
        """转账/红包卡片(RGBA, 圆角)。

        真实微信转账卡片是**橙色**(#FA9D3B)圆角卡片 + 白色金额/状态; 旧实现用绿色
        直角矩形, 与真实不符。返回 RGBA, 由调用方按 alpha 合成(圆角处透明)。
        / Real WeChat transfer card is orange with rounded corners; returns RGBA.
        """
        w, h = 240, 96
        bg = (250, 81, 81) if "红包" in status else (250, 157, 59)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill=255)
        img.paste(Image.new("RGBA", (w, h), bg + (255,)), (0, 0), mask)
        draw = ImageDraw.Draw(img)

        small_font = ImageFont.truetype(font.path, 14) if getattr(font, "path", None) else font

        # 左侧圆形"¥"图标
        draw.ellipse((14, 32, 52, 70), fill=(255, 255, 255, 210))
        ic = small_font.getbbox("¥")
        draw.text((33 - (ic[2] - ic[0]) // 2, 51 - (ic[3] - ic[1]) // 2 - ic[1]),
                  "¥", font=small_font, fill=bg + (255,))

        # 金额 + 状态 + 底部"微信转账"
        draw.text((66, 16), f"¥{amount:.2f}", font=font, fill=(255, 255, 255, 255))
        draw.text((66, 46), status or ("等待对方收款" if not is_recv else "已收款"),
                  font=small_font, fill=(255, 255, 240, 255))
        draw.line([(66, 70), (w - 16, 70)], fill=(255, 255, 255, 130), width=1)
        draw.text((66, 74), "微信转账", font=small_font, fill=(255, 255, 240, 235))
        return np.array(img)          # RGBA
    
    def gen_screenshot_embed(self, app_name: str, w: int, h: int) -> np.ndarray:
        """生成嵌套App截图（模拟聊天记录中的转账截图/支付截图）"""
        img = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)
        # 顶部状态栏
        cv2.rectangle(img, (0, 0), (w, 24), (60, 60, 60), -1)
        cv2.putText(img, "12:30", (w//2-15, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        # App标题
        cv2.rectangle(img, (0, 24), (w, 56), (80, 80, 80), -1)
        label = f"{app_name} - 交易详情"
        cv2.putText(img, label, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        # 金额区域
        cv2.putText(img, "-5,000.00", (w//2-50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 2)
        cv2.putText(img, "转账成功", (w//2-30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,180,80), 1)
        # 边框
        cv2.rectangle(img, (0, 0), (w-1, h-1), (200, 200, 200), 1)
        return img
    
    def gen_sticker(self, name: str, size: int = 120) -> np.ndarray:
        """生成表情包占位（带表情轮廓）"""
        img = np.full((size, size, 3), (255, 255, 255), dtype=np.uint8)
        rng = np.random.RandomState(hash(name) % (2**31))
        # 随机柔和底色
        bg = tuple(rng.randint(200, 255, size=3).tolist())
        img[:] = bg
        # 简单笑脸轮廓
        cx, cy = size // 2, size // 2
        r = size // 3
        cv2.circle(img, (cx, cy), r, (60, 60, 60), 2)
        cv2.circle(img, (cx - r//3, cy - r//5), r//8, (60, 60, 60), -1)
        cv2.circle(img, (cx + r//3, cy - r//5), r//8, (60, 60, 60), -1)
        cv2.ellipse(img, (cx, cy + r//5), (r//3, r//5), 0, 0, 180, (60, 60, 60), 2)
        return img

def _inject_multimedia(messages: List[ChatMessage]) -> List[ChatMessage]:
    """对消息列表注入多媒体类型转换 / Inject multimedia type conversion into message list"""
    density_factor = random.uniform(0.0, 0.25)

    base_probs = {
        "image": 0.15, "voice": 0.12, "sticker": 0.08, "video": 0.04,
        "transfer": 0.06, "screenshot": 0.05, "system": 0.03,
    }
    actual_probs = {k: v * density_factor for k, v in base_probs.items()}

    system_keywords = ["撤回", "移出群聊", "已读", "加入群聊", "开启了朋友验证", "拍了拍"]
    money_keywords = ["转账", "红包", "付款", "收款", "到账", "充值", "提现", "¥", "元"]
    image_keywords = ["照片", "图片", "截图", "拍", "看看这个", "你看", "发了个"]
    voice_keywords = ["语音", "说不清楚", "打字太慢", "听着", "我说一下"]

    for i in range(len(messages)):
        msg = messages[i]
        forced_type = None

        if any(kw in msg.text for kw in system_keywords):
            forced_type = "system"
        elif any(kw in msg.text for kw in money_keywords):
            forced_type = random.choice(["transfer", "screenshot"])
        elif any(kw in msg.text for kw in image_keywords):
            forced_type = "image"
        elif any(kw in msg.text for kw in voice_keywords):
            forced_type = "voice"

        if forced_type is None:
            roll = random.random()
            cumulative = 0.0
            for mtype, prob in actual_probs.items():
                cumulative += prob
                if roll < cumulative:
                    forced_type = mtype
                    break

        if forced_type and forced_type != "text":
            original_text = msg.text
            msg.msg_type = forced_type

            if forced_type == "image":
                msg.media_width = random.randint(120, 280)
                msg.media_height = random.randint(int(msg.media_width * 0.5), int(msg.media_width * 1.5))
                msg.text = "[图片]"
            elif forced_type == "voice":
                msg.voice_duration = round(random.uniform(1.0, 59.0), 1)
                msg.has_waveform = random.random() < 0.75
                msg.voice_transcribed = original_text if random.random() < 0.4 else ""
                msg.text = "[语音消息]"
            elif forced_type == "sticker":
                sticker_names = ["捂脸", "笑哭", "点赞", "OK", "再见", "害羞", "愤怒", "疑问", "爱心", "鼓掌"]
                msg.sticker_name = random.choice(sticker_names)
                msg.media_width = random.choice([80, 100, 120, 150])
                msg.text = f"[表情:{msg.sticker_name}]"
            elif forced_type == "video":
                msg.media_width = random.randint(160, 280)
                msg.media_height = int(msg.media_width * random.uniform(0.5, 0.75))
                msg.video_cover_color = tuple(np.random.randint(30, 180, size=3).tolist())
                msg.text = "[视频]"
            elif forced_type == "transfer":
                msg.transfer_amount = round(random.uniform(0.01, 50000.0), 2)
                msg.transfer_status = random.choice(["已收款", "等待对方收款", "已退还", ""])
                msg.text = f"[转账 ¥{msg.transfer_amount:.2f}]"
            elif forced_type == "screenshot":
                apps = ["支付宝", "微信支付", "银行APP", "淘宝", "京东", "抖音"]
                msg.screenshot_app = random.choice(apps)
                msg.media_width = random.randint(180, 260)
                msg.media_height = int(msg.media_width * random.uniform(1.2, 1.8))
                msg.text = f"[{msg.screenshot_app}截图]"
            elif forced_type == "system":
                sys_templates = ["对方撤回了一条消息", "你已被移出群聊", "消息已发出，但被对方拒收了", "该用户不存在"]
                if not any(kw in msg.text for kw in system_keywords):
                    msg.text = random.choice(sys_templates)

    return messages


def load_cases(data_dir: str, min_per_case: int = 50) -> Dict[str, List[ChatMessage]]:
    """加载真实案件数据，按案件分组，保持对话结构 / Load real case data, grouped by case, preserving conversation structure

    核心改进 / Key improvements:
    - 每个案件独立解析，保持对话上下文 / Each case parsed independently, preserving conversation context
    - 每个案件使用唯一 sender_id，不同案件有不同的"说话人" / Unique sender_id per case
    - 不再合并所有案件到扁平列表 / No longer merge all cases into a flat list
    """
    cases: Dict[str, List[ChatMessage]] = {}
    pattern = os.path.join(data_dir, "*", "AIResources", "casecontext.md")
    files = glob.glob(pattern)
    print(f"  找到 {len(files)} 个案件文件")

    for case_idx, f in enumerate(files):
        case_id = f"case_{case_idx:04d}"
        messages: List[ChatMessage] = []

        try:
            with open(f, "r", encoding="utf-8") as fh:
                content = fh.read(300000)

            # 提取嫌疑人/受害者昵称 / Extract suspect/victim nicknames
            suspect_name = ""
            victim_name = ""
            nick_match = re.search(r'嫌疑人账号信息.*?昵称[：:]\s*(.+?)[\n\r]', content, re.DOTALL)
            if nick_match:
                suspect_name = nick_match.group(1).strip()[:20]
            vict_match = re.search(r'事主账号信息.*?昵称[：:]\s*(.+?)[\n\r]', content, re.DOTALL)
            if vict_match:
                victim_name = vict_match.group(1).strip()[:20]

            # 每个案件使用唯一的 sender_id / Unique sender_id per case
            suspect_sender = f"{case_id}_suspect"
            victim_sender = f"{case_id}_victim"

            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                text = ""
                is_recv = None

                if '[收]' in line:
                    is_recv = True
                    parts = line.split('[收]', 1)
                    text = parts[-1].lstrip(':： ').strip() if len(parts) > 1 else ""
                elif '[发]' in line:
                    is_recv = False
                    parts = line.split('[发]', 1)
                    text = parts[-1].lstrip(':： ').strip() if len(parts) > 1 else ""

                # 清理尾部时间戳 / Clean trailing timestamps (e.g. "你好18:39" -> "你好")
                text = re.sub(r'\d{1,2}[:：]\d{2}.*$', '', text).strip()

                if is_recv is not None and text and len(text) > 1:
                    sender_id = suspect_sender if is_recv else victim_sender
                    messages.append(ChatMessage(
                        sender_id=sender_id,
                        text=text[:150],
                        is_received=is_recv
                    ))

        except Exception as e:
            print(f"  [WARN] 读取文件失败 {f}: {e}")
            continue

        if not messages:
            continue

        # 按案件单独注入多媒体 / Inject multimedia per case
        messages = _inject_multimedia(messages)

        # 消息太少时适度循环扩充 / Circularly augment if too few messages
        if len(messages) < min_per_case:
            original = messages[:]
            while len(messages) < min_per_case:
                messages.extend(original[:min_per_case - len(messages)])
            messages = messages[:min_per_case]

        cases[case_id] = messages
        print(f"    [{case_id}] {len(messages)} 条消息 (嫌疑人: {suspect_name or '未知'}, 受害者: {victim_name or '未知'})")

    # 兜底：无真实数据时生成模拟案件 / Fallback: generate synthetic cases if no real data
    if not cases:
        print("  [WARN] 未找到有效数据, 使用模拟消息")
        sample_dialogs = [
            [("你好，我是公安局的", True), ("怎么了", False), ("你的银行卡涉嫌洗钱", True), ("我没有啊", False)],
            [("请配合调查，不要告诉任何人", True), ("好的", False), ("把验证码发给我", True), ("验证码是123456", False)],
            [("你的快递到了", True), ("我没买快递啊", False), ("点击链接确认地址", True), ("好的我看看", False)],
            [("我发个语音跟你说", True), ("好的", False), ("钱已经打过去了", True), ("收到谢谢", False)],
        ]
        for ci, dialog in enumerate(sample_dialogs):
            case_id = f"case_{ci:04d}"
            msgs = []
            for text, is_recv in dialog:
                msgs.append(ChatMessage(
                    sender_id=f"{case_id}_suspect" if is_recv else f"{case_id}_victim",
                    text=text, is_received=is_recv
                ))
            # 扩充到 min_per_case
            original = msgs[:]
            while len(msgs) < min_per_case:
                msgs.extend(original[:min_per_case - len(msgs)])
            msgs = msgs[:min_per_case]
            cases[case_id] = msgs

    total = sum(len(v) for v in cases.values())
    print(f"  共加载 {len(cases)} 个案件, {total} 条消息")

    return cases

# ==============================================================================
# PART 4: 渲染引擎 / Rendering Engine
# ==============================================================================

class LabelRecorder:
    """单帧标签采集器 / Per-frame label collector

    记录该帧所有被绘制的文本及其 bbox(全帧坐标), 作为合成数据的"精确真值"。
    与 OCR 输入坐标的换算: pipeline 送 OCR 的是 ROI 裁剪帧, 其换算见 meta.json
    中的 title_height / nav_height(ROI = (0, title_height, W, H-title_height-nav_height))。
    """

    def __init__(self, frame_id: int, timestamp: float):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.records: List[Dict] = []
        # 当前绘制的行类别: "message"(普通消息) / "system"(系统提示)。
        # 渲染器在画系统提示行前临时置为 "system", 画完复位。
        # / Kind of the line being drawn: "message" or "system".
        self.current_kind = "message"

    def add(self, text: str, bbox: Tuple[int, int, int, int]) -> None:
        if not (text or "").strip():
            return
        self.records.append({
            "text": text,
            "kind": self.current_kind,
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "frame_id": self.frame_id,
            "timestamp": round(float(self.timestamp), 4),
        })


class RecordingDraw:
    """包裹 PIL ImageDraw, 记录所有 draw.text 调用(内容 + bbox)。

    只拦截 text() 以便采集标签, 其余绘制(矩形/圆角矩形/粘贴)通过 __getattr__
    原样透传给底层 draw, 不影响渲染结果。
    """

    def __init__(self, draw, recorder: LabelRecorder):
        self._draw = draw
        self._rec = recorder

    def __getattr__(self, name):
        return getattr(self._draw, name)

    def text(self, xy, text, fill=None, font=None, anchor=None, *args, **kwargs):
        self._draw.text(xy, text, fill=fill, font=font, anchor=anchor, *args, **kwargs)
        try:
            x0, y0, x1, y1 = self._draw.textbbox(xy, text, font=font, anchor=anchor)
            bbox = (x0, y0, max(0, x1 - x0), max(0, y1 - y0))
        except Exception:
            x0, y0 = int(xy[0]), int(xy[1])
            bbox = (x0, y0, int(len(text or "") * 16), 16)
        self._rec.add(text, bbox)


class FrameRenderer:
    def __init__(self, width: int = 540, height: int = 1200, asset_loader: IndexedAssetLoader = None):
        self.W = width
        self.H = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.asset_loader = asset_loader # 保存加载器实例
         # --- 新增这一行 ---
        self.media_gen = MediaAssetGenerator() 
        # -----------------
        self._font_cache = {} # 建议显式初始化字体缓存字典
        self._wallpaper_cache = {}  # 墙纸纹理缓存(固定纹理, 静止帧逐字节一致)
        # 标签采集器: 非 None 时记录本帧所有文本绘制(供合成数据打标)
        self.label_recorder: Optional[LabelRecorder] = None

    def _make_draw(self, pil_img):
        """按需返回带标签采集的 draw 代理 / Return a label-recording draw proxy when enabled"""
        base = ImageDraw.Draw(pil_img)
        if self.label_recorder is not None:
            return RecordingDraw(base, self.label_recorder)
        return base


    def _get_font(self, style: AppStyle, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """带缓存的字体加载器"""
        path = style.font_bold_path if (bold and style.font_bold_path) else style.font_path
        cache_key = f"{path}_{size}"
        if cache_key not in self._font_cache:
            try:
                self._font_cache[cache_key] = ImageFont.truetype(path, size)
            except (IOError, OSError):
                # Fallback: 尝试常见系统中文字体
                fallbacks = [
                    "msyh.ttc", "simhei.ttf", "pingfang.ttc", 
                    "NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
                ]
                font_loaded = False
                for fb in fallbacks:
                    try:
                        self._font_cache[cache_key] = ImageFont.truetype(fb, size)
                        font_loaded = True
                        break
                    except (IOError, OSError):
                        continue
                if not font_loaded:
                    self._font_cache[cache_key] = ImageFont.load_default()
        return self._font_cache[cache_key]

    def _wrap_pil(self, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
        """基于 Pillow 的精确中文换行"""
        lines, cur = [], ""
        for ch in text:
            t = cur + ch
            bbox = font.getbbox(t)
            if bbox[2] - bbox[0] > max_w and cur:
                lines.append(cur)
                cur = ch
            else:
                cur = t
        if cur:
            lines.append(cur)
        return lines if lines else [""]

    def _pil_draw_rounded_rect(self, draw: ImageDraw.Draw, xy, radius, fill):
        """Pillow 圆角矩形(接受 (x1,y1,x2,y2) 或 [(x1,y1),(x2,y2)] 两种写法)"""
        if len(xy) == 2 and hasattr(xy[0], "__len__"):
            (x1, y1), (x2, y2) = xy
        else:
            x1, y1, x2, y2 = xy
        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=r, fill=fill)

    def _estimate_msg_height(self, msg: ChatMessage, chat_font, max_bubble_w: int,
                             line_height: int, margin_y: int,
                             style: "AppStyle" = None) -> int:
        """估算消息渲染高度（不实际绘制）/ Estimate rendered height without drawing

        用于跳过不可见消息时正确推进 y_cursor / Used to correctly advance y_cursor when skipping invisible messages
        """
        pad_x = style.bubble_pad_x if style is not None else 10
        pad_y = style.bubble_pad_y if style is not None else 8
        voice_h = style.voice_bar_height if style is not None else 44
        if msg.msg_type == "text":
            lines = self._wrap_pil(msg.text, chat_font, max_bubble_w - 2 * pad_x)
            return len(lines) * line_height + 2 * pad_y + margin_y * 2
        elif msg.msg_type == "image":
            iw = min(msg.media_width or 180, max_bubble_w)
            ih = msg.media_height or int(iw * 0.75)
            return ih + margin_y * 2
        elif msg.msg_type == "sticker":
            sticker_size = min(msg.media_width or 120, max_bubble_w)
            return sticker_size + margin_y * 2
        elif msg.msg_type == "voice":
            # 与 render_chat_frame 的语音分支保持一致: 转写文字在气泡**内**,
            # 气泡高度 = 语音行 + 内边距 + 文字行。
            # / Must match the voice branch in render_chat_frame: the transcript sits
            # INSIDE the bubble and the bubble height = voice row + pad + text lines.
            h = voice_h + margin_y * 2
            if msg.voice_transcribed:
                trans_lines = self._wrap_pil(msg.voice_transcribed, chat_font, max_bubble_w - 2 * pad_x)
                h += pad_y + len(trans_lines) * line_height
            return h
        elif msg.msg_type == "video":
            vw = min(msg.media_width or 200, max_bubble_w)
            vh = msg.media_height or int(vw * 0.5625)
            return vh + margin_y * 2
        elif msg.msg_type == "transfer":
            return 96 + margin_y * 2
        elif msg.msg_type == "screenshot":
            sw = min(msg.media_width or 220, max_bubble_w)
            sh = msg.media_height or int(sw * 1.4)
            return sh + margin_y * 2
        elif msg.msg_type == "system":
            # 与 render_chat_frame 的系统提示分支保持一致(小号居中灰字行, 无气泡)
            # / Must match the system-line branch in render_chat_frame.
            return self._system_line_height(chat_font) + margin_y * 2
        else:
            return line_height + margin_y * 2

    # ------------------------------------------------------------------
    # 拟真 UI 辅助绘制 / Realistic-UI helpers
    # ------------------------------------------------------------------
    def _divider_height(self, style: AppStyle, line_height: int, margin_y: int) -> int:
        """时间分隔条占用的高度(无则 0) / Height reserved by the time divider."""
        return line_height + margin_y if style.divider_text else 0

    def _make_wallpaper(self, style: AppStyle) -> Image.Image:
        """墙纸: 图片壁纸(优先) / 纯平底色(+可选极淡纹理)。

        图片模式读取 style.wallpaper_image 并缩放为 WxH; 与底色模式一样缓存后每次
        返回 `.copy()`, 因此**内容相同的帧逐字节一致**(不干扰去重/ROI)。
        / Image wallpaper (preferred) or flat base colour with optional faint noise.
        The image is resized to WxH and cached; identical content stays byte-identical.
        """
        W, H = self.W, self.H
        if style.wallpaper_image:
            key = ("img", style.wallpaper_image, W, H)
            if key not in self._wallpaper_cache:
                if os.path.isfile(style.wallpaper_image):
                    im = Image.open(style.wallpaper_image).convert("RGB")
                    if im.size != (W, H):
                        im = im.resize((W, H), Image.Resampling.LANCZOS)
                    self._wallpaper_cache[key] = im
                else:
                    # 图片缺失: 回退到纯平底色, 不打断渲染 / fall back to flat colour
                    self._wallpaper_cache[key] = Image.new(
                        "RGB", (W, H), tuple(int(v) for v in style.bg_color[::-1]))
            return self._wallpaper_cache[key].copy()
        rgb = tuple(int(v) for v in style.bg_color[::-1])   # BGR -> RGB
        if style.wallpaper_noise_std <= 0:
            return Image.new("RGB", (W, H), rgb)
        key = (W, H, round(float(style.wallpaper_noise_std), 3), rgb)
        if key not in self._wallpaper_cache:
            rng = np.random.RandomState(20240516)
            noise = rng.normal(0.0, style.wallpaper_noise_std, size=(H, W)).astype(np.float32)
            base = np.tile(np.array(rgb, dtype=np.float32), (H, W, 1))
            arr = np.clip(base + noise[:, :, None], 0, 255).astype(np.uint8)
            self._wallpaper_cache[key] = Image.fromarray(arr)
        return self._wallpaper_cache[key].copy()

    def _system_font_size(self, chat_font) -> int:
        """系统提示字号(正文的 SYSTEM_FONT_RATIO 倍, 不小于 12px) / system-line size."""
        base = int(getattr(chat_font, "size", 24) or 24)
        return max(12, int(base * SYSTEM_FONT_RATIO))

    def _system_line_height(self, chat_font) -> int:
        """系统提示行高(渲染与高度估计共用, 保证底部锚定对齐)。"""
        return self._system_font_size(chat_font) + 6

    def _draw_time_divider(self, draw, style: AppStyle, font, W: int, y0: int) -> None:
        """居中时间分隔条(如 "18:48") / Centred time divider."""
        txt = style.divider_text
        bb = font.getbbox(txt)
        draw.text(((W - (bb[2] - bb[0])) // 2, y0 + 2), txt, font=font,
                  fill=tuple(style.divider_text_color[::-1]))

    def _draw_bubble_tail(self, draw, bx: int, by: int, bw: int, bh: int,
                          is_recv: bool, color, radius: int) -> None:
        """气泡小尾巴(朝向头像一侧的小三角) / Bubble tail pointing at the avatar."""
        ty = by + min(max(radius, bh // 4), max(radius, bh - 18))
        if is_recv:
            draw.polygon([(bx + 1, ty), (bx + 1, ty + 12), (bx - 6, ty + 6)], fill=color)
        else:
            draw.polygon([(bx + bw - 1, ty), (bx + bw - 1, ty + 12),
                          (bx + bw + 6, ty + 6)], fill=color)

    def _draw_status_icons(self, draw, style: AppStyle, W: int, sbh: int) -> None:
        """右上角信号/WiFi/电量图标 / Signal + WiFi + battery glyphs."""
        c = tuple(style.status_text_color[::-1])
        cy = sbh // 2
        x = int(W * 0.755)
        for i, h in enumerate((6, 9, 12, 15)):
            draw.rectangle([(x + i * 5, cy + 7 - h), (x + i * 5 + 3, cy + 7)], fill=c)
        wx = x + 30
        for r in (5, 9, 13):
            draw.arc([(wx - r, cy - 2 - r), (wx + r, cy - 2 + r)], 205, 335,
                     fill=c, width=2)
        draw.ellipse([(wx - 1, cy + 5), (wx + 1, cy + 7)], fill=c)
        bx = x + 50
        draw.rectangle([(bx, cy - 6), (bx + 21, cy + 6)], outline=c, width=1)
        draw.rectangle([(bx + 21, cy - 2), (bx + 23, cy + 2)], fill=c)
        draw.rectangle([(bx + 2, cy - 4), (bx + 15, cy + 4)], fill=c)

    def _draw_input_bar(self, draw, style: AppStyle, font, W: int, H: int, nh: int) -> None:
        """底部输入条: 浅色条 + 圆角输入框 + 提示文字 / Bottom input bar."""
        y0 = H - nh
        cy = y0 + nh // 2
        field_h = max(20, int(nh * 0.56))
        fx0, fx1 = int(W * 0.13), int(W * 0.87)
        fy0, fy1 = cy - field_h // 2, cy + field_h // 2
        self._pil_draw_rounded_rect(draw, (fx0, fy0, fx1, fy1),
                                    max(4, field_h // 2),
                                    tuple(style.input_field_color[::-1]))
        icon = (150, 150, 150)
        # 左侧语音/键盘按钮 + 右侧表情按钮(圆形示意)
        r = max(8, nh // 8)
        draw.ellipse([(int(W * 0.04), cy - r), (int(W * 0.04) + 2 * r, cy + r)],
                     outline=icon, width=2)
        draw.ellipse([(int(W * 0.92), cy - r), (int(W * 0.92) + 2 * r, cy + r)],
                     outline=icon, width=2)
        if style.input_hint:
            bb = font.getbbox(style.input_hint)
            hh = bb[3] - bb[1]
            tw = bb[2] - bb[0]
            draw.text((fx0 + (fx1 - fx0 - tw) // 2, cy - hh // 2 - bb[1]),
                      style.input_hint, font=font,
                      fill=tuple(style.input_hint_color[::-1]))

    def render_chat_frame(self, msgs: List[ChatMessage], scroll_y: int, style: AppStyle) -> np.ndarray:
        W, H = self.W, self.H

        th = style.title_height
        nh = style.nav_height
        sbh = style.status_bar_height
        top = style.top_offset          # 内容视口上沿 = 状态栏 + 标题栏

        # 画布: 墙纸底色 + 可选极淡纹理(固定种子, 静止帧逐字节一致)
        pil_img = self._make_wallpaper(style)
        draw = self._make_draw(pil_img)

        base_font_size = int(16 * style.font_scale / 0.4)
        chat_font = self._get_font(style, base_font_size)
        title_font = self._get_font(style, int(base_font_size * 1.2), bold=True)
        line_height = int(base_font_size * style.line_spacing)
        pad_x, pad_y = style.bubble_pad_x, style.bubble_pad_y

        # 1./2. 标题栏与导航栏在内容绘制**之后**再画(见函数末尾),
        # 使聊天内容像真实 App 一样"滚动到标题栏/导航栏下方", 而不是盖住它们。

        # 3. 聊天内容区域（裁剪绘制）
        max_bubble_w = int(W * style.bubble_max_w_ratio)
        avatar_size = style.avatar_size
        margin_x, margin_y = 10, 8
        # 内容原点从标题栏下沿开始: 滚动为 0 时第一条消息紧贴标题栏下方(真实 App 行为),
        # 滚动时内容上移并被标题栏遮住。
        y_cursor = top - scroll_y

        # 时间分隔条(会话起始时间, 随内容滚动; 置于消息之前)
        dh = self._divider_height(style, line_height, margin_y)
        if dh and y_cursor > -200:
            self._draw_time_divider(draw, style, chat_font, W, y_cursor)
        y_cursor += dh

        for msg_idx, msg in enumerate(msgs):
            if y_cursor > H - nh + 200:
                break
            if y_cursor < -300:
                y_cursor += self._estimate_msg_height(msg, chat_font, max_bubble_w,
                                                      line_height, margin_y, style)
                continue

            is_recv = msg.is_received
            bubble_color = tuple((style.recv_bubble if is_recv else style.sent_bubble)[::-1])
            text_color = tuple(style.text_color[::-1])
            # 发送侧若单独配了文字色(如深蓝气泡配白字)则用之, 保证对比度。
            # / Sent-side text colour (e.g. white on a deep-blue bubble) for contrast.
            run_text_color = (tuple(style.sent_text_color[::-1])
                              if (not is_recv and style.sent_text_color) else text_color)
            
            rendered_h = 0  # 本条消息占用的总高度

            # ========== 按消息类型分发渲染 ==========
            
            if msg.msg_type == "image":
                # --- 新增逻辑：尝试加载真实图片 ---
                img_asset = None
                iw, ih = 180, int(180 * 0.75) # 默认尺寸

                # 1. 优先尝试从真实素材库加载
                if self.asset_loader and self.asset_loader.pools.get('photos'):
                    img_path = self.asset_loader.get_deterministic('photos', msg_idx)
                    if img_path:
                        # 使用 OpenCV 读取图片 (BGR格式)
                        img_cv = cv2.imread(img_path)
                        if img_cv is not None:
                            # 获取原始尺寸
                            h, w = img_cv.shape[:2]
                            # 计算新尺寸，保持宽高比，宽度不超过最大气泡宽度
                            iw = min(msg.media_width or 180, max_bubble_w)
                            ih = int(h * (iw / w))
                            # 限制最大高度，避免图片过长
                            ih = min(ih, 400)
                            # 缩放图片
                            img_cv = cv2.resize(img_cv, (iw, ih), interpolation=cv2.INTER_AREA)
                            # 转换颜色空间 BGR -> RGB，以匹配 PIL
                            img_asset = img_cv[:, :, ::-1]

                # 2. 如果没有真实素材或加载失败，则回退到程序生成的占位图
                if img_asset is None:
                    iw = min(msg.media_width or 180, max_bubble_w)
                    ih = msg.media_height or int(iw * 0.75)
                    # 调用生成器创建占位图 (BGR格式)
                    img_placeholder_bgr = self.media_gen.gen_image_placeholder(iw, ih, seed=hash(msg.text))
                    # 转换颜色空间 BGR -> RGB
                    img_asset = img_placeholder_bgr[:, :, ::-1]

                # --- 绘制图片 ---
                # 计算图片位置
                bx = margin_x + avatar_size + 5 if is_recv else W - margin_x - avatar_size - 5 - iw
                # 将 numpy 数组转换为 PIL 图像并粘贴
                pil_asset = Image.fromarray(img_asset)
                pil_img.paste(pil_asset, (bx, max(0, y_cursor)))
                rendered_h = ih + margin_y * 2

            elif msg.msg_type == "sticker":
                # --- 新增逻辑：尝试加载真实表情包 ---
                sticker_asset = None
                sticker_size = min(msg.media_width or 120, max_bubble_w)

                if self.asset_loader and self.asset_loader.pools.get('stickers'):
                    sticker_path = self.asset_loader.get_deterministic('stickers', msg_idx)
                    if sticker_path:
                        try:
                            pil_sticker = Image.open(sticker_path).convert("RGBA")
                            pil_sticker = pil_sticker.resize((sticker_size, sticker_size), Image.Resampling.LANCZOS)
                            sticker_asset = np.array(pil_sticker)
                        except Exception:
                            sticker_asset = None  # 回退到占位图 / Fallback to placeholder

                # 如果没有真实素材，则回退到程序生成的占位图
                if sticker_asset is None:
                    # 生成器返回的是 BGR 格式
                    sticker_bgr = self.media_gen.gen_sticker(msg.sticker_name or msg.text, sticker_size)
                    # 转换为 RGB
                    sticker_asset = sticker_bgr[:, :, ::-1]

                # --- 绘制表情包 ---
                bx = margin_x + avatar_size + 5 if is_recv else W - margin_x - avatar_size - 5 - sticker_size
                # 真实素材带透明通道时用自身作 mask; 占位图是 3 通道 RGB, 无 alpha,
                # 直接拿它当 mask 会报 "bad transparency mask", 故仅在 4 通道时传 mask。
                # / Use the asset itself as mask only when it has alpha; the 3-channel
                # placeholder has none, and passing it as a mask raises PIL's error.
                st_img = Image.fromarray(sticker_asset)
                st_mask = st_img if sticker_asset.ndim == 3 and sticker_asset.shape[2] == 4 else None
                pil_img.paste(st_img, (bx, max(0, y_cursor)), st_mask)
                rendered_h = sticker_size + margin_y * 2

            elif msg.msg_type == "voice":
                # --- 新增逻辑：尝试加载真实语音 ---
                voice_asset = None
                voice_h = style.voice_bar_height
                voice_w = min(max(120, int(msg.voice_duration * 8 + 60)), max_bubble_w)

                # 转写文字(若有)与语音行同处**一个气泡内**(真实微信行为): 气泡宽度取
                # 「语音行/文字行」较大者, 高度 = 语音行 + 文字行。旧实现把转写文字画在
                # 气泡**外面**的墙纸上, 导致真实样本中"文字落在白色气泡内"的占比(synth
                # 仅 4%, real 27-34%)对不上。
                # / Transcript (if any) lives INSIDE the same bubble as the voice row —
                # what real WeChat does. The old code drew it on the wallpaper *outside*
                # the bubble, which is why only 4% of synth text sat on white vs 27-34%
                # in the real recordings.
                trans_lines = []
                if msg.voice_transcribed:
                    trans_lines = self._wrap_pil(msg.voice_transcribed, chat_font,
                                                 max_bubble_w - 2 * pad_x)
                text_w = 0
                for tl in trans_lines:
                    bb = chat_font.getbbox(tl)
                    text_w = max(text_w, bb[2] - bb[0])
                bw = min(max(voice_w, text_w) + 2 * pad_x, max_bubble_w)
                bh = voice_h + (pad_y + len(trans_lines) * line_height if trans_lines else 0)

                if self.asset_loader and self.asset_loader.pools.get('voices'):
                    voice_path = self.asset_loader.get_deterministic('voices', msg_idx)
                    if voice_path:
                        try:
                            pil_voice = Image.open(voice_path).convert("RGBA")
                            pil_voice = pil_voice.resize((voice_w, voice_h), Image.Resampling.LANCZOS)
                            voice_asset = np.array(pil_voice)
                        except Exception:
                            voice_asset = None  # 语音文件非图片，回退到占位图 / Voice file is not image, fallback

                # 如果没有真实素材，则回退到程序生成的占位图
                if voice_asset is None:
                    # 生成器返回 RGBA(小图标 + 时长), 直接按 alpha 合成;
                    # 图标/时长颜色取气泡文字色(深蓝气泡 → 白图标), 保证可见。
                    # / Icon/duration ink = bubble text colour (white on deep blue).
                    voice_asset = self.media_gen.gen_voice_bubble(
                        msg.voice_duration, msg.has_waveform, msg.voice_transcribed,
                        chat_font, voice_w, voice_h, is_recv, ink=run_text_color
                    )

                # --- 绘制语音气泡 ---
                bx = margin_x + avatar_size + 5 if is_recv else W - margin_x - avatar_size - 5 - bw
                # 先画气泡底色
                self._pil_draw_rounded_rect(draw, (bx, y_cursor, bx + bw, y_cursor + bh), style.bubble_radius, bubble_color)
                if style.bubble_tail:
                    self._draw_bubble_tail(draw, bx, y_cursor, bw, bh,
                                           is_recv, bubble_color, style.bubble_radius)
                # 语音行(喇叭图标+时长)贴气泡顶部
                pil_img.paste(Image.fromarray(voice_asset[:, :, :3]),
                              (bx + pad_x, y_cursor + pad_y // 2),
                              Image.fromarray(voice_asset[:, :, 3]))
                # 转写文字画在同一气泡内、语音行下方
                ty = y_cursor + voice_h + pad_y // 2
                for tl in trans_lines:
                    bb = chat_font.getbbox(tl)
                    if is_recv:
                        tx = bx + pad_x
                    else:
                        # 发送侧右对齐: 气泡右内边距处收尾
                        tx = bx + bw - pad_x - (bb[2] - bb[0])
                    draw.text((tx, ty), tl, font=chat_font, fill=run_text_color)
                    ty += line_height
                rendered_h = bh + margin_y * 2


            elif msg.msg_type == "video":
                # --- 新增逻辑：尝试加载真实视频封面 ---
                cover_asset = None
                vw = min(msg.media_width or 200, max_bubble_w)
                vh = msg.media_height or int(vw * 0.5625) # 16:9

                if self.asset_loader and self.asset_loader.pools.get('videos'):
                    video_path = self.asset_loader.get_deterministic('videos', msg_idx)
                    if video_path:
                        # 使用 OpenCV 读取视频的第一帧作为封面
                        cap = cv2.VideoCapture(video_path)
                        ret, frame = cap.read()
                        if ret:
                            # 缩放封面
                            frame = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA)
                            # 转换为 RGB
                            cover_asset = frame[:, :, ::-1]
                            # 可以在封面上叠加一个播放图标，这里为了简化省略
                        cap.release()

                # 如果没有真实素材，则回退到程序生成的占位图
                if cover_asset is None:
                    # 生成器返回的是 BGR 格式
                    cover_bgr = self.media_gen.gen_video_cover(vw, vh, msg.video_cover_color)
                    # 转换为 RGB
                    cover_asset = cover_bgr[:, :, ::-1]

                # --- 绘制视频封面 ---
                bx = margin_x + avatar_size + 5 if is_recv else W - margin_x - avatar_size - 5 - vw
                pil_img.paste(Image.fromarray(cover_asset), (bx, max(0, y_cursor)))
                rendered_h = vh + margin_y * 2

            elif msg.msg_type == "transfer": # 红包、转账
                # --- 新增逻辑：尝试加载真实转账截图 ---
                card_asset = None
                # 预设一个卡片尺寸
                cw, ch = 240, 96

                if self.asset_loader and self.asset_loader.pools.get('docs'):
                    # 从文档池中筛选可能是转账截图的图片（例如文件名包含 transfer, redpacket, pay 等）
                    # 这里简化处理，直接随机取一个，实际可根据文件名做更精细的筛选
                    doc_path = self.asset_loader.get_deterministic('docs', msg_idx)
                    if doc_path:
                        # 尝试用 PIL 读取
                        try:
                            pil_card = Image.open(doc_path).convert("RGB")
                            # 缩放到合适大小
                            pil_card = pil_card.resize((cw, ch), Image.Resampling.LANCZOS)
                            card_asset = np.array(pil_card)
                        except Exception:
                            pass

                # 如果没有真实素材，则回退到程序生成的占位图
                if card_asset is None:
                    # 生成器返回 RGBA(橙色圆角卡片), 按 alpha 合成
                    card_asset = self.media_gen.gen_transfer_card(
                        msg.transfer_amount, msg.transfer_status, is_recv, chat_font
                    )
                    ch, cw = card_asset.shape[0], card_asset.shape[1]

                # --- 绘制转账卡片 ---
                bx = margin_x + avatar_size + 5 if is_recv else W - margin_x - avatar_size - 5 - cw
                card_img = Image.fromarray(card_asset)
                card_mask = (card_img if card_asset.ndim == 3 and card_asset.shape[2] == 4
                             else None)
                pil_img.paste(card_img, (bx, max(0, y_cursor)), card_mask)
                rendered_h = ch + margin_y * 2

            elif msg.msg_type == "screenshot": # 聊天内截图
                # --- 新增逻辑：尝试加载真实截图 ---
                ss_asset = None
                sw = min(msg.media_width or 220, max_bubble_w)
                sh = msg.media_height or int(sw * 1.4)

                if self.asset_loader and self.asset_loader.pools.get('photos'):
                    # 截图也可以从 photos 池中获取
                    img_path = self.asset_loader.get_deterministic('photos', msg_idx + 1000)
                    if img_path:
                        img_cv = cv2.imread(img_path)
                        if img_cv is not None:
                            img_cv = cv2.resize(img_cv, (sw, sh), interpolation=cv2.INTER_AREA)
                            ss_asset = img_cv[:, :, ::-1] # BGR -> RGB

                # 如果没有真实素材，则回退到程序生成的占位图
                if ss_asset is None:
                    # 生成器返回的是 BGR 格式
                    ss_bgr = self.media_gen.gen_screenshot_embed(msg.screenshot_app or "支付宝", sw, sh)
                    # 转换为 RGB
                    ss_asset = ss_bgr[:, :, ::-1]

                # --- 绘制截图 ---
                bx = margin_x + avatar_size + 5 if is_recv else W - margin_x - avatar_size - 5 - sw
                pil_img.paste(Image.fromarray(ss_asset), (bx, max(0, y_cursor)))
                rendered_h = sh + margin_y * 2

            elif msg.msg_type == "system": # 系统提示行
                # 真实录屏里消息之间会有系统提示(时间戳/撤回/加好友等): 渲染为
                # **居中 + 小号 + 灰色 + 无气泡** 的文本行, 不带头像。
                # / Real recordings interleave system lines (timestamps, recalls,
                # "added as friend" etc.): centred, small, grey, no bubble/avatar.
                sys_font = self._get_font(style, self._system_font_size(chat_font))
                if self.label_recorder is not None:
                    self.label_recorder.current_kind = "system"
                sbbox = sys_font.getbbox(msg.text)
                sx = (W - (sbbox[2] - sbbox[0])) // 2
                draw.text((sx, y_cursor + margin_y), msg.text, font=sys_font,
                          fill=tuple(style.secondary_text[::-1]))
                if self.label_recorder is not None:
                    self.label_recorder.current_kind = "message"
                rendered_h = self._system_line_height(chat_font) + margin_y * 2

            else:  # text
                # --- 新增逻辑：使用 PIL 渲染高质量文本 ---
                lines = self._wrap_pil(msg.text, chat_font, max_bubble_w - 2 * pad_x)
                if style.fit_bubble_to_text:
                    # 气泡宽度贴合文字(真实微信短消息不撑满整屏)
                    text_w = max((chat_font.getbbox(l)[2] - chat_font.getbbox(l)[0])
                                 for l in lines)
                    bw = min(max_bubble_w, text_w + 2 * pad_x)
                else:
                    bw = max_bubble_w
                bh = len(lines) * line_height + 2 * pad_y
                bx = margin_x + avatar_size + 5 if is_recv else W - margin_x - avatar_size - 5 - bw

                # 1. 先画气泡底色
                self._pil_draw_rounded_rect(draw, (bx, y_cursor, bx+bw, y_cursor+bh), style.bubble_radius, bubble_color)
                if style.bubble_tail:
                    self._draw_bubble_tail(draw, bx, y_cursor, bw, bh,
                                           is_recv, bubble_color, style.bubble_radius)

                # 2. 再用 PIL 逐行绘制文本
                ty = y_cursor + pad_y
                for line in lines:
                    draw.text((bx + pad_x, ty), line, font=chat_font, fill=run_text_color)
                    ty += line_height

                rendered_h = bh + margin_y * 2

            # 绘制头像
            if style.has_avatar and msg.msg_type != "system":
                ay = y_cursor + (rendered_h - margin_y * 2 - avatar_size) // 2
                ax = margin_x if is_recv else W - margin_x - avatar_size
                av_radius = max(4, avatar_size // 6)
                
                # # 优先尝试加载真实头像
                avatar_path = None
                if self.asset_loader:
                    avatar_path = self.asset_loader.get_avatar(msg.sender_id) 

                if avatar_path and os.path.isfile(avatar_path):
                    try:
                        # 用 PIL 读取头像并缩放为圆角方形(微信头像为圆角方形, 非正圆)
                        av_img = Image.open(avatar_path).convert("RGB")
                        av_img = av_img.resize((avatar_size, avatar_size), Image.Resampling.LANCZOS)
                        
                        # 创建圆角方形蒙版
                        mask = Image.new("L", (avatar_size, avatar_size), 0)
                        mask_draw = ImageDraw.Draw(mask)
                        mask_draw.rounded_rectangle((0, 0, avatar_size - 1, avatar_size - 1),
                                                    radius=av_radius, fill=255)
                        
                        # 粘贴到画布上（用 mask 实现圆角裁剪）
                        pil_img.paste(av_img, (ax, ay), mask)
                    except Exception:
                        # 加载失败，回退到纯色圆角方形
                        av_color = (180, 180, 180) if is_recv else (100, 180, 100)
                        self._pil_draw_rounded_rect(draw, [(ax, ay), (ax + avatar_size, ay + avatar_size)],
                                                    av_radius, av_color)
                else:
                    # 没有头像素材，回退到纯色圆角方形
                    av_color = (180, 180, 180) if is_recv else (100, 180, 100)
                    self._pil_draw_rounded_rect(draw, [(ax, ay), (ax + avatar_size, ay + avatar_size)], av_radius, av_color)
            
            y_cursor += rendered_h

        # 1./2. 顶部系统状态栏 + 标题栏 + 底部输入条(最后绘制, 覆盖内容区上/下边界)
        if sbh > 0:
            self._draw_status_bar(draw, style, W, sbh, base_font_size)
        draw.rectangle([(0, sbh), (W, sbh + th)], fill=tuple(style.title_bar_color[::-1]))
        title_str = style.chat_title or style.app_type.value
        title_bbox = title_font.getbbox(title_str)
        tw, thh = title_bbox[2] - title_bbox[0], title_bbox[3] - title_bbox[1]
        title_color = tuple(style.title_text_color[::-1])
        draw.text(((W - tw) // 2, sbh + (th - thh) // 2 - title_bbox[1]),
                  title_str, font=title_font, fill=title_color)
        if style.chat_title:
            # 真实微信标题栏左侧有返回箭头
            draw.text((int(W * 0.05), sbh + (th - thh) // 2 - title_bbox[1]),
                      "<", font=title_font, fill=title_color)
        nav_col = style.input_bar_color or style.nav_bar_color
        draw.rectangle([(0, H - nh), (W, H)], fill=tuple(nav_col[::-1]))
        # 仅拟真样式绘制输入框细节(其余样式保持旧行为: 纯色导航栏)
        if style.input_bar_color is not None or style.input_hint:
            self._draw_input_bar(draw, style, chat_font, W, H, nh)

        # 4. PIL -> OpenCV numpy array (RGB -> BGR)
        frame = np.array(pil_img)[:, :, ::-1].copy()
        return frame

    def _draw_status_bar(self, draw, style: AppStyle, W: int, sbh: int,
                         base_font_size: int) -> None:
        """顶部系统状态栏(底色 + 时间 + 信号/WiFi/电量) / OS status bar."""
        draw.rectangle([(0, 0), (W, sbh)], fill=tuple(style.bg_color[::-1]))
        if style.status_time:
            sfont = self._get_font(style, max(12, int(base_font_size * 0.62)))
            bb = sfont.getbbox(style.status_time)
            draw.text((int(W * 0.08), (sbh - (bb[3] - bb[1])) // 2 - bb[1]),
                      style.status_time, font=sfont,
                      fill=tuple(style.status_text_color[::-1]))
        self._draw_status_icons(draw, style, W, sbh)

    def render_video_fullscreen_frame(self, frame_idx: int, style: AppStyle) -> np.ndarray:
        """渲染全屏视频播放帧 / Render fullscreen video playback frame"""
        W, H = self.W, self.H
        pil_img = Image.new('RGB', (W, H), (0, 0, 0))
        draw = self._make_draw(pil_img)

        base_font_size = int(16 * style.font_scale / 0.4)
        font = self._get_font(style, base_font_size)

        # 模拟视频内容：缓慢变化的渐变背景 / Simulated video content: slowly shifting gradient
        segment = frame_idx // 15  # 每15帧换一次色调 / Change hue every 15 frames
        rng = np.random.RandomState(segment)
        c1 = tuple(int(v) for v in rng.randint(20, 80, size=3))
        c2 = tuple(int(v) for v in rng.randint(20, 80, size=3))
        for y in range(H):
            t = y / max(H - 1, 1)
            c = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
            draw.line([(0, y), (W, y)], fill=c)

        # 播放按钮（中心）/ Play button (center)
        cx, cy = W // 2, H // 2
        r = 40
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 255, 255))
        draw.polygon([(cx - 12, cy - 18), (cx - 12, cy + 18), (cx + 18, cy)], fill=(0, 0, 0))

        # 进度条（底部）/ Progress bar (bottom)
        bar_y = H - 60
        draw.rectangle([(20, bar_y), (W - 20, bar_y + 4)], fill=(60, 60, 60))
        progress = min(1.0, frame_idx / 300.0)
        draw.rectangle([(20, bar_y), (20 + int((W - 40) * progress), bar_y + 4)], fill=(255, 255, 255))

        # 返回按钮（左上角）/ Back button (top-left)
        draw.text((15, 15), "< 返回", font=font, fill=(255, 255, 255))

        frame = np.array(pil_img)[:, :, ::-1].copy()
        return frame

# ==============================================================================
# PART 5: 行为引擎 / Behavior Engine
# ==============================================================================

class BehaviorEngine:
    """行为引擎：生成真实的滚动序列 / Behavior engine: generate realistic scroll sequences

    核心改进 / Key improvements:
    - 段式行为 (Segment-based)：每个手势持续1-5秒，不再每帧随机切换
    - 缓动插值 (Easing)：滑动有加速和减速过程，模拟手指惯性
    - 静止停顿 (Dwell)：阅读时完全不动，符合真实用户行为
    """

    # 用户画像 -> 行为段权重 / User profile -> segment behavior weights
    WEIGHT_PROFILES = {
        "normal_user": {"slow_scroll": 0.35, "static_reading": 0.35, "variable_scroll": 0.15, "view_image": 0.15},
        "fraud_victim": {"static_reading": 0.45, "slow_scroll": 0.25, "view_image": 0.20, "variable_scroll": 0.10},
        "fraud_suspect": {"fast_scroll": 0.30, "static_reading": 0.25, "variable_scroll": 0.25, "slow_scroll": 0.20},
        "elderly_user": {"static_reading": 0.50, "slow_scroll": 0.35, "variable_scroll": 0.15},
        "power_user": {"fast_scroll": 0.20, "variable_scroll": 0.25, "static_reading": 0.30, "slow_scroll": 0.25}
    }

    @staticmethod
    def _ease_in_out_cubic(t: float) -> float:
        """三次缓入缓出 / Cubic ease-in-out"""
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2

    @staticmethod
    def _ease_out_cubic(t: float) -> float:
        """三次缓出（滑动手势惯用）/ Cubic ease-out (for scroll gestures)"""
        return 1 - pow(1 - t, 3)

    @staticmethod
    def _generate_segment(action: str, fps: int, current_y: int, max_scroll: int,
                          rng: random.Random) -> Tuple[List[int], int, int]:
        """生成一个行为段 / Generate one behavior segment

        返回: (该段的滚动位置列表, 段结束后的y位置, 段持续帧数)
        Returns: (scroll positions for this segment, y after segment, segment frame count)
        """
        positions = []

        if action in ("static_reading", "view_image", "voice_playback",
                       "transfer_screen", "avatar_click", "large_font_static"):
            # 静止段：2-6秒完全不滚动 / Dwell: 2-6s no scroll at all
            seg_frames = rng.randint(fps * 2, fps * 6)
            positions = [current_y] * seg_frames
            return positions, current_y, seg_frames

        if action == "slow_scroll":
            # 慢速滑动：1.5-4秒，总位移30-120px，带缓动 / Slow scroll with easing
            seg_frames = rng.randint(int(fps * 1.5), fps * 4)
            displacement = rng.randint(30, 120)
            direction = 1  # 向下 / downward
            if rng.random() < 0.15:
                direction = -1  # 偶尔向上回看 / occasionally scroll up

            target_y = max(0, min(max_scroll, current_y + displacement * direction))
            actual_disp = target_y - current_y

            for f in range(seg_frames):
                t = f / max(seg_frames - 1, 1)
                eased = BehaviorEngine._ease_in_out_cubic(t)
                y = current_y + int(actual_disp * eased)
                positions.append(y)
            return positions, target_y, seg_frames

        if action == "fast_scroll":
            # 快速滑动：0.8-2秒，总位移200-500px，先加速后减速 / Fast scroll with momentum
            seg_frames = rng.randint(int(fps * 0.8), fps * 2)
            displacement = rng.randint(200, 500)
            direction = 1
            if rng.random() < 0.10:
                direction = -1

            target_y = max(0, min(max_scroll, current_y + displacement * direction))
            actual_disp = target_y - current_y

            for f in range(seg_frames):
                t = f / max(seg_frames - 1, 1)
                # 快速滑动用 ease-out：先快后慢（手指甩出后惯性减速）
                eased = BehaviorEngine._ease_out_cubic(t)
                y = current_y + int(actual_disp * eased)
                positions.append(y)

            # 滑动后加0.5-1秒静止（惯性停止后阅读）/ Add brief dwell after fast scroll
            dwell = rng.randint(int(fps * 0.5), fps)
            positions.extend([target_y] * dwell)
            return positions, target_y, len(positions)

        if action == "variable_scroll":
            # 变速滑动：3-6秒，多个小滑动段组合 / Variable speed: multiple mini-scrolls
            seg_frames = rng.randint(fps * 3, fps * 6)
            y = current_y
            sub_segments = rng.randint(2, 4)
            sub_len = seg_frames // sub_segments

            for s in range(sub_segments):
                sub_disp = rng.randint(20, 80) * rng.choice([1, 1, 1, -1])
                target = max(0, min(max_scroll, y + sub_disp))
                actual = target - y

                for f in range(sub_len):
                    t = f / max(sub_len - 1, 1)
                    eased = BehaviorEngine._ease_in_out_cubic(t)
                    positions.append(y + int(actual * eased))

                y = target
                # 子段之间加短暂停顿 / Brief pause between sub-segments
                pause = rng.randint(int(fps * 0.3), int(fps * 0.8))
                positions.extend([y] * pause)

            return positions[:seg_frames], positions[min(len(positions)-1, seg_frames-1)], min(len(positions), seg_frames)

        if action == "reciprocating":
            # 往返滑动：3-5秒，来回滑动 / Reciprocating: scroll back and forth
            seg_frames = rng.randint(fps * 3, fps * 5)
            half = seg_frames // 2
            disp = rng.randint(40, 100)
            target_down = min(max_scroll, current_y + disp)

            # 下半段：向下滑 / First half: scroll down
            for f in range(half):
                t = f / max(half - 1, 1)
                eased = BehaviorEngine._ease_in_out_cubic(t)
                positions.append(current_y + int((target_down - current_y) * eased))

            # 上半段：滑回来 / Second half: scroll back up
            for f in range(seg_frames - half):
                t = f / max(seg_frames - half - 1, 1)
                eased = BehaviorEngine._ease_in_out_cubic(t)
                positions.append(target_down + int((current_y - target_down) * eased))

            return positions, current_y, seg_frames

        if action == "overscroll_bounce":
            # 过界回弹：1-2秒，滑到底部后弹回 / Overscroll bounce
            seg_frames = rng.randint(fps, fps * 2)
            overshoot = rng.randint(15, 40)
            target = min(max_scroll + overshoot, max_scroll + 50)

            # 快速滑过底部 / Quick scroll past bottom
            peak_frame = seg_frames // 3
            for f in range(peak_frame):
                t = f / max(peak_frame - 1, 1)
                eased = BehaviorEngine._ease_out_cubic(t)
                positions.append(current_y + int((target - current_y) * eased))

            # 弹回 / Bounce back
            bounce_frames = seg_frames - peak_frame
            for f in range(bounce_frames):
                t = f / max(bounce_frames - 1, 1)
                eased = BehaviorEngine._ease_in_out_cubic(t)
                positions.append(target + int((max_scroll - target) * eased))

            return positions, max_scroll, seg_frames

        # 默认：静止 / Default: dwell
        seg_frames = rng.randint(fps * 2, fps * 4)
        return [current_y] * seg_frames, current_y, seg_frames

    @staticmethod
    def generate_scroll_sequence(duration_sec: int, fps: int, profile: str, total_msgs: int) -> List[int]:
        """生成完整的滚动序列 / Generate complete scroll sequence

        核心设计 / Core design:
        - 将视频时长切分为多个行为段（每段1-6秒），而非每帧独立随机
        - 每个段内使用缓动函数，模拟手指滑动的物理特性
        - 段之间自然过渡，模拟用户"滑动→停顿阅读→再滑动"的真实节奏
        """
        total_frames = duration_sec * fps
        weights = BehaviorEngine.WEIGHT_PROFILES.get(profile, BehaviorEngine.WEIGHT_PROFILES["normal_user"])
        weight_keys = list(weights.keys())
        weight_vals = list(weights.values())

        # 估算内容总高度 / Estimate content height
        avg_msg_height = 60
        content_height = total_msgs * avg_msg_height
        viewport_height = 840  # 960 - 50(title) - 70(nav)
        max_scroll = max(1, content_height - viewport_height)

        # 随机起始偏移 / Random start offset
        start_offset = random.randint(0, max(0, max_scroll // 4))
        current_y = start_offset

        scroll_positions = []
        rng = random.Random(random.randint(0, 999999))

        while len(scroll_positions) < total_frames:
            # 选择下一个行为段 / Choose next behavior segment
            action = rng.choices(weight_keys, weights=weight_vals)[0]

            # 到达底部后不再向下滑 / Don't scroll down if at bottom
            if action in ("slow_scroll", "fast_scroll", "variable_scroll") and current_y >= max_scroll - 5:
                if rng.random() < 0.7:
                    action = rng.choice(["static_reading", "reciprocating", "overscroll_bounce"])

            seg_positions, current_y, seg_len = BehaviorEngine._generate_segment(
                action, fps, current_y, max_scroll, rng
            )

            scroll_positions.extend(seg_positions)

            # 段间加短暂过渡停顿（0.3-0.8秒）/ Brief transition pause between segments
            if len(scroll_positions) < total_frames:
                pause = rng.randint(int(fps * 0.3), int(fps * 0.8))
                scroll_positions.extend([current_y] * pause)

        # 裁剪到精确帧数 / Trim to exact frame count
        scroll_positions = scroll_positions[:total_frames]
        return scroll_positions


    # 生成一个 frame_type 序列，标记每一帧是 "chat" 还是 "video_fullscreen"
    @staticmethod
    def generate_frame_types(duration_sec: int, fps: int, messages: List[ChatMessage]) -> List[str]:
        frames = duration_sec * fps
        types = ["chat"] * frames
        
        # 找到所有视频消息的时间位置
        video_indices = [i for i, m in enumerate(messages) if m.msg_type == "video"]
        if not video_indices:
            return types
        
        # 随机选择1-3个视频进行"点击播放"模拟
        play_count = min(len(video_indices), random.randint(1, 3))
        selected = random.sample(video_indices, play_count)
        
        for vi in selected:
            # 估算该视频消息出现在屏幕上的大致帧范围
            approx_frame = int(vi / max(len(messages), 1) * frames * 0.6)
            play_start = min(approx_frame + random.randint(0, fps*2), frames - fps*4)
            play_duration = random.randint(fps * 2, fps * 6)  # 2~6秒播放
            play_end = min(play_start + play_duration, frames)
            
            for f in range(play_start, play_end):
                types[f] = "video_fullscreen"
        
        return types

# ==============================================================================
# PART 6: 任务定义与单视频渲染 / Task & Single Video Render
# ==============================================================================

@dataclass
class VideoTask:
    video_id: str
    style: AppStyle
    device: DeviceType
    fps: int
    duration_sec: int
    behavior_profile: str
    case_id: str = ""  # 案件ID，用于选择对应的对话消息 / Case ID for selecting conversation messages
    resolution: Tuple[int, int] = field(default=(540, 1200))

def render_single_video(task: VideoTask, messages: List[ChatMessage], 
                        output_dir: str, renderer: FrameRenderer,
                        label_dir: Optional[str] = None) -> str:
    """渲染单个视频（纯函数，无副作用，适合多进程）

    label_dir 非 None 时, 额外产出精确标签:
      <label_dir>/<video_id>.labels.jsonl  每帧每行文本(text + bbox[全帧坐标])
      <label_dir>/<video_id>.meta.json     真值消息/滚动序列/帧类型/ROI 提示
    """
    out_path = os.path.join(output_dir, f"{task.video_id}.mp4")
    
    scroll_seq = BehaviorEngine.generate_scroll_sequence(
        task.duration_sec, task.fps, task.behavior_profile, len(messages)
    )
    print(f"[DEBUG] 目标时长: {task.duration_sec}s, 目标FPS: {task.fps}, 实际生成帧数: {len(scroll_seq)}") 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, task.fps, task.resolution)
    
    special_pages = ["red_packet", "transfer", "image_view", "splash"]
    
    frame_types = BehaviorEngine.generate_frame_types(task.duration_sec, task.fps, messages)

    all_labels: List[Dict] = []
    for frame_idx in range(len(scroll_seq)):
        recorder = None
        if label_dir is not None:
            recorder = LabelRecorder(frame_idx, frame_idx / float(task.fps))
            renderer.label_recorder = recorder

        if frame_types[frame_idx] == "video_fullscreen":
            # 全屏视频播放帧：黑底 + 模拟视频内容 + 进度条 + 返回按钮
            frame = renderer.render_video_fullscreen_frame(frame_idx, task.style)
        else:
            scroll_y = scroll_seq[frame_idx]
            frame = renderer.render_chat_frame(messages, scroll_y, task.style)
        
        if recorder is not None:
            # 只保留"内容区"内可见的行: 标题栏/导航栏会遮住其覆盖区域的行,
            # 这些行 OCR 看不到, 不应进入标签(否则会污染 precision/recall)。
            th = task.style.top_offset
            nh = task.style.nav_height
            H = task.resolution[1]
            for r in recorder.records:
                cy = r["bbox"][1] + r["bbox"][3] / 2.0
                if th <= cy <= (H - nh):
                    all_labels.append(r)

        # frame = renderer.apply_device_overlay(frame, task.device)
        # frame = renderer.apply_quality_effects(frame, task.fps)
        writer.write(frame)

    writer.release()
    renderer.label_recorder = None

    if label_dir is not None:
        os.makedirs(label_dir, exist_ok=True)
        labels_path = os.path.join(label_dir, f"{task.video_id}.labels.jsonl")
        with open(labels_path, "w", encoding="utf-8") as f:
            for r in all_labels:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta = {
            "video_id": task.video_id,
            "fps": task.fps,
            "resolution": list(task.resolution),
            "duration_sec": task.duration_sec,
            "behavior_profile": task.behavior_profile,
            "case_id": task.case_id,
            "style_title": task.style.app_type.value,
            "title_height": task.style.title_height,
            "nav_height": task.style.nav_height,
            "frame_count": len(scroll_seq),
            "scroll_seq": scroll_seq,
            "frame_types": frame_types,
            "messages": [
                {"idx": i, "sender_id": m.sender_id, "is_received": m.is_received,
                 "msg_type": m.msg_type, "text": m.text,
                 "voice_transcribed": m.voice_transcribed}
                for i, m in enumerate(messages)
            ],
        }
        meta_path = os.path.join(label_dir, f"{task.video_id}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_path


# ==============================================================================
# PART 7: 断点续连与正交采样 / Checkpoint & Orthogonal Sampling
# ==============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file 
        # ----------------------
        self.data = {"completed": []}
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass
        self.file = checkpoint_file
        self.completed_ids = self._load()

    def _load(self) -> set:
        if os.path.exists(self.file):
            with open(self.file, 'r') as f:
                return set(json.load(f))
        return set()

    def save(self):
        """安全地保存 checkpoint，带重试机制"""
        temp_path = self.checkpoint_file + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            # 核心修改：加入重试机制应对 Windows 的文件锁冲突
            max_retries = 5
            for i in range(max_retries):
                try:
                    os.replace(temp_path, self.checkpoint_file)
                    return  # 替换成功，直接退出方法
                except PermissionError as e:
                    if i < max_retries - 1:
                        time.sleep(0.1 * (i + 1))  # 递增等待时间：0.1s, 0.2s, 0.3s...
                    else:
                        print(f"  [WARN] 无法更新 checkpoint 文件，已重试 {max_retries} 次: {e}")
        except Exception as e:
            print(f"  [WARN] 保存 checkpoint 失败: {e}")

    def mark_complete(self, video_id: str):
        self.completed_ids.add(video_id)
        # 原子写入防止中断损坏
        tmp = self.file + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(list(self.completed_ids), f)
        self.save()

    def is_done(self, video_id: str) -> bool:
        return video_id in self.completed_ids



class OrthogonalSampler:
    @staticmethod
    def generate_tasks(styles: List[AppStyle], devices: List[DeviceType], 
                       fps_options: List[int], profiles: List[str], 
                       durations: List[int], case_ids: List[str],
                       sample_size: int = 500) -> List[VideoTask]:
        tasks = []
        for i in range(sample_size):
            tasks.append(VideoTask(
                video_id=f"ortho_{i:04d}",
                style=random.choice(styles),
                device=random.choice(devices),
                fps=random.choice(fps_options),
                duration_sec=random.choice(durations),
                behavior_profile=random.choice(profiles),
                case_id=random.choice(case_ids) if case_ids else ""
            ))
        return tasks


# ==============================================================================
# PART 8: 主入口 / Main Entry
# ==============================================================================

def render_worker(task: VideoTask, cases: Dict[str, List[ChatMessage]], output_dir: str, checkpoint_file: str):
    """子进程工作函数 / Subprocess worker function"""
    checkpoint = CheckpointManager(checkpoint_file)
    if checkpoint.is_done(task.video_id):
        return f"[SKIP] {task.video_id}"
    
    # 根据任务分配的 case_id 选择对应案件的消息 / Select messages by task's case_id
    messages = cases.get(task.case_id, [])
    if not messages:
        return f"[FAIL] {task.video_id}: 案件 {task.case_id} 无消息数据"

    asset_loader = IndexedAssetLoader()
    asset_loader.add_avatar_pool(r"D:\手机录屏内容提取\tests\avatars")
    renderer = FrameRenderer(task.resolution[0], task.resolution[1], asset_loader=asset_loader)

    try:
        path = render_single_video(task, messages, output_dir, renderer)
        checkpoint.mark_complete(task.video_id)
        return f"[DONE] {task.video_id} -> {os.path.basename(path)} (案件: {task.case_id}, 消息: {len(messages)}条)"
    except Exception as e:
        return f"[FAIL] {task.video_id}: {str(e)}"


def main():
    print("=" * 70)
    print("工业级全覆盖测试视频生成器 V2.0")
    print("=" * 70)

    DATA_DIR = r"D:\AntiFraudProFiles\data\appdata"
    OUTPUT_DIR = r"d:\手机录屏内容提取\full_coverage_test_v2"
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint = CheckpointManager(CHECKPOINT_FILE)
    print(f"[INIT] 断点续连已启用，已完成: {len(checkpoint.completed_ids)} 个")

    print("[LOAD] 加载案件数据（按案件分组）...")
    cases = load_cases(DATA_DIR, min_per_case=50)
    case_ids = list(cases.keys())
    print(f"[DEBUG] 案件数: {len(cases)}, 总消息数: {sum(len(v) for v in cases.values())}")

    print("[PLAN] 生成正交组合任务矩阵...")
    styles = generate_all_styles()
    tasks = OrthogonalSampler.generate_tasks(
        styles=styles,
        devices=list(DeviceType),
        fps_options=[30, 60],
        profiles=list(BehaviorEngine.WEIGHT_PROFILES.keys()),
        durations=[30, 60, 180, 300, 600, 1800, 240, 383],
        case_ids=case_ids,
        sample_size=10
    )
    
    pending_tasks = [t for t in tasks if not checkpoint.is_done(t.video_id)]
    print(f"[PLAN] 总任务: {len(tasks)}, 待执行: {len(pending_tasks)}")

    if not pending_tasks:
        print("[DONE] 所有任务已完成！")
        return

    cpu_count = mp.cpu_count()
    print(f"[EXEC] 启动 {cpu_count} 个进程并发渲染...")
    
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {
            executor.submit(render_worker, task, cases, OUTPUT_DIR, CHECKPOINT_FILE): task 
            for task in pending_tasks
        }
        for future in as_completed(futures):
            print(future.result())

    print("[FINISH] 全部生成完毕！")


if __name__ == "__main__":
    mp.freeze_support()
    main()
