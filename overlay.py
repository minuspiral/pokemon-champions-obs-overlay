"""
OBS配信用 Pokemon Champions 相手ポケモンアイコン切り出しオーバーレイ

OBS WebSocket経由で映像キャプチャソースのスクリーンショットを取得し、
選出画面を検出→相手6体のスプライトをROI切り出し→横一列PNGとして出力。
OBSの「画像ソース」で読み取れば配信画面にオーバーレイ表示できる。

依存: opencv-python, numpy, obsws-python
"""

import cv2
import numpy as np
import os
import sys
import time
import base64
import threading
import logging
from pathlib import Path

log = logging.getLogger("overlay")

# ─────────────────── リソースパス ───────────────────
# Nuitka standalone: テンプレは exe と同じフォルダ
# PyInstaller onefile: sys._MEIPASS に展開
# PyInstaller onedir: exe と同じ or _internal サブフォルダ
if "__compiled__" in globals():
    BASE_DIR = Path(sys.executable).parent
    EXE_DIR = BASE_DIR
elif getattr(sys, "frozen", False):
    _mei = getattr(sys, "_" + "MEIPASS", None)
    if _mei:
        BASE_DIR = Path(_mei)
    else:
        _exe_parent = Path(sys.executable).parent
        _internal = _exe_parent / "_internal"
        BASE_DIR = _internal if (_internal / "templates").exists() else _exe_parent
    EXE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
    EXE_DIR = BASE_DIR

TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = EXE_DIR / "output"

# ─────────────────── 定数 ───────────────────
# パネル共通 Y座標 (prep layout, 1920x1080 基準)
PANEL_Y_FIRST = 0.1510
PANEL_Y_STEP = 0.1167
PANEL_Y_H = 0.0950

# 相手スプライト ROI
OPP_X_START = 0.7240
OPP_X_END = 0.7864

# 自分スプライト ROI
MY_X_START = 0.160
MY_X_END = 0.250

# 選出判定: パネル背景色の領域 (対戦準備中画面)
MY_BG_X_START = 0.15
MY_BG_X_END = 0.20

# アイテムアイコン ROI (選出前画面, pokemonselection)
# パネル上端: 0.1333, ステップ: 0.1167 (対戦準備中と同じ)
ITEM_PANEL_Y_FIRST = 0.1333
ITEM_PANEL_Y_STEP = 0.1167
ITEM_Y_OFFSET = 0.083     # パネル上端からアイテムアイコン中心までのオフセット
ITEM_X_START = 0.070
ITEM_X_END = 0.098

# 番号テンプレート ROI (対戦準備中画面, 選出番号 1/2/3)
NUM_X_START = 0.152
NUM_X_END = 0.188
NUM_Y_OFFSET = 0.02
NUM_Y_H = 0.06

# リザルト画面 ROI (対戦終了後の continue_screen)
RESULT_RANK_ROI = (0.610, 0.420, 0.720, 0.465)  # 順位バナー (集計中/XX位)
RESULT_RATE_ROI = (0.745, 0.420, 0.875, 0.465)  # レートバナー

# タイプアイコン ROI (prep layout, スプライトの右隣)
TYPE_Y_OFFSET = 0.0105
TYPE_Y_H = 0.0380
TYPE_LEFT_X0 = 0.7935
TYPE_LEFT_X1 = 0.8100
TYPE_RIGHT_X0 = 0.8190
TYPE_RIGHT_X1 = 0.8390

# 画面検出テンプレート
SCREEN_TEMPLATES = [
    ("team_preview", "team_preview_header.png", (0.0, 0.08, 0.25, 0.75), 0.65),
    ("continue_screen", "continue_button.png", (0.85, 1.0, 0.7, 1.0), 0.65),
]

ICON_SIZE = 128   # スプライトのリサイズサイズ
TYPE_SIZE = 24    # タイプアイコンのリサイズサイズ
SEP_WIDTH = 2     # パネル間セパレータ幅


# ─────────────────── 画面検出 ───────────────────
class ScreenDetector:
    def __init__(self):
        self.templates = []

    def load(self, template_dir: str) -> int:
        d = Path(template_dir)
        if not d.is_dir():
            return 0
        for key, fname, roi, threshold in SCREEN_TEMPLATES:
            path = d / fname
            if not path.exists():
                continue
            tmpl = cv2.imread(str(path))
            if tmpl is not None:
                self.templates.append((key, tmpl, roi, threshold))
        return len(self.templates)

    def detect(self, frame) -> tuple:
        if frame is None or not self.templates:
            return None, 0.0
        h, w = frame.shape[:2]
        best_key, best_score = None, 0.0
        for key, tmpl, (ry0, ry1, rx0, rx1), threshold in self.templates:
            y0, y1 = int(h * ry0), int(h * ry1)
            x0, x1 = int(w * rx0), int(w * rx1)
            roi = frame[y0:y1, x0:x1]
            if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
                continue
            result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score >= threshold and score > best_score:
                best_key, best_score = key, score
        return best_key, best_score


# ─────────────────── スプライト切り出し ───────────────────
def _trim_red_bg(roi_bgr):
    """赤パネル背景をトリム (上下左右の赤い帯を除去してスプライトのみにする)"""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 40, 40), (15, 255, 255)),
        cv2.inRange(hsv, (160, 40, 40), (180, 255, 255)),
    )
    rh, rw = red.shape
    top, bot, left, right = 0, rh, 0, rw
    for row in range(rh):
        if cv2.countNonZero(red[row:row+1]) / rw < 0.85:
            top = row
            break
    for row in range(rh - 1, -1, -1):
        if cv2.countNonZero(red[row:row+1]) / rw < 0.85:
            bot = row + 1
            break
    for col in range(rw):
        if cv2.countNonZero(red[:, col:col+1]) / rh < 0.85:
            left = col
            break
    for col in range(rw - 1, -1, -1):
        if cv2.countNonZero(red[:, col:col+1]) / rh < 0.85:
            right = col + 1
            break
    if bot > top and right > left:
        return roi_bgr[top:bot, left:right]
    return roi_bgr


def extract_opponent_strip(frame, icon_size=ICON_SIZE, type_size=TYPE_SIZE):
    """選出画面から相手6体のスプライトを切り出し、横一列に連結した画像を返す。
    各スプライトの右下にタイプアイコン2個を重ねて描画する。

    Returns: numpy array (icon_size x (icon_size*6 + sep*5), 3ch BGR) or None
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    icons = []
    for i in range(6):
        # スプライト切り出し
        y0 = max(0, int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i)))
        y1 = min(h, int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i + PANEL_Y_H)))
        x0 = max(0, int(w * OPP_X_START))
        x1 = min(w, int(w * OPP_X_END))
        if y1 <= y0 or x1 <= x0:
            return None
        roi = frame[y0:y1, x0:x1]
        resized = cv2.resize(roi, (icon_size, icon_size), interpolation=cv2.INTER_CUBIC)

        # タイプアイコン切り出し → スプライト右下に重ねる
        ty0 = int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i + TYPE_Y_OFFSET))
        ty1 = ty0 + int(h * TYPE_Y_H)
        if ty1 <= h:
            lx0, lx1 = int(w * TYPE_LEFT_X0), int(w * TYPE_LEFT_X1)
            rx0, rx1 = int(w * TYPE_RIGHT_X0), int(w * TYPE_RIGHT_X1)
            if rx1 <= w:
                # 各タイプを正方形 (type_size x type_size) にリサイズ (歪みなし)
                type_l = cv2.resize(frame[ty0:ty1, lx0:lx1],
                                     (type_size, type_size), interpolation=cv2.INTER_AREA)
                type_r = cv2.resize(frame[ty0:ty1, rx0:rx1],
                                     (type_size, type_size), interpolation=cv2.INTER_AREA)
                # 右下端に2個横並びで配置
                bx = icon_size - type_size * 2
                by = icon_size - type_size
                resized[by:by + type_size, bx:bx + type_size] = type_l
                resized[by:by + type_size, bx + type_size:bx + type_size * 2] = type_r

        icons.append(resized)

    # セパレータ(白2px)を挟んで横連結
    parts = []
    for i, icon in enumerate(icons):
        parts.append(icon)
        if i < 5:
            sep = np.ones((icon_size, SEP_WIDTH, 3), dtype=np.uint8) * 255
            parts.append(sep)
    return np.hstack(parts)


def extract_opponent_strip_vertical(frame, icon_size=ICON_SIZE, type_size=TYPE_SIZE):
    """相手6体をタテ一列に連結した画像を返す (左右反転して配信画面の左側配置向け)。"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    icons = []
    for i in range(6):
        y0 = max(0, int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i)))
        y1 = min(h, int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i + PANEL_Y_H)))
        x0 = max(0, int(w * OPP_X_START))
        x1 = min(w, int(w * OPP_X_END))
        if y1 <= y0 or x1 <= x0:
            return None
        roi = frame[y0:y1, x0:x1]
        resized = cv2.resize(roi, (icon_size, icon_size), interpolation=cv2.INTER_CUBIC)

        ty0 = int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i + TYPE_Y_OFFSET))
        ty1 = ty0 + int(h * TYPE_Y_H)
        if ty1 <= h:
            lx0, lx1 = int(w * TYPE_LEFT_X0), int(w * TYPE_LEFT_X1)
            rx0, rx1 = int(w * TYPE_RIGHT_X0), int(w * TYPE_RIGHT_X1)
            if rx1 <= w:
                type_l = cv2.resize(frame[ty0:ty1, lx0:lx1],
                                     (type_size, type_size), interpolation=cv2.INTER_AREA)
                type_r = cv2.resize(frame[ty0:ty1, rx0:rx1],
                                     (type_size, type_size), interpolation=cv2.INTER_AREA)
                bx = icon_size - type_size * 2
                by = icon_size - type_size
                resized[by:by + type_size, bx:bx + type_size] = type_l
                resized[by:by + type_size, bx + type_size:bx + type_size * 2] = type_r

        icons.append(resized)

    parts = []
    for i, icon in enumerate(icons):
        parts.append(icon)
        if i < 5:
            sep = np.ones((SEP_WIDTH, icon_size, 3), dtype=np.uint8) * 255
            parts.append(sep)
    return np.vstack(parts)


def extract_item_icons(frame):
    """選出前画面から6体分のアイテムアイコンを切り出す。

    Returns: list of 6 BGR images (正方形) or Noneのリスト
    """
    if frame is None:
        return [None] * 6
    h, w = frame.shape[:2]
    x0 = int(w * ITEM_X_START)
    x1 = int(w * ITEM_X_END)
    x_pixels = x1 - x0
    y_half = x_pixels // 2  # 正方形にするためXの幅に合わせる

    items = []
    for i in range(6):
        center_y = int(h * (ITEM_PANEL_Y_FIRST + ITEM_PANEL_Y_STEP * i + ITEM_Y_OFFSET))
        y0 = center_y - y_half
        y1 = center_y + y_half
        if y0 < 0 or y1 > h:
            items.append(None)
            continue
        items.append(frame[y0:y1, x0:x1].copy())
    return items


def count_selected_panels(frame):
    """自分パネルの選出済み数をカウント (背景色 H<60, V>180 = 選出済み)。

    Returns: 選出済みパネル数 (0-6)
    """
    if frame is None:
        return 0
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    count = 0
    for i in range(6):
        y0 = int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i))
        y1 = int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i + PANEL_Y_H))
        bg_x0 = int(w * MY_BG_X_START)
        bg_x1 = int(w * MY_BG_X_END)
        bg = hsv[y0:y1, bg_x0:bg_x1]
        if bg[:, :, 0].mean() < 60 and bg[:, :, 2].mean() >= 180:
            count += 1
    return count


def detect_selection_order(frame):
    """選出済みパネルの番号(1/2/3)をテンプレートマッチングで判定。

    Returns: list of (slot_index, order_number, score) 番号順ソート済み
    """
    if frame is None:
        return []
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 番号テンプレート読み込み
    num_templates = {}
    for n in [1, 2, 3]:
        path = TEMPLATES_DIR / f"num_{n}.png"
        if path.exists():
            num_templates[n] = cv2.imread(str(path))

    results = []
    for i in range(6):
        y0 = int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i))
        y1 = int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i + PANEL_Y_H))

        # 選出判定
        bg_x0 = int(w * MY_BG_X_START)
        bg_x1 = int(w * MY_BG_X_END)
        bg = hsv[y0:y1, bg_x0:bg_x1]
        if bg[:, :, 0].mean() >= 60 or bg[:, :, 2].mean() < 180:
            continue  # 未選出

        if not num_templates:
            # テンプレートなし → パネル順のまま
            results.append((i, len(results) + 1, 0.0))
            continue

        # 番号領域切り出し
        ny0 = int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i + NUM_Y_OFFSET))
        ny1 = int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * i + NUM_Y_OFFSET + NUM_Y_H))
        nx0 = int(w * NUM_X_START)
        nx1 = int(w * NUM_X_END)
        num_roi = frame[ny0:ny1, nx0:nx1]

        # テンプレートマッチングで番号判定
        best_num, best_score = 0, -1.0
        for n, tmpl in num_templates.items():
            # テンプレートをROIサイズに合わせてリサイズ
            tmpl_r = cv2.resize(tmpl, (num_roi.shape[1], num_roi.shape[0]))
            result = cv2.matchTemplate(num_roi, tmpl_r, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score > best_score:
                best_score = score
                best_num = n
        results.append((i, best_num, best_score))

    # 番号順にソート
    results.sort(key=lambda x: x[1])
    return results


def _build_my_selection_icons(frame, icon_size=ICON_SIZE, item_icons=None):
    """自分の選出済みポケモンのアイコンリストを返す (番号順ソート済み)。"""
    if frame is None:
        return []
    h, w = frame.shape[:2]
    ordered = detect_selection_order(frame)
    if not ordered:
        return []

    icons = []
    for slot_idx, order_num, score in ordered:
        y0 = max(0, int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * slot_idx)))
        y1 = min(h, int(h * (PANEL_Y_FIRST + PANEL_Y_STEP * slot_idx + PANEL_Y_H)))
        x0 = max(0, int(w * MY_X_START))
        x1 = min(w, int(w * MY_X_END))
        if y1 <= y0 or x1 <= x0:
            continue
        roi = frame[y0:y1, x0:x1]
        resized = cv2.resize(roi, (icon_size, icon_size), interpolation=cv2.INTER_CUBIC)

        # アイテムアイコンを左下に大きめに重ねる
        if item_icons and slot_idx < len(item_icons) and item_icons[slot_idx] is not None:
            item_sz = icon_size // 3
            item = cv2.resize(item_icons[slot_idx], (item_sz, item_sz),
                              interpolation=cv2.INTER_AREA)
            by = icon_size - item_sz
            resized[by:by + item_sz, 0:item_sz] = item

        icons.append(resized)
    return icons


def extract_my_selection_strip(frame, icon_size=ICON_SIZE, item_icons=None):
    """自分選出をタテ一列に連結。"""
    icons = _build_my_selection_icons(frame, icon_size, item_icons)
    if not icons:
        return None
    parts = []
    for i, icon in enumerate(icons):
        parts.append(icon)
        if i < len(icons) - 1:
            parts.append(np.ones((SEP_WIDTH, icon_size, 3), dtype=np.uint8) * 255)
    return np.vstack(parts)


def extract_my_selection_strip_horizontal(frame, icon_size=ICON_SIZE, item_icons=None):
    """自分選出を横一列に連結。"""
    icons = _build_my_selection_icons(frame, icon_size, item_icons)
    if not icons:
        return None
    parts = []
    for i, icon in enumerate(icons):
        parts.append(icon)
        if i < len(icons) - 1:
            parts.append(np.ones((icon_size, SEP_WIDTH, 3), dtype=np.uint8) * 255)
    return np.hstack(parts)


def extract_result_regions(frame):
    """リザルト画面(continue_screen)から順位/レート を切り出す。

    Returns: dict {rank, rate} → 各 BGR image or None
    """
    if frame is None:
        return {}
    h, w = frame.shape[:2]
    out = {}
    for name, (x0, y0, x1, y1) in (
        ('rank', RESULT_RANK_ROI),
        ('rate', RESULT_RATE_ROI),
    ):
        px0, py0 = int(w * x0), int(h * y0)
        px1, py1 = int(w * x1), int(h * y1)
        if py1 > h or px1 > w or py0 >= py1 or px0 >= px1:
            out[name] = None
        else:
            out[name] = frame[py0:py1, px0:px1].copy()
    return out


# ─────────────────── OBS WebSocket ───────────────────
def obs_grab_frame(client, source_name, width=1920, height=1080):
    """OBSソースからスクリーンショット1枚取得→OpenCV frame"""
    resp = client.get_source_screenshot(
        name=source_name,
        img_format="jpg",
        width=width, height=height,
        quality=85,
    )
    img_b64 = resp.image_data
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    raw = base64.b64decode(img_b64)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


# ─────────────────── GUI ───────────────────
import tkinter as tk
from tkinter import ttk, scrolledtext


class OverlayApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OBS Pokemon Champions Overlay v1.4.0")
        self.root.geometry("780x580")
        self.root.resizable(True, True)

        self.running = False
        self.worker_thread = None
        self.detector = None

        self._build_ui()
        self.root.after(100, self._init_engine)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # === OBS接続 ===
        conn = ttk.LabelFrame(self.root, text="OBS接続", padding=8)
        conn.pack(fill="x", padx=8, pady=(8, 4))

        r0 = ttk.Frame(conn)
        r0.pack(fill="x")
        ttk.Label(r0, text="ホスト:").pack(side="left")
        self.host_var = tk.StringVar(value="localhost")
        ttk.Entry(r0, textvariable=self.host_var, width=15).pack(side="left", padx=4)
        ttk.Label(r0, text="ポート:").pack(side="left", padx=(8, 0))
        self.port_var = tk.StringVar(value="4455")
        ttk.Entry(r0, textvariable=self.port_var, width=6).pack(side="left", padx=4)
        ttk.Label(r0, text="パスワード:").pack(side="left", padx=(8, 0))
        self.pass_var = tk.StringVar(value="")
        ttk.Entry(r0, textvariable=self.pass_var, width=14, show="*").pack(side="left", padx=4)
        self.btn_connect = ttk.Button(r0, text="接続", command=self._connect_obs, width=8)
        self.btn_connect.pack(side="right", padx=4)

        r1 = ttk.Frame(conn)
        r1.pack(fill="x", pady=(4, 0))
        ttk.Label(r1, text="ソース:").pack(side="left")
        self.source_var = tk.StringVar()
        self.source_combo = ttk.Combobox(r1, textvariable=self.source_var,
                                          width=35, state="readonly")
        self.source_combo.pack(side="left", padx=4)
        self.conn_status = tk.StringVar(value="未接続")
        ttk.Label(r1, textvariable=self.conn_status,
                  foreground="gray").pack(side="right", padx=4)

        # === 設定 ===
        settings = ttk.LabelFrame(self.root, text="設定", padding=8)
        settings.pack(fill="x", padx=8, pady=4)

        sr = ttk.Frame(settings)
        sr.pack(fill="x")
        ttk.Label(sr, text="検出間隔:").pack(side="left")
        self.interval_var = tk.DoubleVar(value=1.5)
        ttk.Scale(sr, from_=0.5, to=5.0, variable=self.interval_var,
                  orient="horizontal", length=150).pack(side="left", padx=4)
        self.interval_label = ttk.Label(sr, text="1.5秒")
        self.interval_label.pack(side="left")
        self.interval_var.trace_add("write", self._update_interval_label)

        sr2 = ttk.Frame(settings)
        sr2.pack(fill="x", pady=(4, 0))
        ttk.Label(sr2, text="出力先:").pack(side="left")
        self.output_var = tk.StringVar(value=str(OUTPUT_DIR))
        ttk.Entry(sr2, textvariable=self.output_var, width=45).pack(side="left", padx=4)
        ttk.Button(sr2, text="参照...", width=6,
                   command=self._browse_output).pack(side="left")

        # === 操作 ===
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=8, pady=4)

        self.btn_start = ttk.Button(ctrl, text="▶ 開始", command=self._start,
                                     width=12, state="disabled")
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(ctrl, text="■ 停止", command=self._stop,
                                    width=12, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.status_var = tk.StringVar(value="初期化中...")
        ttk.Label(ctrl, textvariable=self.status_var,
                  font=("", 11, "bold")).pack(side="left", padx=12)

        # === プレビュー ===
        preview = ttk.LabelFrame(self.root, text="プレビュー", padding=4)
        preview.pack(fill="x", padx=8, pady=4)
        preview_inner = ttk.Frame(preview)
        preview_inner.pack()
        # 相手チーム (横一列)
        opp_frame = ttk.Frame(preview_inner)
        opp_frame.pack(side="left", padx=(0, 8))
        ttk.Label(opp_frame, text="相手(横)", font=("", 8)).pack()
        self.preview_label = ttk.Label(opp_frame, text="待機中...", anchor="center")
        self.preview_label.pack()
        # 相手チーム (タテ一列)
        opp_v_frame = ttk.Frame(preview_inner)
        opp_v_frame.pack(side="left", padx=(0, 8))
        ttk.Label(opp_v_frame, text="相手(縦)", font=("", 8)).pack()
        self.opp_v_preview_label = ttk.Label(opp_v_frame, text="待機中...", anchor="center")
        self.opp_v_preview_label.pack()
        # ランク+レート (リザルト画面で更新)
        rank_frame = ttk.Frame(preview_inner)
        rank_frame.pack(side="left", padx=(0, 8))
        ttk.Label(rank_frame, text="ランク", font=("", 8)).pack()
        self.rank_preview_label = ttk.Label(rank_frame, text="---", anchor="center")
        self.rank_preview_label.pack()
        ttk.Label(rank_frame, text="レート", font=("", 8)).pack(pady=(4, 0))
        self.rate_preview_label = ttk.Label(rank_frame, text="---", anchor="center")
        self.rate_preview_label.pack()
        # 自分選出 (タテ一列)
        my_frame = ttk.Frame(preview_inner)
        my_frame.pack(side="left", padx=(0, 8))
        ttk.Label(my_frame, text="自分(縦)", font=("", 8)).pack()
        self.my_preview_label = ttk.Label(my_frame, text="待機中...", anchor="center")
        self.my_preview_label.pack()
        # 自分選出 (横一列)
        my_h_frame = ttk.Frame(preview_inner)
        my_h_frame.pack(side="left")
        ttk.Label(my_h_frame, text="自分(横)", font=("", 8)).pack()
        self.my_h_preview_label = ttk.Label(my_h_frame, text="待機中...", anchor="center")
        self.my_h_preview_label.pack()

        # === ログ ===
        log_frame = ttk.LabelFrame(self.root, text="ログ", padding=4)
        log_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=8, font=("Consolas", 9),
            state="disabled", bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill="both", expand=True)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.after(0, self._insert_log, f"{ts} {msg}")

    def _insert_log(self, line):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        n = int(self.log_text.index("end-1c").split(".")[0])
        if n > 500:
            self.log_text.delete("1.0", f"{n - 400}.0")
        self.log_text.configure(state="disabled")

    def _update_interval_label(self, *_):
        self.interval_label.configure(text=f"{self.interval_var.get():.1f}秒")

    def _browse_output(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self.output_var.get(),
                                    title="出力フォルダを選択")
        if d:
            self.output_var.set(d)

    # --- 初期化 ---
    def _init_engine(self):
        self.detector = ScreenDetector()
        n = self.detector.load(str(TEMPLATES_DIR))
        self._log(f"テンプレート読込: {n}件")
        if n == 0:
            self.status_var.set("テンプレートなし")
            self._log("[!] templates/ にテンプレート画像を配置してください")
        else:
            self.status_var.set("待機中 (OBS未接続)")
            self._log("初期化完了")

    # --- OBS接続 ---
    def _connect_obs(self):
        host = self.host_var.get().strip()
        port = int(self.port_var.get().strip() or "4455")
        pw = self.pass_var.get()
        self._log(f"OBS接続中... {host}:{port}")
        try:
            import obsws_python as obsws
            cl = obsws.ReqClient(host=host, port=port, password=pw, timeout=5)
            resp = cl.get_input_list()
            names = [inp.get("inputName", "") for inp in resp.inputs if inp.get("inputName")]
            cl.disconnect()
            self.source_combo["values"] = names
            if names:
                self.source_var.set(names[0])
            self.conn_status.set(f"接続OK ({len(names)}ソース)")
            self._log(f"OBS接続成功: {len(names)}ソース")
            self.btn_start.configure(state="normal")
            self.status_var.set("準備完了")
        except Exception as e:
            self.conn_status.set("接続失敗")
            self._log(f"OBS接続失敗: {e}")
            self._log("OBS → ツール → WebSocketサーバー設定 で有効化してください")

    # --- 開始/停止 ---
    def _start(self):
        if self.running:
            return
        source = self.source_var.get()
        if not source:
            self._log("[!] ソースを選択してください")
            return
        conn_info = {
            "host": self.host_var.get().strip(),
            "port": int(self.port_var.get().strip() or "4455"),
            "password": self.pass_var.get(),
            "source": source,
            "interval": self.interval_var.get(),
            "output_dir": self.output_var.get().strip(),
        }
        self.running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_var.set("検出中...")
        self._log(f"開始: ソース={source}")
        self.worker_thread = threading.Thread(target=self._worker_loop,
                                               args=(conn_info,), daemon=True)
        self.worker_thread.start()

    def _stop(self):
        self.running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.status_var.set("停止")
        self._log("停止")

    def _worker_loop(self, conn_info: dict):
        import obsws_python as obsws
        try:
            client = obsws.ReqClient(
                host=conn_info["host"],
                port=conn_info["port"],
                password=conn_info["password"],
                timeout=10,
            )
            self._log("OBSワーカー接続OK")
        except Exception as e:
            self._log(f"OBS接続失敗: {e}")
            self.root.after(0, self._stop)
            return

        source_name = conn_info["source"]
        interval = conn_info["interval"]
        out_dir = Path(conn_info["output_dir"])
        img_out = out_dir / "opponent_team.png"
        img_out_v = out_dir / "opponent_team_vertical.png"
        my_img_out = out_dir / "my_selection.png"
        my_img_out_h = out_dir / "my_selection_horizontal.png"
        result_out_paths = {
            "rank": out_dir / "rank.png",
            "rate": out_dir / "rate.png",
        }
        last_hash = None
        last_hash_v = None
        last_my_hash = None
        last_my_hash_h = None
        last_result_hash = {k: None for k in result_out_paths}
        saved_item_icons = [None] * 6  # 選出前画面で保存したアイテムアイコン
        items_saved = False  # アイテム保存済みフラグ
        selection_locked = False  # 選出完了ロック (3体検出後は相手・自分とも固定)

        while self.running:
            try:
                frame = obs_grab_frame(client, source_name)
                if frame is None:
                    time.sleep(interval)
                    continue

                key, score = self.detector.detect(frame)

                if key == "team_preview":
                    # 選出前 vs 対戦準備中を判定
                    n_selected = count_selected_panels(frame)

                    if n_selected <= 1 and not items_saved:
                        # 選出前画面 → アイテムアイコンを保存 (1回のみ)
                        self.root.after(0, self.status_var.set, "選出前画面 (アイテム取得)")
                        items = extract_item_icons(frame)
                        if any(it is not None for it in items):
                            saved_item_icons = items
                            items_saved = True
                            self._log(f"アイテムアイコン保存: {sum(1 for it in items if it is not None)}体分")
                    elif n_selected < 3:
                        # 選出中 (まだ3体揃っていない)
                        self.root.after(0, self.status_var.set, f"選出中... ({n_selected}/3)")
                    else:
                        # 選出完了 (3体) → 相手+自分を切り出してロック
                        if not selection_locked:
                            self.root.after(0, self.status_var.set, "選出完了!")

                            # 相手チーム横一列
                            strip = extract_opponent_strip(frame)
                            if strip is not None:
                                img_out.parent.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(img_out), strip)
                                self._log(f"相手チーム更新(横): {strip.shape[1]}x{strip.shape[0]}")
                                self.root.after(0, self._update_preview, strip)

                            # 相手チーム縦一列
                            strip_v = extract_opponent_strip_vertical(frame)
                            if strip_v is not None:
                                img_out_v.parent.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(img_out_v), strip_v)
                                self.root.after(0, self._update_opp_v_preview, strip_v)

                            # 自分選出 縦一列
                            my_strip = extract_my_selection_strip(
                                frame, item_icons=saved_item_icons
                            )
                            if my_strip is not None:
                                my_img_out.parent.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(my_img_out), my_strip)
                                self._log(f"自分選出更新(縦): {my_strip.shape[1]}x{my_strip.shape[0]}")
                                self.root.after(0, self._update_my_preview, my_strip)

                            # 自分選出 横一列
                            my_strip_h = extract_my_selection_strip_horizontal(
                                frame, item_icons=saved_item_icons
                            )
                            if my_strip_h is not None:
                                my_img_out_h.parent.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(my_img_out_h), my_strip_h)
                                self.root.after(0, self._update_my_h_preview, my_strip_h)

                            # ロック
                            selection_locked = True
                            self._log("選出ロック (次の対戦までロック)")
                elif key == "continue_screen":
                    # リザルト画面 → 連勝数/ランク/VP を切り出し
                    self.root.after(0, self.status_var.set, "リザルト画面!")
                    regions = extract_result_regions(frame)
                    for name, img in regions.items():
                        if img is None:
                            continue
                        rh = hash(img.tobytes()[:1024])
                        if rh != last_result_hash[name]:
                            path = result_out_paths[name]
                            path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(path), img)
                            last_result_hash[name] = rh
                            self._log(f"リザルト更新 {name}: {img.shape[1]}x{img.shape[0]}")
                            self.root.after(0, self._update_result_preview, name, img)
                    # 次対戦に備えてロック解除
                    items_saved = False
                    selection_locked = False
                else:
                    # team_preview以外 → 次の対戦に備えてリセット
                    items_saved = False
                    selection_locked = False
                    s = f"{key}({score:.2f})" if key else "待機中"
                    self.root.after(0, self.status_var.set, s)

            except Exception as e:
                self._log(f"エラー: {e}")

            time.sleep(interval)

        try:
            client.disconnect()
        except Exception:
            pass

    def _update_preview(self, strip_bgr):
        """相手チーム切り出し結果をGUIにプレビュー表示"""
        try:
            from PIL import Image as PILImage, ImageTk
            rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            max_w = 460
            if pil.width > max_w:
                ratio = max_w / pil.width
                pil = pil.resize((max_w, int(pil.height * ratio)))
            tk_img = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=tk_img, text="")
            self.preview_label._tk_img = tk_img
        except ImportError:
            self.preview_label.configure(text="(Pillow未インストール)")

    def _update_opp_v_preview(self, strip_bgr):
        """相手チーム縦一列をGUIにプレビュー表示"""
        try:
            from PIL import Image as PILImage, ImageTk
            rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            max_h = 150
            if pil.height > max_h:
                ratio = max_h / pil.height
                pil = pil.resize((int(pil.width * ratio), max_h))
            tk_img = ImageTk.PhotoImage(pil)
            self.opp_v_preview_label.configure(image=tk_img, text="")
            self.opp_v_preview_label._tk_img = tk_img
        except ImportError:
            self.opp_v_preview_label.configure(text="(Pillow未インストール)")

    def _update_my_preview(self, strip_bgr):
        """自分選出切り出し結果をGUIにプレビュー表示"""
        try:
            from PIL import Image as PILImage, ImageTk
            rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            max_h = 150
            if pil.height > max_h:
                ratio = max_h / pil.height
                pil = pil.resize((int(pil.width * ratio), max_h))
            tk_img = ImageTk.PhotoImage(pil)
            self.my_preview_label.configure(image=tk_img, text="")
            self.my_preview_label._tk_img = tk_img
        except ImportError:
            self.my_preview_label.configure(text="(Pillow未インストール)")

    def _update_my_h_preview(self, strip_bgr):
        """自分選出横一列をGUIにプレビュー表示"""
        try:
            from PIL import Image as PILImage, ImageTk
            rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            max_w = 150
            if pil.width > max_w:
                ratio = max_w / pil.width
                pil = pil.resize((max_w, int(pil.height * ratio)))
            tk_img = ImageTk.PhotoImage(pil)
            self.my_h_preview_label.configure(image=tk_img, text="")
            self.my_h_preview_label._tk_img = tk_img
        except ImportError:
            self.my_h_preview_label.configure(text="(Pillow未インストール)")

    def _update_result_preview(self, name, img_bgr):
        """ランク/レートをGUIにプレビュー表示"""
        label = {"rank": self.rank_preview_label, "rate": self.rate_preview_label}.get(name)
        if label is None:
            return
        try:
            from PIL import Image as PILImage, ImageTk
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            max_w = 150
            if pil.width > max_w:
                ratio = max_w / pil.width
                pil = pil.resize((max_w, int(pil.height * ratio)))
            tk_img = ImageTk.PhotoImage(pil)
            label.configure(image=tk_img, text="")
            label._tk_img = tk_img
        except ImportError:
            label.configure(text="(Pillow未インストール)")

    def _on_close(self):
        self.running = False
        time.sleep(0.2)
        self.root.destroy()


# ─────────────────── メイン ───────────────────
def main():
    root = tk.Tk()
    style = ttk.Style()
    for theme in ["vista", "clam", "winnative"]:
        if theme in style.theme_names():
            style.theme_use(theme)
            break
    OverlayApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
