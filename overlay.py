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
import json
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
CONFIG_PATH = EXE_DIR / "config.json"

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
    # リザルト画面: カーソル位置で 緑/青 が切替わるため全パターンを登録 (どれかがマッチすれば検出)
    ("continue_screen", "continue_button.png", (0.85, 1.0, 0.6, 1.0), 0.65),       # 緑「対戦を続ける」(カーソル右)
    ("continue_screen", "continue_button_blue.png", (0.85, 1.0, 0.6, 1.0), 0.65),  # 青「対戦を続ける」(カーソル外)
    ("continue_screen", "stop_button.png", (0.85, 1.0, 0.0, 0.4), 0.65),           # 青「対戦をやめる」(カーソル外)
    ("continue_screen", "stop_button_green.png", (0.85, 1.0, 0.0, 0.4), 0.65),     # 緑「対戦をやめる」(カーソル左)
    ("continue_screen", "team_edit_button.png", (0.85, 1.0, 0.3, 0.7), 0.65),      # 青「チームを編成する」
    # 左半分 ROI で自分側の結果を判定 (右半分は相手側)
    ("win_banner", "win_banner.png", (0.5, 0.8, 0.0, 0.5), 0.65),
    ("lose_banner", "lose_banner.png", (0.5, 0.8, 0.0, 0.5), 0.65),
    # 引き分け: DRAW は画面中央
    ("draw_banner", "draw_banner.png", (0.5, 0.8, 0.3, 0.8), 0.65),
]

# WIN/LOSE 検出後のブースト設定 (リザルト遷移が速いので短間隔で追いかける)
BOOST_DURATION_SEC = 15.0
BOOST_INTERVAL_SEC = 0.3

ICON_SIZE = 128   # スプライトのリサイズサイズ
TYPE_SIZE = 24    # タイプアイコンのリサイズサイズ
SEP_WIDTH = 2     # パネル間セパレータ幅


# ─────────────────── UI テーマ (ライトモード) ───────────────────
THEME = {
    "BG": "#f8f9fa",          # メイン背景 (オフホワイト)
    "PANEL": "#ffffff",       # パネル背景
    "HEADER": "#eef0f2",      # セクションヘッダ
    "FG": "#1f1f1f",          # 通常テキスト
    "FG_DIM": "#6e6e6e",      # 補助テキスト
    "ACCENT": "#0067c0",      # アクセント (Win11 青)
    "ACCENT_HOVER": "#005a9e",
    "SUCCESS": "#107c10",     # 成功 (緑)
    "WARN": "#b89500",        # 警告 (黄)
    "DANGER": "#c42b1c",      # エラー (赤)
    "BOOST": "#d97706",       # ブースト (橙)
    "BTN_BG": "#fafafa",
    "BTN_HOVER": "#ededed",
    "BTN_ACTIVE": "#dadada",
    "BTN_DISABLED": "#f0f0f0",
    "ENTRY_BG": "#ffffff",
    "BORDER": "#d1d1d1",
}


def imwrite_unicode(path, img, ext: str = ".png") -> bool:
    """日本語パス対応の cv2.imwrite ラッパー。
    cv2.imwrite は Windows + 非ASCIIパスで失敗するため imencode→tofile 経由で書き出す。
    """
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


# ─────────────────── 画面検出 ───────────────────
class ScreenDetector:
    def __init__(self):
        self.templates = []

    def load(self, template_dir: str) -> tuple:
        """テンプレートを読み込む。
        Returns: (loaded_count, missing_files_with_reason, abs_dir)
        """
        self.templates = []
        d = Path(template_dir)
        try:
            abs_dir = str(d.resolve())
        except Exception as e:
            return 0, [(fn, f"パス解決失敗: {e}") for _, fn, _, _ in SCREEN_TEMPLATES], str(d)
        missing = []
        if not d.exists():
            return 0, [(fn, "ディレクトリが存在しない") for _, fn, _, _ in SCREEN_TEMPLATES], abs_dir
        if not d.is_dir():
            return 0, [(fn, "指定パスはディレクトリでない") for _, fn, _, _ in SCREEN_TEMPLATES], abs_dir
        for key, fname, roi, threshold in SCREEN_TEMPLATES:
            path = d / fname
            if not path.exists():
                missing.append((fname, "ファイルが存在しない"))
                continue
            try:
                size = path.stat().st_size
            except Exception as e:
                missing.append((fname, f"stat失敗: {e}"))
                continue
            if size == 0:
                missing.append((fname, "ファイルサイズ 0 (OneDrive未ダウンロード等の可能性)"))
                continue
            try:
                # cv2.imread は Windows で日本語パス未対応 → bytes経由でデコード
                raw = np.fromfile(str(path), dtype=np.uint8)
                tmpl = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            except Exception as e:
                missing.append((fname, f"読み込み例外: {e}"))
                continue
            if tmpl is None:
                missing.append((fname, f"PNGデコード失敗 (ファイル破損の可能性, size={size}B)"))
                continue
            self.templates.append((key, tmpl, roi, threshold))
        return len(self.templates), missing, abs_dir

    def detect(self, frame) -> tuple:
        if frame is None or not self.templates:
            return None, 0.0, {}
        h, w = frame.shape[:2]
        best_key, best_score = None, 0.0
        all_scores = {}
        for key, tmpl, (ry0, ry1, rx0, rx1), threshold in self.templates:
            y0, y1 = int(h * ry0), int(h * ry1)
            x0, x1 = int(w * rx0), int(w * rx1)
            roi = frame[y0:y1, x0:x1]
            if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
                all_scores[key] = 0.0
                continue
            result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            all_scores[key] = score
            if score >= threshold and score > best_score:
                best_key, best_score = key, score
        return best_key, best_score, all_scores


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

    # 番号テンプレート読み込み (日本語パス対応のため np.fromfile + cv2.imdecode)
    num_templates = {}
    for n in [1, 2, 3]:
        path = TEMPLATES_DIR / f"num_{n}.png"
        if not path.exists():
            continue
        try:
            raw = np.fromfile(str(path), dtype=np.uint8)
            tmpl = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            tmpl = None
        if tmpl is not None and tmpl.size > 0:
            num_templates[n] = tmpl

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


def load_digit_templates(templates_dir) -> dict:
    """templates/digits/ 配下の 0.png〜9.png を読込 (日本語パス対応)。
    Returns: {digit_str: bgr_image}
    """
    digits_dir = Path(templates_dir) / "digits"
    if not digits_dir.is_dir():
        return {}
    out = {}
    for d in "0123456789":
        p = digits_dir / f"{d}.png"
        if not p.exists():
            continue
        try:
            raw = np.fromfile(str(p), dtype=np.uint8)
            tmpl = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            tmpl = None
        if tmpl is not None and tmpl.size > 0:
            out[d] = tmpl
    return out


def digit_ocr(bgr_crop, digit_templates: dict, target_h: int = 24,
              score_threshold: float = 0.6) -> str:
    """白文字数字を切り出してテンプレートマッチで認識。
    小さい blob はドット (.) として X 位置に挿入。kana 等ノイズはスキップ。
    """
    if bgr_crop is None or not digit_templates:
        return ""
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    digit_boxes = []
    dot_xs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # ドット候補: 小さい+低い+狭い
        if 2 <= area <= 30 and h <= 8 and w <= 8:
            dot_xs.append(x + w // 2)
            continue
        # 数字候補: 通常サイズ
        if area < 20 or h < 12 or w < 2 or w > 35:
            continue
        digit_boxes.append((x, y, w, h))
    digit_boxes.sort(key=lambda b: b[0])
    if not digit_boxes:
        return ""
    # テンプレートを高さ揃えて正規化
    norm_templates = {}
    for d, tmpl in digit_templates.items():
        th, tw = tmpl.shape[:2]
        new_w = max(1, int(tw * target_h / th))
        norm_templates[d] = cv2.resize(tmpl, (new_w, target_h))
    # 各 blob を OCR
    digit_results = []  # (x_center, char)
    for x, y, w, h in digit_boxes:
        pad = 2
        y0 = max(0, y - pad); y1 = min(bgr_crop.shape[0], y + h + pad)
        x0 = max(0, x - pad); x1 = min(bgr_crop.shape[1], x + w + pad)
        blob = bgr_crop[y0:y1, x0:x1]
        bh, bw = blob.shape[:2]
        new_bw = max(1, int(bw * target_h / bh))
        blob_norm = cv2.resize(blob, (new_bw, target_h))
        best_d = None
        best_score = score_threshold
        for d, tnorm in norm_templates.items():
            try:
                if blob_norm.shape[1] >= tnorm.shape[1]:
                    r = cv2.matchTemplate(blob_norm, tnorm, cv2.TM_CCOEFF_NORMED)
                else:
                    r = cv2.matchTemplate(tnorm, blob_norm, cv2.TM_CCOEFF_NORMED)
                score = float(r.max())
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_d = d
        if best_d is not None:
            digit_results.append((x + w // 2, best_d))
    # ドットを X 位置でマージ (数字の間に挟まるドットだけ採用)
    if not digit_results:
        return ""
    min_x = digit_results[0][0]
    max_x = digit_results[-1][0]
    merged = list(digit_results)
    for dx in dot_xs:
        if min_x < dx < max_x:
            merged.append((dx, "."))
    merged.sort(key=lambda b: b[0])
    return "".join(c for _, c in merged)


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
    """OBSソースからスクリーンショット1枚取得→OpenCV frame。失敗時None。

    例外が発生しても None を返してワーカーループを継続可能にする。
    """
    try:
        resp = client.get_source_screenshot(
            name=source_name,
            img_format="jpg",
            width=width, height=height,
            quality=85,
        )
    except Exception:
        return None
    img_b64 = getattr(resp, "image_data", None)
    if not img_b64:
        return None
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(img_b64)
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


# ─────────────────── GUI ───────────────────
import tkinter as tk
from tkinter import ttk, scrolledtext


def _setup_theme(root: tk.Tk):
    """ライトテーマを ttk.Style に適用"""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")  # 最もカスタマイズしやすい
    except Exception:
        pass
    T = THEME
    root.configure(bg=T["BG"])
    style.configure("TFrame", background=T["BG"])
    style.configure("Panel.TFrame", background=T["PANEL"])
    style.configure("Header.TFrame", background=T["HEADER"])
    style.configure("TLabel", background=T["BG"], foreground=T["FG"])
    style.configure("Panel.TLabel", background=T["PANEL"], foreground=T["FG"])
    style.configure("Header.TLabel", background=T["HEADER"], foreground=T["FG"], font=("Segoe UI", 9, "bold"))
    style.configure("Dim.TLabel", background=T["BG"], foreground=T["FG_DIM"])
    style.configure("Score.TLabel", background=T["BG"], foreground=T["FG"],
                    font=("Segoe UI", 22, "bold"))
    style.configure("Status.TLabel", background=T["HEADER"], foreground=T["FG"], font=("Segoe UI", 9))
    style.configure("TButton",
                    background=T["BTN_BG"], foreground=T["FG"],
                    bordercolor=T["BORDER"], focusthickness=0,
                    relief="flat", padding=(10, 4))
    style.map("TButton",
              background=[("active", T["BTN_HOVER"]), ("pressed", T["BTN_ACTIVE"]),
                          ("disabled", T["BTN_DISABLED"])],
              foreground=[("disabled", T["FG_DIM"])])
    style.configure("Accent.TButton", background=T["ACCENT"], foreground="white",
                    bordercolor=T["ACCENT"], padding=(12, 4))
    style.map("Accent.TButton",
              background=[("active", T["ACCENT_HOVER"]), ("disabled", T["BTN_DISABLED"])],
              foreground=[("disabled", T["FG_DIM"])])
    style.configure("TEntry",
                    fieldbackground=T["ENTRY_BG"], foreground=T["FG"],
                    bordercolor=T["BORDER"], insertcolor=T["FG"], padding=2)
    style.configure("TCombobox",
                    fieldbackground=T["ENTRY_BG"], foreground=T["FG"],
                    background=T["BTN_BG"], bordercolor=T["BORDER"],
                    arrowcolor=T["FG"], padding=2)
    style.map("TCombobox",
              fieldbackground=[("readonly", T["ENTRY_BG"])],
              foreground=[("readonly", T["FG"])])
    style.configure("TCheckbutton", background=T["BG"], foreground=T["FG"], focuscolor="")
    style.map("TCheckbutton", background=[("active", T["BG"])])
    style.configure("Horizontal.TScale", background=T["BG"], troughcolor=T["BTN_BG"])


class StatusLED(tk.Canvas):
    """接続/動作状態を色で示す円形インジケータ"""
    COLORS = {
        "off":          THEME["FG_DIM"],
        "disconnected": THEME["DANGER"],
        "connecting":   THEME["WARN"],
        "ready":        THEME["SUCCESS"],
        "running":      THEME["ACCENT"],
        "boost":        THEME["BOOST"],
    }

    def __init__(self, parent, size=14, **kw):
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, bg=THEME["BG"], **kw)
        self._oval = self.create_oval(2, 2, size-2, size-2,
                                       fill=self.COLORS["off"], outline="")
    def set_state(self, state: str):
        self.itemconfigure(self._oval, fill=self.COLORS.get(state, self.COLORS["off"]))


class CollapsibleSection(tk.Frame):
    """クリックで折りたたみ可能なセクション (ライトテーマ用)"""
    def __init__(self, parent, title: str, expanded: bool = True):
        super().__init__(parent, bg=THEME["BG"])
        self._expanded = expanded
        # ヘッダ (クリック可能)
        self._header = tk.Frame(self, bg=THEME["HEADER"], cursor="hand2")
        self._header.pack(fill="x")
        self._chevron = tk.Label(self._header, text="▾" if expanded else "▸",
                                  bg=THEME["HEADER"], fg=THEME["FG_DIM"],
                                  font=("Segoe UI", 9), cursor="hand2")
        self._chevron.pack(side="left", padx=(8, 4), pady=4)
        self._title_lbl = tk.Label(self._header, text=title,
                                    bg=THEME["HEADER"], fg=THEME["FG"],
                                    font=("Segoe UI", 9, "bold"), cursor="hand2")
        self._title_lbl.pack(side="left", pady=4)
        for w in (self._header, self._chevron, self._title_lbl):
            w.bind("<Button-1>", self._toggle)
        # ボディ
        self.body = tk.Frame(self, bg=THEME["PANEL"], padx=8, pady=6)
        if expanded:
            self.body.pack(fill="both", expand=True)

    def _toggle(self, _=None):
        self._expanded = not self._expanded
        self._chevron.configure(text="▾" if self._expanded else "▸")
        if self._expanded:
            self.body.pack(fill="both", expand=True)
        else:
            self.body.pack_forget()


class OverlayApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OBS Pokemon Champions Overlay v1.5.4")
        self.root.geometry("1024x720")  # プレビュー全表示の余裕を確保
        self.root.minsize(900, 600)
        self.root.resizable(True, True)
        # ウィンドウ/タスクバーアイコン (Tk feather → app_icon.ico)
        for icon_path in (BASE_DIR / "app_icon.ico", EXE_DIR / "app_icon.ico"):
            if icon_path.exists():
                try:
                    self.root.iconbitmap(default=str(icon_path))
                except Exception:
                    try:
                        self.root.iconbitmap(str(icon_path))
                    except Exception:
                        pass
                break
        _setup_theme(root)

        self.running = False
        self.worker_thread = None
        self.detector = None
        self._last_frame = None
        self._last_frame_stats = None
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._config = self._load_config()

        self._build_ui()
        self.root.after(100, self._init_engine)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        T = THEME
        # ─── ヘッダ: LED + 接続情報 ───
        header = tk.Frame(self.root, bg=T["BG"])
        header.pack(fill="x", padx=10, pady=(10, 4))

        # 1段目: LED + ホスト/ポート/PW + 接続ボタン
        h1 = tk.Frame(header, bg=T["BG"])
        h1.pack(fill="x")
        self.led = StatusLED(h1, size=14)
        self.led.pack(side="left", padx=(0, 6), pady=2)
        ttk.Label(h1, text="ホスト").pack(side="left")
        self.host_var = tk.StringVar(value=self._config.get("host", "localhost"))
        ttk.Entry(h1, textvariable=self.host_var, width=12).pack(side="left", padx=(4, 8))
        ttk.Label(h1, text=":").pack(side="left")
        self.port_var = tk.StringVar(value=str(self._config.get("port", "4455")))
        ttk.Entry(h1, textvariable=self.port_var, width=6).pack(side="left", padx=4)
        ttk.Label(h1, text="PW").pack(side="left", padx=(8, 0))
        self.pass_var = tk.StringVar(value=self._config.get("password", ""))
        ttk.Entry(h1, textvariable=self.pass_var, width=12, show="*").pack(side="left", padx=4)
        self.save_password_var = tk.BooleanVar(value=self._config.get("save_password", True))
        ttk.Checkbutton(h1, text="保存", variable=self.save_password_var).pack(side="left", padx=2)
        self.btn_connect = ttk.Button(h1, text="接続", command=self._connect_obs,
                                       width=8, style="Accent.TButton")
        self.btn_connect.pack(side="right", padx=2)
        self.conn_status = tk.StringVar(value="未接続")
        ttk.Label(h1, textvariable=self.conn_status, style="Dim.TLabel").pack(side="right", padx=8)

        # 2段目: ソース + 操作ボタン群
        h2 = tk.Frame(header, bg=T["BG"])
        h2.pack(fill="x", pady=(6, 0))
        ttk.Label(h2, text="ソース").pack(side="left", padx=(20, 4))
        self.source_var = tk.StringVar(value=self._config.get("source", ""))
        self.source_combo = ttk.Combobox(h2, textvariable=self.source_var,
                                          width=28, state="readonly")
        self.source_combo.pack(side="left", padx=4)
        self.btn_save_frame = ttk.Button(h2, text="📷 フレーム保存",
                                          command=self._save_frame,
                                          width=14, state="disabled")
        self.btn_save_frame.pack(side="right", padx=2)
        self.btn_stop = ttk.Button(h2, text="■ 停止", command=self._stop,
                                    width=10, state="disabled")
        self.btn_stop.pack(side="right", padx=2)
        self.btn_start = ttk.Button(h2, text="▶ 開始", command=self._start,
                                     width=10, state="disabled",
                                     style="Accent.TButton")
        self.btn_start.pack(side="right", padx=2)

        # ─── スコア + 状態表記 ───
        score_box = tk.Frame(self.root, bg=T["PANEL"], highlightbackground=T["BORDER"],
                              highlightthickness=1)
        score_box.pack(fill="x", padx=10, pady=4)
        sb_inner = tk.Frame(score_box, bg=T["PANEL"])
        sb_inner.pack(fill="x", pady=6)
        # 左: 勝敗数 + リセット
        left = tk.Frame(sb_inner, bg=T["PANEL"])
        left.pack(side="left", padx=(20, 0))
        self.score_var = tk.StringVar(value="0勝 0敗 0引")
        tk.Label(left, textvariable=self.score_var,
                  bg=T["PANEL"], fg=T["FG"],
                  font=("Segoe UI", 22, "bold")).pack(side="left", padx=(0, 12))
        ttk.Button(left, text="リセット", command=self._reset_score,
                   width=10).pack(side="left", padx=8)
        # 右: 状態表記 (選出中、リザルト など)
        self.state_var = tk.StringVar(value="初期化中...")
        tk.Label(sb_inner, textvariable=self.state_var,
                 bg=T["PANEL"], fg=T["ACCENT"],
                 font=("Segoe UI", 16, "bold")).pack(side="right", padx=(0, 24))

        # ─── 折りたたみ可能セクション ───
        sections = tk.Frame(self.root, bg=T["BG"])
        sections.pack(fill="both", expand=True, padx=10, pady=4)

        # 設定セクション (デフォルト折りたたみ — プレビュー領域を最大化)
        settings_sec = CollapsibleSection(sections, "設定", expanded=False)
        settings_sec.pack(fill="x", pady=(0, 4))

        # 設定行1: 間隔
        sr = tk.Frame(settings_sec.body, bg=T["PANEL"])
        sr.pack(fill="x", pady=(0, 4))
        tk.Label(sr, text="検出間隔", bg=T["PANEL"], fg=T["FG"]).pack(side="left")
        self.interval_var = tk.DoubleVar(value=float(self._config.get("interval", 1.5)))
        ttk.Scale(sr, from_=0.5, to=5.0, variable=self.interval_var,
                  orient="horizontal", length=200).pack(side="left", padx=8)
        self.interval_label = tk.Label(sr, text="1.5秒", bg=T["PANEL"], fg=T["FG"], width=6)
        self.interval_label.pack(side="left")
        self.interval_var.trace_add("write", self._update_interval_label)

        # 設定行2: 出力先
        sr2 = tk.Frame(settings_sec.body, bg=T["PANEL"])
        sr2.pack(fill="x", pady=2)
        tk.Label(sr2, text="出力先", bg=T["PANEL"], fg=T["FG"], width=8, anchor="w").pack(side="left")
        self.output_var = tk.StringVar(value=self._config.get("output_dir", str(OUTPUT_DIR)))
        ttk.Entry(sr2, textvariable=self.output_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(sr2, text="参照", width=6, command=self._browse_output).pack(side="left", padx=2)

        # 設定行3: テンプレート
        sr3 = tk.Frame(settings_sec.body, bg=T["PANEL"])
        sr3.pack(fill="x", pady=2)
        tk.Label(sr3, text="テンプレ", bg=T["PANEL"], fg=T["FG"], width=8, anchor="w").pack(side="left")
        self.templates_var = tk.StringVar(value=self._config.get("templates_dir", str(TEMPLATES_DIR)))
        ttk.Entry(sr3, textvariable=self.templates_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(sr3, text="参照", width=6, command=self._browse_templates).pack(side="left", padx=2)
        ttk.Button(sr3, text="再読込", width=8, command=self._reload_templates).pack(side="left", padx=2)

        # プレビューセクション (生画像 + 切り出し結果)
        preview_sec = CollapsibleSection(sections, "プレビュー", expanded=True)
        preview_sec.pack(fill="both", expand=True, pady=(0, 4))

        # 切り出しプレビュー
        thumbs = tk.Frame(preview_sec.body, bg=T["PANEL"])
        thumbs.pack(fill="both", expand=True)
        def _thumb_box(parent, title, w=24, h=4):
            f = tk.Frame(parent, bg=T["PANEL"])
            tk.Label(f, text=title, bg=T["PANEL"], fg=T["FG_DIM"],
                     font=("Segoe UI", 9, "bold")).pack(anchor="w")
            lbl = tk.Label(f, text="---", bg=T["BTN_BG"], fg=T["FG_DIM"],
                           width=w, height=h, anchor="center")
            lbl.pack(fill="both", expand=True)
            return f, lbl

        # Row 0: 相手チーム (横一列) — full width
        f_oh, self.preview_label = _thumb_box(thumbs, "相手チーム (横一列)", w=80, h=2)
        f_oh.grid(row=0, column=0, columnspan=3, padx=4, pady=(0, 4), sticky="ew")

        # Row 1: 自分選出 (横一列) — full width
        f_mh, self.my_h_preview_label = _thumb_box(thumbs, "自分選出 (横一列)", w=80, h=2)
        f_mh.grid(row=1, column=0, columnspan=3, padx=4, pady=4, sticky="ew")

        # Row 2: 縦プレビュー3つ (相手縦 / 自分縦 / ランク+レート)
        f_ov, self.opp_v_preview_label = _thumb_box(thumbs, "相手(縦)", w=14, h=10)
        f_ov.grid(row=2, column=0, padx=4, pady=4, sticky="n")
        f_mv, self.my_preview_label = _thumb_box(thumbs, "自分(縦)", w=14, h=10)
        f_mv.grid(row=2, column=1, padx=4, pady=4, sticky="n")

        rr_box = tk.Frame(thumbs, bg=T["PANEL"])
        rr_box.grid(row=2, column=2, padx=4, pady=4, sticky="nw")
        f_rk, self.rank_preview_label = _thumb_box(rr_box, "ランク", w=22, h=2)
        f_rk.pack(fill="x", pady=(0, 6))
        f_rt, self.rate_preview_label = _thumb_box(rr_box, "レート", w=22, h=2)
        f_rt.pack(fill="x")

        thumbs.grid_columnconfigure(0, weight=1)
        thumbs.grid_columnconfigure(1, weight=1)
        thumbs.grid_columnconfigure(2, weight=2)

        # ─── ステータスバー (下部固定) ───
        status_bar = tk.Frame(self.root, bg=T["HEADER"])
        status_bar.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="初期化中...")
        tk.Label(status_bar, textvariable=self.status_var,
                  bg=T["HEADER"], fg=T["FG"], font=("Segoe UI", 9),
                  anchor="w", padx=10).pack(side="left", fill="x", expand=True)

        # ─── ログ (折りたたみ、下部) ───
        log_sec = CollapsibleSection(self.root, "ログ", expanded=False)
        log_sec.pack(side="bottom", fill="x", padx=10, pady=(0, 4))
        self.log_text = scrolledtext.ScrolledText(
            log_sec.body, height=6, font=("Consolas", 9),
            state="disabled", bg="#fafafa", fg="#1f1f1f",
            relief="flat", borderwidth=0)
        self.log_text.pack(fill="both", expand=True)

        # 初期 LED 状態
        self.led.set_state("disconnected")

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

    def _browse_templates(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self.templates_var.get(),
                                    title="テンプレートフォルダを選択")
        if d:
            self.templates_var.set(d)
            self._reload_templates()

    # --- 初期化 ---
    def _init_engine(self):
        self.detector = ScreenDetector()
        self._reload_templates(initial=True)

    def _reload_templates(self, initial: bool = False):
        path = self.templates_var.get().strip() or str(TEMPLATES_DIR)
        n, missing, abs_dir = self.detector.load(path)
        # 0件 かつ 配布同梱パス以外 → バンドル既定にフォールバック (旧config.json対応)
        bundled = str(TEMPLATES_DIR)
        try:
            same = Path(abs_dir).resolve() == Path(bundled).resolve()
        except Exception:
            same = False
        if n == 0 and not same and Path(bundled).is_dir():
            self._log(f"[!] {abs_dir} で見つからず、バンドル既定にフォールバック")
            self.templates_var.set(bundled)
            n, missing, abs_dir = self.detector.load(bundled)
            try:
                self._save_config()
            except Exception:
                pass
        self._log(f"テンプレート読込: {n}件 (探索: {abs_dir})")
        if n == 0:
            self.state_var.set("テンプレートなし")
            self._log(f"[!] テンプレート画像が見つかりません: {abs_dir}")
            for fname, reason in missing:
                self._log(f"  - {fname}: {reason}")
        else:
            self.state_var.set("待機中 (OBS未接続)" if initial else f"再読込完了 ({n}件)")
            for fname, reason in missing:
                self._log(f"[?] {fname}: {reason}")
            self._log("初期化完了" if initial else f"テンプレート再読込: {n}件")

    # --- 設定永続化 ---
    def _load_config(self) -> dict:
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                # 8.3短縮パスを長い名前に正規化
                for key in ("output_dir", "templates_dir"):
                    v = cfg.get(key)
                    if v:
                        try:
                            cfg[key] = str(Path(v).resolve())
                        except Exception:
                            pass
                return cfg
        except Exception:
            pass
        return {}

    def _resolve_path(self, p: str) -> str:
        """Windows 8.3 短縮名 (POKEMO~1 等) を長い名前に展開してフルパスに正規化"""
        if not p:
            return p
        try:
            return str(Path(p).resolve())
        except Exception:
            return p

    def _save_config(self):
        save_pw = bool(self.save_password_var.get())
        # 8.3短縮パスを長い名前に正規化してから保存
        out_dir = self._resolve_path(self.output_var.get().strip())
        tmpl_dir = self._resolve_path(self.templates_var.get().strip())
        # GUIの表示も正規化済み値に揃える
        if out_dir != self.output_var.get():
            self.output_var.set(out_dir)
        if tmpl_dir != self.templates_var.get():
            self.templates_var.set(tmpl_dir)
        cfg = {
            "host": self.host_var.get().strip(),
            "port": self.port_var.get().strip(),
            "password": self.pass_var.get() if save_pw else "",
            "save_password": save_pw,
            "source": self.source_var.get(),
            "interval": self.interval_var.get(),
            "output_dir": out_dir,
            "templates_dir": tmpl_dir,
        }
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"[!] 設定保存失敗: {e}")

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
            # 前回保存していたソース名が候補にあればそれを採用、なければ先頭
            saved_source = self._config.get("source", "")
            if saved_source and saved_source in names:
                self.source_var.set(saved_source)
            elif names and not self.source_var.get():
                self.source_var.set(names[0])
            self.conn_status.set(f"接続OK ({len(names)}ソース)")
            self._log(f"OBS接続成功: {len(names)}ソース")
            self.btn_start.configure(state="normal")
            self.state_var.set("準備完了")
            self.led.set_state("ready")
            self._save_config()
        except Exception as e:
            self.conn_status.set("接続失敗")
            self.led.set_state("disconnected")
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
        self.btn_save_frame.configure(state="normal")
        self.state_var.set("検出中...")
        self.led.set_state("running")
        self._log(f"開始: ソース={source}")
        self._save_config()
        self.worker_thread = threading.Thread(target=self._worker_loop,
                                               args=(conn_info,), daemon=True)
        self.worker_thread.start()

    def _stop(self):
        self.running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_save_frame.configure(state="disabled")
        self.state_var.set("停止")
        self.led.set_state("ready")
        self._log("停止")

    def _reset_score(self):
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._log("勝敗カウントをリセット")
        out_dir = Path(self.output_var.get().strip() or OUTPUT_DIR)
        self._write_score_image(out_dir / "score.png")
        self._update_score_label()

    def _update_score_label(self):
        if hasattr(self, "score_var"):
            self.score_var.set(f"{self.win_count}勝 {self.loss_count}敗 {self.draw_count}引")

    def _write_score_image(self, path: Path):
        """score.png として X勝Y敗 画像を出力 (透明PNG, 配信オーバーレイ向け)"""
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            w, h = 480, 120
            im = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            text = f"{self.win_count}勝 {self.loss_count}敗 {self.draw_count}引"
            # Windowsの日本語フォントを試す (Yu Gothic → Meiryo → デフォルト)
            font = None
            for fname in ("YuGothB.ttc", "meiryob.ttc", "meiryo.ttc", "YuGothM.ttc"):
                try:
                    font = ImageFont.truetype(fname, 80)
                    break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()
            # 白文字 + 黒縁取り
            x, y = 20, 10
            for dx in (-3, 0, 3):
                for dy in (-3, 0, 3):
                    if dx == 0 and dy == 0:
                        continue
                    d.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0, 255))
            d.text((x, y), text, font=font, fill=(255, 255, 255, 255))
            path.parent.mkdir(parents=True, exist_ok=True)
            im.save(str(path))
        except Exception as e:
            self._log(f"[!] score.png 生成失敗: {e}")

    def _save_frame(self):
        """現在取得している生フレームをディスクに保存 (デバッグ用)"""
        if self._last_frame is None:
            self._log("[!] 保存できるフレームがまだありません")
            return
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = Path(self.output_var.get().strip() or OUTPUT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            png_path = out_dir / f"debug_frame_{ts}.png"
            txt_path = out_dir / f"debug_frame_{ts}.txt"
            imwrite_unicode(png_path, self._last_frame)
            stats = self._last_frame_stats or {}
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"timestamp: {ts}\n")
                f.write(f"source_name: {stats.get('source_name')}\n")
                f.write(f"resolution: {stats.get('width')}x{stats.get('height')}\n")
                f.write(f"mean: {stats.get('mean'):.2f}\n")
                f.write(f"std: {stats.get('std'):.2f}\n")
                scores = stats.get("scores", {})
                for k, v in scores.items():
                    f.write(f"score[{k}]: {v:.3f}\n")
            self._log(f"フレーム保存: {png_path.name}")
        except Exception as e:
            self._log(f"[!] フレーム保存失敗: {e}")

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

        def _current_paths():
            """GUI の出力先を毎ループ参照してパス構築 (起動中の変更を即反映)"""
            d = Path(self.output_var.get().strip() or OUTPUT_DIR)
            return d, {
                "opp_h": d / "opponent_team.png",
                "opp_v": d / "opponent_team_vertical.png",
                "my_v": d / "my_selection.png",
                "my_h": d / "my_selection_horizontal.png",
                "rank": d / "rank.png",
                "rate": d / "rate.png",
            }
        last_hash = None
        last_hash_v = None
        last_my_hash = None
        last_my_hash_h = None
        # ベストフレーム方式: ブースト中のリザルト切り出しから最高品質1枚を選んで書き出す
        best_result = {"rank": (None, -1.0), "rate": (None, -1.0)}
        best_result_pending = False
        last_result_hash = {"rank": None, "rate": None}
        saved_item_icons = [None] * 6  # 選出前画面で保存したアイテムアイコン
        items_saved = False  # アイテム保存済みフラグ
        selection_locked = False  # 選出完了ロック (3体検出後は相手・自分とも固定)
        boost_until = 0.0  # WIN/LOSE検出後の検出頻度ブースト期限
        match_counted = False  # この対戦の勝敗を既にカウント済みか
        match_flushed = False  # この対戦のリザルト(rank/rate)を既に書き出し済みか
        last_match_result = None  # 直近対戦の勝敗 ("WIN"/"LOSE") - 履歴ログ用

        # 数字テンプレートをロード (テンプレ参照先の digits/ サブフォルダ)
        digit_templates = load_digit_templates(self.templates_var.get().strip() or str(TEMPLATES_DIR))
        if digit_templates:
            self._log(f"数字テンプレ読込: {len(digit_templates)}件 ({''.join(sorted(digit_templates.keys()))})")
        else:
            self._log("[?] 数字テンプレ無し → rank.txt/rate.txt は出力されません")

        def _flush_best_results():
            """ためたベスト rank/rate をディスクに書き出してリセット。
            rank.txt: 整数 / rate.txt: 整数部のみ (OBSテキストソース用)
            rate_full.txt: 小数含む完全値
            battle_history.txt: 1試合1行で全データ追記
            """
            nonlocal best_result_pending, last_match_result
            _, paths_now = _current_paths()
            ocr_results = {}  # {name: full_text}
            for name, (img, q) in best_result.items():
                if img is not None and q > 0:
                    p = paths_now[name]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    imwrite_unicode(p, img)
                    self._log(f"リザルト確定 {name}: q={q:.1f} → {p.name}")
                    if digit_templates:
                        text = digit_ocr(img, digit_templates)
                        ocr_results[name] = text
                        # OBSテキストソース用: rate は整数部のみ、rank は数字のみ
                        out_text = text.split(".")[0] if name == "rate" else text
                        txt_path = p.with_suffix(".txt")
                        try:
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(out_text)
                            self._log(f"  → {txt_path.name}: '{out_text}'")
                        except Exception as e:
                            self._log(f"[!] {txt_path.name} 書き出し失敗: {e}")
                        # rate は完全値も別ファイルに残す
                        if name == "rate" and "." in text:
                            full_path = p.parent / "rate_full.txt"
                            try:
                                with open(full_path, "w", encoding="utf-8") as f:
                                    f.write(text)
                            except Exception:
                                pass
            # 対戦履歴ログ追記 (対戦1試合あたり1行)
            if ocr_results:
                history_path = paths_now["opp_h"].parent / "battle_history.txt"
                try:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    rank_v = ocr_results.get("rank", "")
                    rate_v = ocr_results.get("rate", "")
                    result = last_match_result or "?"
                    line = f"{ts}\t{result}\trank={rank_v}\trate={rate_v}\n"
                    with open(history_path, "a", encoding="utf-8") as f:
                        f.write(line)
                    self._log(f"履歴追記: {history_path.name}")
                except Exception as e:
                    self._log(f"[!] 履歴書き出し失敗: {e}")
            best_result["rank"] = (None, -1.0)
            best_result["rate"] = (None, -1.0)
            best_result_pending = False
            last_match_result = None

        while self.running:
            try:
                out_dir, paths = _current_paths()
                in_boost = time.time() < boost_until
                # ブースト期間中は短い interval で回す
                current_interval = BOOST_INTERVAL_SEC if in_boost else interval
                # LED更新 (ブースト中は橙、それ以外は青)
                self.root.after(0, self.led.set_state,
                                "boost" if in_boost else "running")
                # ブースト終了を検出したらベスト書き出し (1試合1回のみ)
                if not in_boost and best_result_pending and not match_flushed:
                    _flush_best_results()
                    match_flushed = True
                frame = obs_grab_frame(client, source_name)
                if frame is None:
                    time.sleep(current_interval)
                    continue

                key, score, all_scores = self.detector.detect(frame)
                fh, fw = frame.shape[:2]
                frame_mean = float(frame.mean())
                frame_std = float(frame.std())
                # フレーム保存ボタン用に最新フレームとメタデータを保持
                self._last_frame = frame
                self._last_frame_stats = {
                    "width": fw, "height": fh,
                    "mean": frame_mean, "std": frame_std,
                    "scores": dict(all_scores),
                    "source_name": source_name,
                }

                if key == "team_preview":
                    # 新しい対戦のサイクル開始 → ベスト rank/rate を書き出し+カウントロック解除
                    if best_result_pending and not match_flushed:
                        _flush_best_results()
                    match_counted = False
                    match_flushed = False
                    # 選出前 vs 対戦準備中を判定
                    n_selected = count_selected_panels(frame)

                    if n_selected <= 1 and not items_saved:
                        # 選出前画面 → アイテムアイコンを保存 (1回のみ)
                        # 画面遷移直後の暗い画面で取得しないよう全体輝度をチェック
                        if frame_mean < 40:
                            self.root.after(0, self.state_var.set,
                                f"選出前画面 (描画待ち)")
                        else:
                            self.root.after(0, self.state_var.set, "選出前画面 (アイテム取得)")
                            items = extract_item_icons(frame)
                            # 各アイテムの輝度を検証し、暗すぎる (未描画) ものは除外
                            valid = [it for it in items
                                     if it is not None and float(it.mean()) >= 30]
                            if len(valid) >= 3:
                                saved_item_icons = items
                                items_saved = True
                                self._log(f"アイテムアイコン保存: {sum(1 for it in items if it is not None)}体分 (有効{len(valid)}体)")
                    elif n_selected < 3:
                        # 選出中 (まだ3体揃っていない)
                        self.root.after(0, self.state_var.set, f"選出中... ({n_selected}/3)")
                    else:
                        # 選出完了 (3体) → 相手+自分を切り出してロック
                        if not selection_locked:
                            self.root.after(0, self.state_var.set, "選出完了!")

                            # 相手チーム横一列
                            strip = extract_opponent_strip(frame)
                            if strip is not None:
                                paths["opp_h"].parent.mkdir(parents=True, exist_ok=True)
                                imwrite_unicode(paths["opp_h"], strip)
                                self._log(f"相手チーム更新(横): {strip.shape[1]}x{strip.shape[0]}")
                                self.root.after(0, self._update_preview, strip)

                            # 相手チーム縦一列
                            strip_v = extract_opponent_strip_vertical(frame)
                            if strip_v is not None:
                                paths["opp_v"].parent.mkdir(parents=True, exist_ok=True)
                                imwrite_unicode(paths["opp_v"], strip_v)
                                self.root.after(0, self._update_opp_v_preview, strip_v)

                            # 自分選出 縦一列
                            my_strip = extract_my_selection_strip(
                                frame, item_icons=saved_item_icons
                            )
                            if my_strip is not None:
                                paths["my_v"].parent.mkdir(parents=True, exist_ok=True)
                                imwrite_unicode(paths["my_v"], my_strip)
                                self._log(f"自分選出更新(縦): {my_strip.shape[1]}x{my_strip.shape[0]}")
                                self.root.after(0, self._update_my_preview, my_strip)

                            # 自分選出 横一列
                            my_strip_h = extract_my_selection_strip_horizontal(
                                frame, item_icons=saved_item_icons
                            )
                            if my_strip_h is not None:
                                paths["my_h"].parent.mkdir(parents=True, exist_ok=True)
                                imwrite_unicode(paths["my_h"], my_strip_h)
                                self.root.after(0, self._update_my_h_preview, my_strip_h)

                            # ロック
                            selection_locked = True
                            self._log("選出ロック (次の対戦までロック)")
                elif key == "continue_screen":
                    # リザルト画面 → 連勝数/ランク/VP を切り出して候補にためる (ベスト1枚を後で書き出し)
                    self.root.after(0, self.state_var.set, "リザルト画面!")
                    if not match_flushed:
                        regions = extract_result_regions(frame)
                        for name, img in regions.items():
                            if img is None:
                                continue
                            cmean = float(img.mean())
                            cstd = float(img.std())
                            # 遷移アニメ中の真っ黒/単色切り出しはスキップ
                            if cmean < 30 or cstd < 15:
                                continue
                            # スコアリング: 明るさ + コントラスト (=くっきり度) の合計
                            quality = cmean + cstd
                            if quality > best_result[name][1]:
                                best_result[name] = (img.copy(), quality)
                                best_result_pending = True
                                self._log(f"リザルト候補更新 {name}: q={quality:.1f} (mean={cmean:.0f} std={cstd:.0f})")
                                # 都度プレビュー更新 (確認用、ファイルはまだ書かない)
                                self.root.after(0, self._update_result_preview, name, img)
                    # 次対戦に備えてロック解除
                    items_saved = False
                    selection_locked = False
                elif key in ("win_banner", "lose_banner", "draw_banner"):
                    # 勝敗判定画面 → この後のリザルト遷移が速いのでブースト発動
                    if time.time() >= boost_until:
                        self._log(f"{key} 検出 → 検出頻度ブースト {BOOST_DURATION_SEC:.0f}秒")
                    boost_until = time.time() + BOOST_DURATION_SEC
                    # 同一対戦の重複カウント防止
                    if not match_counted:
                        if key == "win_banner":
                            self.win_count += 1
                            last_match_result = "WIN"
                        elif key == "lose_banner":
                            self.loss_count += 1
                            last_match_result = "LOSE"
                        else:  # draw_banner
                            self.draw_count += 1
                            last_match_result = "DRAW"
                        match_counted = True
                        # 新対戦サイクル開始: rank/rate 書き出しフラグもリセット
                        # (team_preview検出が不安定な環境でも次対戦のリザルトを取れるように)
                        match_flushed = False
                        score_path = out_dir / "score.png"
                        self._write_score_image(score_path)
                        self.root.after(0, self._update_score_label)
                        self._log(f"勝敗更新: {self.win_count}勝 {self.loss_count}敗 {self.draw_count}引")
                    state_label = {"win_banner": "勝利!", "lose_banner": "敗北...", "draw_banner": "引き分け"}.get(key, key)
                    self.root.after(0, self.state_var.set, f"{state_label} (ブースト中)")
                    items_saved = False
                    selection_locked = False
                else:
                    # team_preview以外 → 次の対戦に備えてリセット
                    items_saved = False
                    selection_locked = False
                    boost_suffix = " [ブースト中]" if time.time() < boost_until else ""
                    if frame_mean < 5 and frame_std < 5:
                        state_msg = "OBS画面が真っ黒"
                        debug_msg = f"mean={frame_mean:.1f} std={frame_std:.1f} {fw}x{fh}"
                    elif frame_std < 3:
                        state_msg = "OBS画面が単色"
                        debug_msg = f"mean={frame_mean:.1f} std={frame_std:.1f} {fw}x{fh}"
                    elif key:
                        state_msg = f"検出中 ({key})"
                        debug_msg = f"{key}={score:.2f} {fw}x{fh} mean={frame_mean:.0f}{boost_suffix}"
                    else:
                        state_msg = "待機中" + boost_suffix
                        score_str = " ".join(f"{k}:{v:.2f}" for k, v in all_scores.items())
                        debug_msg = f"[{score_str}] {fw}x{fh} mean={frame_mean:.0f}"
                    self.root.after(0, self.state_var.set, state_msg)
                    self.root.after(0, self.status_var.set, debug_msg)

            except Exception as e:
                self._log(f"エラー: {e}")

            time.sleep(current_interval)

        try:
            client.disconnect()
        except Exception:
            pass


    def _show_pil_in_label(self, label, img_bgr, max_w=None, max_h=None):
        """PIL経由で BGR 画像を Tk Label にプレビュー表示する共通関数。
        画像時は width/height がピクセル単位になるため、placeholder時の
        char単位指定をクリアして画像が完全表示されるようにする。
        """
        try:
            from PIL import Image as PILImage, ImageTk
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            if max_w and pil.width > max_w:
                ratio = max_w / pil.width
                pil = pil.resize((max_w, int(pil.height * ratio)))
            if max_h and pil.height > max_h:
                ratio = max_h / pil.height
                pil = pil.resize((int(pil.width * ratio), max_h))
            tk_img = ImageTk.PhotoImage(pil)
            # width/height を 0 にして自動サイズ化 (画像のpx寸法で表示される)
            label.configure(image=tk_img, text="", width=0, height=0)
            label._tk_img = tk_img
        except ImportError:
            label.configure(text="(Pillow未インストール)")

    def _update_preview(self, strip_bgr):
        """相手チーム横一列 (大きめ表示)"""
        self._show_pil_in_label(self.preview_label, strip_bgr, max_w=720)

    def _update_opp_v_preview(self, strip_bgr):
        """相手チーム縦一列"""
        self._show_pil_in_label(self.opp_v_preview_label, strip_bgr, max_h=200)

    def _update_my_preview(self, strip_bgr):
        """自分選出 縦一列"""
        self._show_pil_in_label(self.my_preview_label, strip_bgr, max_h=200)

    def _update_my_h_preview(self, strip_bgr):
        """自分選出 横一列"""
        self._show_pil_in_label(self.my_h_preview_label, strip_bgr, max_w=360)

    def _update_result_preview(self, name, img_bgr):
        """ランク/レート"""
        label = {"rank": self.rank_preview_label, "rate": self.rate_preview_label}.get(name)
        if label is None:
            return
        self._show_pil_in_label(label, img_bgr, max_w=240)

    def _on_close(self):
        self.running = False
        try:
            self._save_config()
        except Exception:
            pass
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
    try:
        main()
    except Exception:
        import traceback
        err_log = EXE_DIR / "startup_error.log"
        try:
            with open(err_log, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        try:
            import tkinter.messagebox as mb
            mb.showerror("起動エラー",
                         f"起動時に例外が発生しました。\n\n"
                         f"詳細: {err_log}\n\n"
                         f"{traceback.format_exc()[:500]}")
        except Exception:
            pass
        raise
