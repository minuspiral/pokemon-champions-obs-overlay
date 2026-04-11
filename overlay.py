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
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
    EXE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
    EXE_DIR = BASE_DIR

TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = EXE_DIR / "output"

# ─────────────────── 定数 ───────────────────
# 相手スプライト ROI (prep layout, 1920x1080 基準)
OPP_Y_FIRST = 0.1528
OPP_Y_STEP = 0.1167
OPP_Y_H = 0.0926
OPP_X_START = 0.7240
OPP_X_END = 0.7864

# 画面検出テンプレート
SCREEN_TEMPLATES = [
    ("team_preview", "team_preview_header.png", (0.0, 0.08, 0.25, 0.75), 0.65),
]

ICON_SIZE = 128  # 切り出し後のリサイズサイズ
SEP_WIDTH = 2    # アイコン間のセパレータ幅


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
            rh, rw = roi.shape[:2]
            th, tw = tmpl.shape[:2]
            # 解像度が異なる場合テンプレートをROIに合わせてリサイズ
            if rh != 0 and th != 0:
                scale = rh / th
                if abs(scale - 1.0) > 0.1:
                    tmpl = cv2.resize(tmpl, (int(tw * scale), rh),
                                      interpolation=cv2.INTER_AREA)
                    th, tw = tmpl.shape[:2]
            if rh < th or rw < tw:
                continue
            result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score >= threshold and score > best_score:
                best_key, best_score = key, score
        return best_key, best_score


# ─────────────────── スプライト切り出し ───────────────────
def extract_opponent_strip(frame, icon_size=ICON_SIZE):
    """選出画面から相手6体のスプライトを切り出し、横一列に連結した画像を返す。

    Returns: numpy array (icon_size x (icon_size*6 + sep*5), 3ch BGR) or None
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    icons = []
    for i in range(6):
        y0 = max(0, int(h * (OPP_Y_FIRST + OPP_Y_STEP * i)))
        y1 = min(h, int(h * (OPP_Y_FIRST + OPP_Y_STEP * i + OPP_Y_H)))
        x0 = max(0, int(w * OPP_X_START))
        x1 = min(w, int(w * OPP_X_END))
        if y1 <= y0 or x1 <= x0:
            return None
        roi = frame[y0:y1, x0:x1]
        resized = cv2.resize(roi, (icon_size, icon_size), interpolation=cv2.INTER_CUBIC)
        icons.append(resized)

    # セパレータ(白2px)を挟んで横連結
    parts = []
    for i, icon in enumerate(icons):
        parts.append(icon)
        if i < 5:
            sep = np.ones((icon_size, SEP_WIDTH, 3), dtype=np.uint8) * 255
            parts.append(sep)
    return np.hstack(parts)


# ─────────────────── OBS WebSocket ───────────────────
def obs_grab_frame(client, source_name, width=1920, height=1080):
    """OBSソースからスクリーンショット1枚取得→OpenCV frame。失敗時None。"""
    try:
        resp = client.get_source_screenshot(
            name=source_name,
            img_format="jpg",
            width=width, height=height,
            quality=85,
        )
        img_b64 = getattr(resp, "image_data", None)
        if not img_b64:
            return None
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        raw = base64.b64decode(img_b64)
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


# ─────────────────── GUI ───────────────────
import tkinter as tk
from tkinter import ttk, scrolledtext


class OverlayApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OBS Pokemon Champions Overlay")
        self.root.geometry("620x480")
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
        preview = ttk.LabelFrame(self.root, text="相手チーム (最新)", padding=4)
        preview.pack(fill="x", padx=8, pady=4)
        self.preview_label = ttk.Label(preview, text="選出画面待ち...", anchor="center")
        self.preview_label.pack()

        # === ログ ===
        log_frame = ttk.LabelFrame(self.root, text="ログ", padding=4)
        log_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=8, font=("Consolas", 9),
            state="disabled", bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill="both", expand=True)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        try:
            self.log_text.after(0, self._insert_log, f"{ts} {msg}")
        except (RuntimeError, Exception):
            pass  # ウィンドウ破棄後のTclError防止

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
        try:
            port = int(self.port_var.get().strip() or "4455")
        except ValueError:
            self._log("[!] ポート番号が不正です")
            return
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
        try:
            port = int(self.port_var.get().strip() or "4455")
        except ValueError:
            self._log("[!] ポート番号が不正です")
            return
        conn_info = {
            "host": self.host_var.get().strip(),
            "port": port,
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
        img_tmp = out_dir / "opponent_team.tmp.png"
        last_hash = None
        error_count = 0
        MAX_ERRORS = 10  # 連続エラーで自動停止

        while self.running:
            try:
                frame = obs_grab_frame(client, source_name)
                if frame is None:
                    error_count += 1
                    if error_count >= MAX_ERRORS:
                        self._log(f"[!] {MAX_ERRORS}回連続取得失敗 → 停止")
                        self.root.after(0, self._stop)
                        break
                    time.sleep(interval)
                    continue
                error_count = 0  # 成功したらリセット

                key, score = self.detector.detect(frame)

                if key == "team_preview":
                    self.root.after(0, self.status_var.set, "選出画面検出!")
                    strip = extract_opponent_strip(frame)
                    if strip is not None:
                        # 変化チェック (全体ハッシュ)
                        h = hash(strip.tobytes())
                        if h != last_hash:
                            img_out.parent.mkdir(parents=True, exist_ok=True)
                            # アトミック書き込み (tmp→rename でOBS読取中の破損防止)
                            cv2.imwrite(str(img_tmp), strip)
                            try:
                                img_tmp.replace(img_out)
                            except OSError:
                                # replace失敗時は直接書き込み
                                cv2.imwrite(str(img_out), strip)
                            last_hash = h
                            self._log(f"出力更新: {strip.shape[1]}x{strip.shape[0]}")
                            self.root.after(0, self._update_preview, strip)
                else:
                    s = f"{key}({score:.2f})" if key else "待機中"
                    self.root.after(0, self.status_var.set, s)

            except Exception as e:
                error_count += 1
                self._log(f"エラー: {e}")
                if error_count >= MAX_ERRORS:
                    self._log(f"[!] 連続エラー{MAX_ERRORS}回 → 停止")
                    self.root.after(0, self._stop)
                    break

            time.sleep(interval)

        try:
            client.disconnect()
        except Exception:
            pass

    def _update_preview(self, strip_bgr):
        """切り出し結果をGUIにプレビュー表示"""
        try:
            from PIL import Image as PILImage, ImageTk
            rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            # GUI幅に合わせてリサイズ
            max_w = 580
            if pil.width > max_w:
                ratio = max_w / pil.width
                pil = pil.resize((max_w, int(pil.height * ratio)))
            tk_img = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=tk_img, text="")
            self.preview_label._tk_img = tk_img
        except ImportError:
            self.preview_label.configure(text="(Pillow未インストール: プレビュー不可)")

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
