# 再現プロンプト: Pokemon Champions OBS Overlay

以下のプロンプトをAIに渡せば、このプロジェクトを一から再現できます。

---

## プロンプト

```
Pokemon Champions (Nintendo Switch 2) の対戦配信用OBSオーバーレイツールを作成してください。

### 目的
キャプチャボードの映像から選出画面を自動検出し、相手パーティ6体のスプライトを切り出して横一列のPNG画像として出力する。OBSの「画像ソース」で読み取ることで配信画面にオーバーレイ表示する。

### 要件

#### 映像取得
- OBS WebSocket API (obsws-python) 経由でOBSの映像キャプチャデバイスソースからスクリーンショットを取得する
- 直接キャプチャデバイスを開かない（OBS経由にすることでどのキャプチャデバイスでも汎用的に動作）
- OBS 28+ は WebSocket プラグイン標準搭載

#### 選出画面検出
- テンプレートマッチング (cv2.matchTemplate, TM_CCOEFF_NORMED) で選出画面を検出
- テンプレート画像は「ランクバトル シングルバトル」ヘッダー部分 (team_preview_header.png)
- フレーム上部ROI (Y: 0-8%, X: 25-75%) と照合、閾値 0.65 以上で検出

#### 相手スプライト切り出し
- 選出画面の右側パネルから相手6体のスプライトを固定ROI座標 (1920x1080基準の比率) で切り出し
  - Y起点: 0.1528, Yステップ: 0.1167, Y高さ: 0.0926
  - X範囲: 0.7240 - 0.7864
- 各スプライトを128x128にリサイズ
- 白セパレータ(2px)を挟んで横一列に連結

#### 出力
- 連結画像を output/opponent_team.png として保存 (BGR, JPG品質)
- 画像に変化があった時のみ上書き (ハッシュ比較)
- 出力フォルダはGUIから変更可能

#### GUI (tkinter)
- OBS接続設定: ホスト(localhost)/ポート(4455)/パスワード
- 「接続」ボタン → OBSに接続してソース一覧をドロップダウンに表示
- ソース選択: 映像キャプチャデバイスを選ぶ
- 検出間隔スライダー: 0.5-5.0秒 (デフォルト1.5)
- 出力先フォルダ: テキスト入力 + 「参照...」ボタン (filedialog)
- 「▶ 開始」/「■ 停止」ボタン
- 状態表示: 「待機中」「検出中...」「選出画面検出!」
- プレビュー: 最新の切り出し画像をGUI内に表示 (PIL ImageTk)
- ログ: ScrolledText でタイムスタンプ付きログ表示

#### スレッド設計
- GUI = メインスレッド (tkinter mainloop)
- 検出ループ = ワーカースレッド (daemon)
- ワーカー起動前にメインスレッドで接続情報を辞書に取得してワーカーに渡す (tkinter変数をスレッドから読まない)
- GUI更新は root.after() 経由

#### PyInstaller exe化
- --onefile --windowed でコンソール非表示のexeを生成
- --add-data でtemplatesフォルダを同梱
- --icon でモンスターボール風アイコン (app_icon.ico) を設定
- --hidden-import obsws_python
- リソースパス解決: sys.frozen 時は sys._MEIPASS、通常時は __file__ のディレクトリ
- exe横に output/ が自動生成される

#### 依存パッケージ (PyTorch不要)
- opencv-python>=4.8
- numpy
- Pillow
- obsws-python>=1.7

#### ファイル構成
```
overlay.py          # メインスクリプト (1ファイルに全ロジック統合)
build.bat           # PyInstaller ビルド用バッチ
app_icon.ico        # exe用アイコン (モンスターボール風、256x256 ICO)
app_icon.png        # アイコン原画
requirements.txt
.gitignore          # output/, dist/, build/, icons/, __pycache__/ を除外
LICENSE             # MIT
README.md           # セットアップ・使い方・トラブルシューティング
version_info.txt    # PyInstaller用バージョン情報
templates/          # 画面検出テンプレート画像 (team_preview_header.png 等)
```

#### overlay.py の構成 (1ファイル、約280行)
1. リソースパス解決 (frozen対応)
2. ROI定数定義
3. ScreenDetector クラス (テンプレートマッチング)
4. extract_opponent_strip() 関数 (6体切り出し+横連結)
5. obs_grab_frame() 関数 (OBS WebSocket → base64 → OpenCV frame)
6. OverlayApp クラス (tkinter GUI + ワーカースレッド)
7. main() (tkinter mainloop)

#### アイコン生成 (Pillow)
- 256x256 RGBA
- モンスターボール風デザイン: 上半分赤、下半分紺、中央に黒帯+白丸ボタン
- PILで描画 → ICO (16/32/48/64/128/256) + PNG で保存

#### GitHub公開
- MIT License
- README.md に使い方(OBS WebSocket有効化手順、GUI操作、OBS画像ソース設定)を記載
- .gitignore で output/, dist/, build/ を除外
- Releases に exe をアップロード

#### セキュリティ要件
- 外部サーバーへの通信なし (OBS localhost のみ)
- eval/exec/subprocess 不使用
- パスワードはログに出力しない
- ファイル書き込みは output/opponent_team.png のみ
```

---

## テンプレート画像について

`templates/` 内の画像は実機のキャプチャ画面から切り出したものです。再現時は自分の環境のキャプチャ画面から以下を切り出してください:

- `team_preview_header.png`: 選出画面上部の「ランクバトル シングルバトル」ヘッダー部分

他のテンプレート (battle_menu_button.png, continue_button.png 等) は将来の拡張用です。
