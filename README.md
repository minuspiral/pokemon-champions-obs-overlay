# Pokemon Champions OBS Overlay

Pokemon Champions の対戦配信用オーバーレイツールです。  
OBS の映像キャプチャデバイスから**選出画面を自動検出**し、相手パーティ6体のアイコンを切り出して横一列のPNG画像として出力します。  
OBS の「画像ソース」で読み込めば、配信画面に相手チームをオーバーレイ表示できます。

![概要図](https://img.shields.io/badge/OBS-WebSocket-blue) ![Python](https://img.shields.io/badge/Python-3.10+-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 仕組み

```
OBS 映像キャプチャデバイス
    ↓ (WebSocket)
PokemonOverlay.exe / overlay.py
    ↓ 選出画面検出 → 相手6体スプライト切り出し
output/opponent_team.png
    ↓
OBS 画像ソース → 配信画面にオーバーレイ
```

## セットアップ

### exe版 (推奨・Python不要)

1. [Releases](../../releases) から `PokemonOverlay.exe` をダウンロード
2. 好きな場所に配置

### Python版

```bash
git clone https://github.com/minuspiral/pokemon-champions-obs-overlay.git
cd pokemon-champions-obs-overlay
pip install -r requirements.txt
```

## 使い方

### 1. OBS側の準備

OBS Studio を開いて、WebSocket サーバーを有効にします。

1. **ツール** → **WebSocketサーバー設定**
2. 「**WebSocketサーバーを有効にする**」にチェック
3. ポート: `4455` (デフォルトのまま)
4. パスワードを設定した場合はメモしておく

### 2. ツールの起動

- exe版: `PokemonOverlay.exe` をダブルクリック
- Python版: `python overlay.py`

### 3. OBSに接続

![GUI Screenshot](https://img.shields.io/badge/GUI-screenshot-lightgrey)

1. **ホスト**: `localhost` (同じPCなら変更不要)
2. **ポート**: `4455`
3. **パスワード**: OBS側で設定した場合のみ入力
4. 「**接続**」ボタンをクリック
5. ソースのドロップダウンから**映像キャプチャデバイス**を選択

### 4. 検出開始

1. 「**▶ 開始**」ボタンをクリック
2. ゲーム内で選出画面に入ると自動で相手チームを検出
3. `output/opponent_team.png` が自動更新されます

### 5. OBSにオーバーレイ表示

1. OBS のソース追加 → **「画像」**
2. ファイルに `output/opponent_team.png` を指定  
   (「参照...」ボタンで出力先フォルダを変更した場合はそのパスを指定)
3. シーン上の好きな位置・サイズで配置  
4. 対戦が始まると自動で画像が更新されます

### 出力先の変更

GUI の「**出力先**」欄で「参照...」ボタンからフォルダを変更できます。  
OBS の画像ソースのパスも合わせて変更してください。

## オプション (Python版)

```bash
python overlay.py                # GUI起動
```

GUI上で全て設定できます:
- OBS接続先 (ホスト/ポート/パスワード)
- 映像ソース選択
- 検出間隔 (0.5〜5.0秒)
- 出力先フォルダ

## exe のビルド (開発者向け)

```bash
pip install pyinstaller
build.bat
```

`dist/PokemonOverlay.exe` が生成されます。

## ファイル構成

```
pokemon-champions-obs-overlay/
├── overlay.py          # メインスクリプト
├── build.bat           # exe ビルド用
├── app_icon.ico        # exe アイコン
├── requirements.txt    # 依存パッケージ
├── .gitignore
├── templates/          # 画面検出用テンプレート画像
└── output/             # OBS読み取り用出力 (自動生成)
    └── opponent_team.png
```

## 動作環境

- **OS**: Windows 10 / 11
- **OBS Studio**: 28以上 (WebSocket標準搭載)
- **キャプチャボード**: 1920x1080 出力
- **GPU**: 不要 (CPUのみで動作)

## 依存パッケージ (Python版)

- opencv-python
- numpy
- Pillow
- obsws-python

PyTorch は**不要**です。

## トラブルシューティング

### 「OBS接続失敗」と表示される
- OBS が起動しているか確認
- OBS → ツール → WebSocketサーバー設定 で「有効にする」にチェック
- パスワードを設定している場合はツール側にも入力

### 相手チームが検出されない
- 選出画面が表示されているか確認
- キャプチャ解像度が 1920x1080 か確認
- 検出間隔を短く(0.5秒)して試す

### 画像がOBSに表示されない
- OBS の画像ソースのパスが `output/opponent_team.png` を正しく指しているか確認
- 「ファイルが存在しない場合非表示」がONになっている場合、初回検出まで表示されません

## ライセンス

MIT License
