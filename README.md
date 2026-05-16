# PoseTrackTest

多人数の姿勢推定・トラッキングパイプラインのテストプロジェクト。

## パイプライン構成

```
入力映像 → YOLOv11s（人物検出）→ BoT-SORT + ReID（追跡）→ RTMPose-m（姿勢推定）→ One-Euro（平滑化）→ 表示/保存
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行

```bash
# Webカメラ
python main.py

# 動画ファイル
python main.py --source path/to/video.mp4

# YouTube URL
python main.py --source "https://www.youtube.com/watch?v=..."

# 結果を保存
python main.py --source video.mp4 --output out.mp4
```

`q` キーで終了。

## オプション

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--source` | `0` | カメラインデックス、動画ファイルパス、またはYouTube URL |
| `--yolo-weights` | `yolo11s.pt` | YOLOの重みファイル |
| `--pose-onnx` | RTMPose-m | RTMPoseのONNXモデルURLまたはローカルパス |
| `--output` | なし | 出力動画の保存先パス |
| `--no-show` | false | ウィンドウ表示を無効化 |
| `--smooth-beta` | `0.01` | One-Euroフィルタのβ値（大きいほど速い動きへの追従が向上） |

## 設定

`pipeline.py` の `Config` で詳細なパラメータを調整できます。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `yolo_conf` | `0.4` | 検出の信頼度閾値 |
| `yolo_imgsz` | `416` | YOLO入力解像度 |
| `reid_model` | `osnet_x1_0_msmt17.pt` | ReIDモデル（初回起動時に自動ダウンロード） |
| `reid_device` | `mps` | ReID推論デバイス（`mps` / `cpu`） |
| `pose_stride` | `2` | 姿勢推定をNフレームに1回実行 |

## ファイル構成

```
pipeline.py    # メインパイプライン（検出・追跡・姿勢推定）
main.py        # CLIエントリーポイント
smoother.py    # One-Euroフィルタによるキーポイント平滑化
visualizer.py  # バウンディングボックス・スケルトン描画
```
