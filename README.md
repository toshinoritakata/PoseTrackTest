# PoseTrackTest

多人数の姿勢推定・トラッキングパイプラインのテストプロジェクト。

## パイプライン構成

```
入力映像
  → YOLOv11s（人物検出）
  → BoT-SORT + ReID（追跡）
  → RTMPose-m（姿勢推定）
  → One-Euro Filter（キーポイント平滑化）
  → 表示 / 保存
```

顔認識モード（`--face`）では MTCNN + InceptionResnetV1 による
ターゲット追跡レイヤーが追加される。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Apple Silicon (MPS)**: デフォルト設定で YOLO・ReID・RTMPose すべて MPS を使用。
> `facenet-pytorch` は MPS が不安定なため CPU で動作。

## 実行

```bash
# Webカメラ（デフォルト）
python main.py

# 動画ファイル
python main.py --source path/to/video.mp4

# YouTube URL
python main.py --source "https://www.youtube.com/watch?v=..."

# 結果を保存（H.264 優先、非対応環境は MPEG-4 にフォールバック）
python main.py --source video.mp4 --output out.mp4

# 顔認識でターゲットを追跡（クリックで登録）
python main.py --face

# 写真からターゲットを事前登録
python main.py --target-photo person.jpg
```

`q` キーで終了。

## CLIオプション

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--source` | `0` | カメラインデックス、動画ファイルパス、YouTube URL |
| `--yolo-weights` | `yolo11s.pt` | YOLO 重みファイル |
| `--pose-onnx` | RTMPose-m | RTMPose ONNX モデル URL またはローカルパス |
| `--output` | なし | 出力動画の保存先パス |
| `--no-show` | false | ウィンドウ表示を無効化 |
| `--smooth-beta` | `0.01` | One-Euro フィルタの β 値（大きいほど速い動きへの追従が向上） |
| `--face` | false | 顔認識によるターゲット追跡を有効化 |
| `--target-photo` | なし | ターゲット登録用の写真パス（指定すると `--face` は不要） |
| `--face-thresh` | `0.6` | 顔マッチングのコサイン類似度閾値（0〜1） |

## Config パラメータ

`pipeline.py` の `Config` dataclass で詳細なパラメータを調整できる。

### 検出

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `yolo_conf` | `0.4` | 検出信頼度閾値 |
| `yolo_iou` | `0.45` | NMS の IoU 閾値 |
| `yolo_imgsz` | `416` | YOLO 入力解像度（小さいほど高速） |
| `yolo_device` | `mps` | 推論デバイス（`mps` / `cpu` / `cuda`） |

### トラッキング（ReID）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `reid_model` | `osnet_x1_0_msmt17.pt` | ReID モデル（初回起動時に自動ダウンロード） |
| `reid_device` | `mps` | ReID 推論デバイス |
| `reid_stability_thresh` | `5` | N フレーム連続マッチで「安定」と判定し embedding を再利用 |
| `reid_stable_iou_thresh` | `0.5` | 安定トラックの embedding 再利用に必要な IoU 閾値 |

### 姿勢推定

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `pose_stride` | `2` | 姿勢推定を N フレームに 1 回実行（スムーザーが補間） |
| `pose_device` | `mps` | 推論デバイス（`mps` → CoreML EP） |

### 顔認識

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `face_similarity_thresh` | `0.6` | 顔マッチング閾値（高いほど厳格） |
| `face_stride` | `15` | ターゲット確定後の確認間隔（フレーム数） |
| `face_device` | `cpu` | 推論デバイス（MPS は不安定なため cpu 推奨） |

## 顔認識モードの使い方

### クリックで登録

```bash
python main.py --face
```

起動後、ウィンドウ上の追跡対象人物をクリックすると顔が登録される。
登録後はターゲットが緑色のバウンディングボックス（`TARGET ID:X`）で強調表示される。

### 写真から登録

```bash
python main.py --target-photo person.jpg
```

起動直後から指定した写真の人物を自動で追跡する。

## ファイル構成

```
pipeline.py        # メインパイプライン（検出・追跡・姿勢推定）
main.py            # CLI エントリーポイント
smoother.py        # One-Euro フィルタによるキーポイント平滑化
visualizer.py      # バウンディングボックス・スケルトン描画
face_identifier.py # 顔認識（MTCNN + InceptionResnetV1/VGGFace2）
requirements.txt   # 依存パッケージ
```
