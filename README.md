# YOLO Sample - AWS Lambda Object Detection

AWS LambdaでYOLOv8を使った物体検出を行うシンプルなサンプルプロジェクトです。

## プロジェクト構成

```
lambda-yolo-sample/
├── cdk/                      # CDKプロジェクト
│   ├── bin/
│   │   └── cdk.ts           # CDKアプリエントリーポイント
│   ├── lib/
│   │   └── cdk-stack.ts     # CDKスタック定義
│   ├── lambda/
│   │   ├── Dockerfile       # Lambda用Dockerイメージ
│   │   ├── requirements.txt # Python依存関係
│   │   └── lambda_function.py # Lambdaハンドラー
│   ├── package.json
│   ├── tsconfig.json
│   └── cdk.json
├── scripts/                  # テストスクリプト
│   ├── invoke_lambda.py     # Lambda呼び出しスクリプト
│   └── requirements.txt     # スクリプト用依存関係
└── README.md
```

## 機能

- YOLOv8を使った物体検出
- Base64エンコードされた画像を受け取り、検出結果を返す
- 検出結果画像（バウンディングボックス付き）をBase64で返す
- 検出されたオブジェクトのリスト（クラス名、信頼度、位置）を返す

## 前提条件

- Node.js 18以上
- Python 3.11以上
- AWS CLI設定済み
- Docker

## セットアップ

### 1. CDK依存関係のインストール

```bash
cd cdk
pnpm install
```

### 2. CDKのブートストラップ（初回のみ）

```bash
pnpm exec cdk bootstrap
```

### 3. デプロイ

```bash
pnpm exec cdk deploy
```

デプロイには10-15分程度かかります（Dockerイメージのビルドに時間がかかります）。

### 4. テストスクリプトの準備

```bash
cd ../scripts
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 使い方

### Lambda関数の呼び出し

```bash
cd scripts
source venv/bin/activate  # Windows: venv\Scripts\activate

# 基本的な使い方
python invoke_lambda.py --image path/to/your/image.jpg

# 検出結果画像を保存
python invoke_lambda.py --image path/to/your/image.jpg --save-result result.jpg

# カスタムLambda関数名を指定
python invoke_lambda.py --image path/to/your/image.jpg --function-name your-function-name

# リージョンを指定
python invoke_lambda.py --image path/to/your/image.jpg --region us-east-1
```

### 出力例

```
画像を読み込んでいます: sample.jpg
Lambda関数 'yolo-sample' を呼び出しています...

============================================================
検出結果:
============================================================

総検出数: 3
検出されたクラス: person, car, dog

詳細:

  [1] person
      信頼度: 0.952
      位置: [120.5, 80.3, 350.2, 450.8]

  [2] car
      信頼度: 0.887
      位置: [400.1, 200.5, 600.3, 380.2]

  [3] dog
      信頼度: 0.723
      位置: [50.2, 300.1, 150.8, 420.5]

検出結果画像を保存しました: result.jpg

Lambda実行時間: 263.26 ミリ秒
推論処理時間: 49.15 ミリ秒

処理時間の内訳:
  Base64デコード: 4.23 ms
  YOLO処理合計: 63.71 ms
    - 推論: 49.15 ms
    - 結果描画: 14.39 ms
    - 検出リスト作成: 0.15 ms
  Base64エンコード: 3.53 ms
  サマリー作成: 0.01 ms
  その他（オーバーヘッド等）: 191.78 ms
  計測合計: 71.48 ms

============================================================
処理が完了しました
============================================================
```

## Lambda関数のAPI仕様

### リクエスト

```json
{
  "image": "base64エンコードされた画像文字列"
}
```

### レスポンス（成功時）

```json
{
  "statusCode": 200,
  "body": {
    "annotatedImage": "base64エンコードされた検出結果画像",
    "detections": [
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": [120.5, 80.3, 350.2, 450.8]
      }
    ],
    "summary": {
      "total_detections": 3,
      "classes_detected": ["person", "car", "dog"]
    }
  }
}
```

### レスポンス（エラー時）

```json
{
  "statusCode": 400,
  "body": {
    "error": "エラーメッセージ",
    "type": "エラータイプ"
  }
}
```

## 環境変数

Lambda関数で設定可能な環境変数:

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `MODEL_NAME` | `yolov8n.pt` | 使用するYOLOモデル |
| `CONF_THRESHOLD` | `0.25` | 信頼度の閾値 |
| `IOU_THRESHOLD` | `0.45` | IoUの閾値 |

## カスタマイズ

### 異なるYOLOモデルを使う

[cdk/lib/cdk-stack.ts](cdk/lib/cdk-stack.ts:36) の環境変数を変更:

```typescript
environment: {
  MODEL_NAME: 'yolov8m.pt',  // n, s, m, l, x から選択
  CONF_THRESHOLD: '0.25',
  IOU_THRESHOLD: '0.45',
},
```

### メモリとタイムアウトの調整

[cdk/lib/cdk-stack.ts](cdk/lib/cdk-stack.ts:33-34) で変更:

```typescript
memorySize: 3008,  // MB
timeout: cdk.Duration.seconds(120),  // 秒
```

## トラブルシューティング

### デプロイ時にエラーが発生する

- Dockerが起動しているか確認
- AWS認証情報が正しく設定されているか確認
- 十分なディスク容量があるか確認

### Lambda関数がタイムアウトする

- メモリサイズを増やす（メモリを増やすとCPUも増える）
- タイムアウト時間を延長する

### 検出精度が低い

- `CONF_THRESHOLD` を調整（低くすると検出数が増えるが誤検出も増える）
- より大きなモデル（yolov8m, yolov8l など）を使用

## クリーンアップ

リソースを削除する場合:

```bash
cd cdk
pnpm exec cdk destroy
```

## 参考リンク

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [AWS CDK](https://docs.aws.amazon.com/cdk/)
- [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)

## ライセンス

MIT License
