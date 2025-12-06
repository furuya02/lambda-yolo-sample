#!/usr/bin/env python3
"""
Lambda関数を呼び出すテストスクリプト

使い方:
    python invoke_lambda.py --image path/to/image.jpg
    python invoke_lambda.py --image path/to/image.jpg --save-result output.jpg
    python invoke_lambda.py --image path/to/image.jpg --function-name yolo-sample
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import boto3
from PIL import Image


def encode_image_to_base64(image_path: str) -> str:
    """
    画像ファイルをBase64文字列にエンコード

    Args:
        image_path: 画像ファイルのパス

    Returns:
        str: Base64エンコードされた画像文字列
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_base64_to_image(base64_string: str, output_path: str) -> None:
    """
    Base64文字列を画像ファイルに保存

    Args:
        base64_string: Base64エンコードされた画像文字列
        output_path: 出力ファイルのパス
    """
    image_bytes = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(image_bytes)


def invoke_lambda_function(
    function_name: str,
    image_base64: str,
    region: str = "ap-northeast-1"
) -> Tuple[Dict[str, Any], float]:
    """
    Lambda関数を呼び出す

    Args:
        function_name: Lambda関数名
        image_base64: Base64エンコードされた画像文字列
        region: AWSリージョン

    Returns:
        Tuple[Dict[str, Any], float]: Lambda関数のレスポンスと実行時間（ミリ秒）
    """
    # Lambda クライアントを作成
    lambda_client = boto3.client("lambda", region_name=region)

    # リクエストペイロードを作成
    payload = {
        "image": image_base64
    }

    print(f"Lambda関数 '{function_name}' を呼び出しています...")

    # 実行時間の計測開始
    start_time = time.time()

    # Lambda関数を呼び出し
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload)
    )

    # レスポンスを解析
    response_payload = json.loads(response["Payload"].read())

    # 実行時間の計測終了
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    return response_payload, elapsed_ms


def main() -> None:
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Lambda関数を呼び出してYOLO物体検出を実行"
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="入力画像のパス"
    )
    parser.add_argument(
        "--function-name",
        "-f",
        default="yolo-sample",
        help="Lambda関数名 (デフォルト: yolo-sample)"
    )
    parser.add_argument(
        "--region",
        "-r",
        default="ap-northeast-1",
        help="AWSリージョン (デフォルト: ap-northeast-1)"
    )
    parser.add_argument(
        "--save-result",
        "-s",
        help="検出結果画像の保存先パス"
    )

    args = parser.parse_args()

    # 画像ファイルが存在するか確認
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"エラー: 画像ファイルが見つかりません: {args.image}")
        sys.exit(1)

    # 画像をBase64エンコード
    print(f"画像を読み込んでいます: {args.image}")
    image_base64 = encode_image_to_base64(str(image_path))

    # Lambda関数を呼び出し
    try:
        response, elapsed_ms = invoke_lambda_function(
            function_name=args.function_name,
            image_base64=image_base64,
            region=args.region
        )

        # ステータスコードを確認
        status_code = response.get("statusCode", 500)
        if status_code != 200:
            print(f"エラー: Lambda関数がエラーを返しました (status code: {status_code})")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            sys.exit(1)

        # レスポンスボディを解析
        body = json.loads(response["body"])

        # 検出結果を表示
        print("\n" + "="*60)
        print("検出結果:")
        print("="*60)

        summary = body.get("summary", {})
        print(f"\n総検出数: {summary.get('total_detections', 0)}")
        print(f"検出されたクラス: {', '.join(summary.get('classes_detected', []))}")

        print("\n詳細:")
        detections = body.get("detections", [])
        for i, detection in enumerate(detections, 1):
            print(f"\n  [{i}] {detection['class_name']}")
            print(f"      信頼度: {detection['confidence']:.3f}")
            print(f"      位置: {detection['bbox']}")

        # 検出結果画像を保存
        if args.save_result:
            annotated_image_base64 = body.get("annotatedImage")
            if annotated_image_base64:
                decode_base64_to_image(annotated_image_base64, args.save_result)
                print(f"\n検出結果画像を保存しました: {args.save_result}")

        # Lambda実行時間と推論時間を表示
        inference_time_ms = body.get("inference_time_ms", 0)
        timing_breakdown = body.get("timing_breakdown", {})

        print(f"\nLambda実行時間: {elapsed_ms:.2f} ミリ秒")
        print(f"推論処理時間: {inference_time_ms:.2f} ミリ秒")

        # 処理時間の内訳を表示
        if timing_breakdown:
            print("\n処理時間の内訳:")
            print(f"  Base64デコード: {timing_breakdown.get('decode_ms', 0):.2f} ms")
            print(f"  YOLO処理合計: {timing_breakdown.get('yolo_total_ms', 0):.2f} ms")
            print(f"    - 推論: {timing_breakdown.get('inference_ms', 0):.2f} ms")
            print(f"    - 結果描画: {timing_breakdown.get('plot_ms', 0):.2f} ms")
            print(f"    - 検出リスト作成: {timing_breakdown.get('detection_list_ms', 0):.2f} ms")
            print(f"  Base64エンコード: {timing_breakdown.get('encode_ms', 0):.2f} ms")
            print(f"  サマリー作成: {timing_breakdown.get('summary_ms', 0):.2f} ms")

            # 合計と差分を計算
            total_measured = (
                timing_breakdown.get('decode_ms', 0) +
                timing_breakdown.get('yolo_total_ms', 0) +
                timing_breakdown.get('encode_ms', 0) +
                timing_breakdown.get('summary_ms', 0)
            )
            other_time = elapsed_ms - total_measured
            print(f"  その他（オーバーヘッド等）: {other_time:.2f} ms")
            print(f"  計測合計: {total_measured:.2f} ms")

        print("\n" + "="*60)
        print("処理が完了しました")
        print("="*60)

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
