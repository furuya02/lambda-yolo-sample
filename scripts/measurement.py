#!/usr/bin/env python3
"""
Lambda関数のパフォーマンス計測スクリプト

11回実行してコールドスタート（1回目）を除いた10回分の平均値を計算
"""
import sys
import time
from pathlib import Path
from typing import Dict, List

# invoke_lambda.pyと同じディレクトリにあるため、直接インポート
from invoke_lambda import encode_image_to_base64, invoke_lambda_function


def run_measurements(
    image_path: str,
    function_name: str = "yolo-sample",
    region: str = "ap-northeast-1",
    num_runs: int = 11
) -> Dict[str, float]:
    """
    Lambda関数を複数回実行して平均値を計算

    Args:
        image_path: 入力画像のパス
        function_name: Lambda関数名
        region: AWSリージョン
        num_runs: 実行回数（デフォルト: 11回）

    Returns:
        Dict[str, float]: 各計測項目の平均値
    """
    # 画像ファイルが存在するか確認
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        sys.exit(1)

    # 画像をBase64エンコード（1回のみ実行）
    print(f"画像を読み込んでいます: {image_path}")
    image_base64 = encode_image_to_base64(str(image_file))

    # 計測結果を格納するリスト
    elapsed_times: List[float] = []
    decode_times: List[float] = []
    yolo_total_times: List[float] = []
    inference_times: List[float] = []
    plot_times: List[float] = []
    detection_list_times: List[float] = []
    encode_times: List[float] = []
    summary_times: List[float] = []

    print(f"\nLambda関数を{num_runs}回実行します...")
    print("=" * 60)

    # Lambda関数を複数回実行
    for i in range(num_runs):
        run_number = i + 1
        print(f"\n[{run_number}/{num_runs}] 実行中...", end="", flush=True)

        try:
            # Lambda関数を呼び出し
            response, elapsed_ms = invoke_lambda_function(
                function_name=function_name,
                image_base64=image_base64,
                region=region
            )

            # ステータスコードを確認
            status_code = response.get("statusCode", 500)
            if status_code != 200:
                print(f"\nエラー: Lambda関数がエラーを返しました (status code: {status_code})")
                print(response)
                continue

            # レスポンスボディを解析
            import json
            body = json.loads(response["body"])
            timing_breakdown = body.get("timing_breakdown", {})

            # 計測値を記録（1回目はスキップ）
            if i > 0:  # 2回目以降のみ記録
                elapsed_times.append(elapsed_ms)
                decode_times.append(timing_breakdown.get('decode_ms', 0))
                yolo_total_times.append(timing_breakdown.get('yolo_total_ms', 0))
                inference_times.append(timing_breakdown.get('inference_ms', 0))
                plot_times.append(timing_breakdown.get('plot_ms', 0))
                detection_list_times.append(timing_breakdown.get('detection_list_ms', 0))
                encode_times.append(timing_breakdown.get('encode_ms', 0))
                summary_times.append(timing_breakdown.get('summary_ms', 0))

            # 進捗表示
            if i == 0:
                print(" 完了（コールドスタート - 集計対象外）")
            else:
                print(f" 完了（{elapsed_ms:.2f} ms）")

        except Exception as e:
            print(f"\nエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)

    # 平均値を計算
    if not elapsed_times:
        print("エラー: 有効な計測データがありません")
        sys.exit(1)

    num_valid_runs = len(elapsed_times)
    print(f"\n集計: {num_valid_runs}回の平均値を計算")

    avg_elapsed = sum(elapsed_times) / num_valid_runs
    avg_decode = sum(decode_times) / num_valid_runs
    avg_yolo_total = sum(yolo_total_times) / num_valid_runs
    avg_inference = sum(inference_times) / num_valid_runs
    avg_plot = sum(plot_times) / num_valid_runs
    avg_detection_list = sum(detection_list_times) / num_valid_runs
    avg_encode = sum(encode_times) / num_valid_runs
    avg_summary = sum(summary_times) / num_valid_runs

    # Lambda内の計測合計
    avg_total_measured = avg_decode + avg_yolo_total + avg_encode + avg_summary

    # その他（オーバーヘッド等）
    avg_other = avg_elapsed - avg_total_measured

    return {
        "elapsed_ms": avg_elapsed,
        "total_measured_ms": avg_total_measured,
        "decode_ms": avg_decode,
        "yolo_total_ms": avg_yolo_total,
        "inference_ms": avg_inference,
        "plot_ms": avg_plot,
        "detection_list_ms": avg_detection_list,
        "encode_ms": avg_encode,
        "summary_ms": avg_summary,
        "other_ms": avg_other,
    }


def print_results(averages: Dict[str, float]) -> None:
    """
    計測結果を階層的フォーマットで出力

    Args:
        averages: 各計測項目の平均値
    """
    print("\n" + "=" * 60)
    print("計測結果（平均値）")
    print("=" * 60)
    print(f"総処理時間: {averages['elapsed_ms']:.2f} ms")
    print(f"   Lambda内の計測: {averages['total_measured_ms']:.2f} ms")
    print(f"      Base64デコード: {averages['decode_ms']:.2f} ms")
    print(f"      YOLO処理合計: {averages['yolo_total_ms']:.2f} ms")
    print(f"         - 推論: {averages['inference_ms']:.2f} ms")
    print(f"         - 結果描画: {averages['plot_ms']:.2f} ms")
    print(f"         - 検出リスト作成: {averages['detection_list_ms']:.2f} ms")
    print(f"      Base64エンコード: {averages['encode_ms']:.2f} ms")
    print(f"      サマリー作成: {averages['summary_ms']:.2f} ms")
    print(f"   その他（オーバーヘッド等）: {averages['other_ms']:.2f} ms")
    print("=" * 60)


def main() -> None:
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Lambda関数のパフォーマンス計測（11回実行、1回目除外）"
    )
    parser.add_argument(
        "--image",
        "-i",
        default="00001.jpeg",
        help="入力画像のパス（デフォルト: 00001.jpeg）"
    )
    parser.add_argument(
        "--function-name",
        "-f",
        default="yolo-sample",
        help="Lambda関数名（デフォルト: yolo-sample）"
    )
    parser.add_argument(
        "--region",
        "-r",
        default="ap-northeast-1",
        help="AWSリージョン（デフォルト: ap-northeast-1）"
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=11,
        help="実行回数（デフォルト: 11）"
    )

    args = parser.parse_args()

    # 計測実行
    start_time = time.time()
    averages = run_measurements(
        image_path=args.image,
        function_name=args.function_name,
        region=args.region,
        num_runs=args.runs
    )
    total_time = time.time() - start_time

    # 結果を表示
    print_results(averages)
    print(f"\n全体の実行時間: {total_time:.2f} 秒")


if __name__ == "__main__":
    main()
