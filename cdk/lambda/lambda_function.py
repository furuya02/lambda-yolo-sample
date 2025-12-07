"""
AWS Lambda関数ハンドラー - シンプルYOLOサンプル
YOLOv8を使った物体検出のサンプル実装
"""

import json
import base64
import os
import time
from io import BytesIO
from typing import Dict, Any, List

import cv2
import numpy as np

# 【YOLO推論の最適化 前】import不要
# import torch
# 【YOLO推論の最適化 後】torch.inference_mode()を使用するため
import torch
from PIL import Image
from ultralytics import YOLO

# グローバル変数（コールドスタート対策）
model = None


def initialize_model():
    """
    YOLOモデルを初期化（初回のみ実行）

    Returns:
        YOLO: YOLOv8モデルインスタンス
    """

    global model

    if model is None:
        # デフォルトでカスタムモデルを使用
        model_name = os.environ.get("MODEL_NAME", "/opt/ml/model/best.pt")

        print(f"Initializing YOLO model: {model_name}")

        # 【YOLO推論の最適化 前】
        # model = YOLO(model_name)
        # print("YOLO model loaded successfully")

        # 【YOLO推論の最適化 後】レイヤーをfuse（10-20%高速化）
        model = YOLO(model_name)
        model.fuse()
        print("YOLO model loaded and fused successfully")

    return model


def decode_base64_image(base64_string: str):
    """
    Base64文字列を画像(numpy配列)にデコード

    Args:
        base64_string: Base64エンコードされた画像文字列

    Returns:
        np.ndarray: OpenCV形式の画像 (BGR, numpy.ndarray)
    """
    # Base64デコード
    image_bytes = base64.b64decode(base64_string)

    # PILで画像を開く
    image_pil = Image.open(BytesIO(image_bytes))

    # RGB -> BGR変換（OpenCV形式）
    image_rgb = np.array(image_pil)
    if len(image_rgb.shape) == 2:  # グレースケール
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    elif image_rgb.shape[2] == 4:  # RGBA
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr


# 【画像エンコード形式 前】
# def encode_image_to_base64(image, format: str = "PNG") -> str:
#     """
#     画像(numpy配列)をBase64文字列にエンコード

#     Args:
#         image: OpenCV形式の画像 (BGR, numpy.ndarray)
#         format: 出力フォーマット ("PNG" or "JPEG")

#     Returns:
#         str: Base64エンコードされた画像文字列
#     """

#     # BGR -> RGB変換
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # PIL Imageに変換
#     image_pil = Image.fromarray(image_rgb)

#     # バイトストリームに書き込み
#     buffer = BytesIO()
#     image_pil.save(buffer, format=format)
#     buffer.seek(0)

#     # Base64エンコード
#     image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

#     return image_base64


# 【画像エンコード形式 後】JPEG形式で圧縮率を上げエンコード時間を98%削減
def encode_image_to_base64(
    image: np.ndarray, format: str = "JPEG", quality: int = 85
) -> str:
    """
    画像(numpy配列)をBase64文字列にエンコード

    Args:
        image: OpenCV形式の画像 (BGR, numpy.ndarray)
        format: 出力フォーマット ("PNG" or "JPEG")
        quality: JPEG品質 (1-100、デフォルト85)

    Returns:
        str: Base64エンコードされた画像文字列
    """
    # BGR -> RGB変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # PIL Imageに変換
    image_pil = Image.fromarray(image_rgb)

    # バイトストリームに書き込み
    buffer = BytesIO()
    if format == "JPEG":
        image_pil.save(buffer, format=format, quality=quality)
    else:
        image_pil.save(buffer, format=format)
    buffer.seek(0)

    # Base64エンコード
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return image_base64


def process_yolo_detection(
    image, conf_threshold: float = 0.25, iou_threshold: float = 0.45
):
    """
    YOLO物体検出を実行

    Args:
        image: 入力画像 (BGR, numpy.ndarray)
        conf_threshold: 信頼度の閾値
        iou_threshold: IoUの閾値

    Returns:
        tuple[np.ndarray, List[Dict], Dict[str, float]]: (検出結果画像, 検出オブジェクトリスト, 処理時間内訳)
    """

    timing = {}

    # モデルを取得
    yolo_model = initialize_model()

    # YOLO推論を実行（時間計測）
    # 【YOLO推論の最適化 前】
    # inference_start = time.time()
    # results = yolo_model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    # inference_end = time.time()
    # timing["inference_ms"] = (inference_end - inference_start) * 1000
    # 【YOLO推論の最適化 後】推論モードで勾配計算を無効化して高速化
    inference_start = time.time()
    with torch.inference_mode():
        results = yolo_model(
            image, conf=conf_threshold, iou=iou_threshold, verbose=False
        )
    inference_end = time.time()
    timing["inference_ms"] = (inference_end - inference_start) * 1000

    # 結果を描画（時間計測）
    plot_start = time.time()
    annotated_image = results[0].plot()
    plot_end = time.time()
    timing["plot_ms"] = (plot_end - plot_start) * 1000

    # 検出されたオブジェクトのリストを作成（時間計測）
    detection_list_start = time.time()
    detections = []
    for box in results[0].boxes:
        detection = {
            "class_id": int(box.cls[0]),
            "class_name": yolo_model.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
        }
        detections.append(detection)
    detection_list_end = time.time()
    timing["detection_list_ms"] = (detection_list_end - detection_list_start) * 1000

    return annotated_image, detections, timing


def lambda_handler(event, context):
    """
    Lambda関数ハンドラー - シンプルYOLOサンプル

    Args:
        event: Lambdaイベント
            {
                "image": "base64エンコードされた画像",
                "conf_threshold": 0.25 (オプション),
                "iou_threshold": 0.45 (オプション)
            }
        context: Lambda実行コンテキスト

    Returns:
        Dict[str, Any]: レスポンス
            {
                "statusCode": 200,
                "body": {
                    "annotatedImage": "base64エンコードされた検出結果画像",
                    "detections": [
                        {
                            "class_id": 0,
                            "class_name": "person",
                            "confidence": 0.95,
                            "bbox": [x1, y1, x2, y2]
                        },
                        ...
                    ],
                    "summary": {
                        "total_detections": 5,
                        "classes_detected": ["person", "car"]
                    }
                }
            }
    """
    try:
        print("Lambda function started")
        print(f"Event keys: {event.keys()}")

        # 処理時間の内訳を記録
        timing_breakdown = {}

        # 入力パラメータを取得
        if "image" not in event:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "入力パラメータ 'image' が必要です"}),
            }

        image_base64 = event["image"]

        # 閾値を取得（優先順位: event > 環境変数 > デフォルト値）
        conf_threshold = float(
            event.get("conf_threshold", os.environ.get("CONF_THRESHOLD", "0.25"))
        )
        iou_threshold = float(
            event.get("iou_threshold", os.environ.get("IOU_THRESHOLD", "0.45"))
        )

        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")

        # Base64デコード（時間計測）
        decode_start = time.time()
        image = decode_base64_image(image_base64)
        decode_end = time.time()
        timing_breakdown["decode_ms"] = (decode_end - decode_start) * 1000
        print(f"Image shape: {image.shape}")
        print(f"Decode time: {timing_breakdown['decode_ms']:.2f} ms")

        # YOLO物体検出を実行
        print("Processing image with YOLO...")
        yolo_start = time.time()
        annotated_image, detections, yolo_timing = process_yolo_detection(
            image, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )
        yolo_end = time.time()
        timing_breakdown["yolo_total_ms"] = (yolo_end - yolo_start) * 1000
        timing_breakdown.update(yolo_timing)  # inference_ms, plot_ms, detection_list_ms
        print(f"Total detections: {len(detections)}")
        print(f"YOLO total time: {timing_breakdown['yolo_total_ms']:.2f} ms")
        print(f"  - Inference: {yolo_timing['inference_ms']:.2f} ms")
        print(f"  - Plot: {yolo_timing['plot_ms']:.2f} ms")
        print(f"  - Detection list: {yolo_timing['detection_list_ms']:.2f} ms")

        # 検出結果画像をBase64エンコード（時間計測）
        print("Encoding annotated image to base64...")
        encode_start = time.time()
        annotated_image_base64 = encode_image_to_base64(annotated_image)
        encode_end = time.time()
        timing_breakdown["encode_ms"] = (encode_end - encode_start) * 1000
        print(f"Encode time: {timing_breakdown['encode_ms']:.2f} ms")

        # サマリー情報を作成（時間計測）
        summary_start = time.time()
        classes_detected = list(set([d["class_name"] for d in detections]))
        summary = {
            "total_detections": len(detections),
            "classes_detected": classes_detected,
        }
        summary_end = time.time()
        timing_breakdown["summary_ms"] = (summary_end - summary_start) * 1000

        # レスポンスを返す
        response = {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "annotatedImage": annotated_image_base64,
                    "detections": detections,
                    "summary": summary,
                    "inference_time_ms": yolo_timing["inference_ms"],
                    "timing_breakdown": timing_breakdown,
                }
            ),
        }

        print("Lambda function completed successfully")
        print(f"Timing breakdown: {timing_breakdown}")
        return response

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()

        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "type": type(e).__name__}),
        }
