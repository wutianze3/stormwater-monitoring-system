"""Lightweight OpenCV visual screening for Raspberry Pi camera images.

This is a deterministic demonstration algorithm, not a trained ML model or a
water-safety test. It only reports visible colour and contour anomalies.
"""

import os

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import numpy as np


router = APIRouter(prefix="/api/vision", tags=["vision"])

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_ANALYSIS_SIDE = int(os.getenv("OPENCV_MAX_SIDE", "960"))
MAX_DETECTIONS = int(os.getenv("OPENCV_MAX_DETECTIONS", "3"))
MIN_SHAPE_CONFIDENCE = float(os.getenv("OPENCV_MIN_SHAPE_CONFIDENCE", "0.64"))


def resize_for_analysis(image: np.ndarray):
    height, width = image.shape[:2]
    largest = max(width, height)
    if largest <= MAX_ANALYSIS_SIDE:
        return image, 1.0
    scale = MAX_ANALYSIS_SIDE / largest
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def water_focus_mask(height: int, width: int):
    """Use a conservative central/lower trapezoid where water is expected."""
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [
            [round(width * 0.14), round(height * 0.30)],
            [round(width * 0.86), round(height * 0.30)],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def build_candidate_masks(image: np.ndarray, focus: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    bright = ((saturation < 45) & (value > 220)).astype(np.uint8) * 255
    synthetic_hue = (hue <= 12) | (hue >= 145)
    vivid = (
        (saturation > 145) & (value > 90) & synthetic_hue
    ).astype(np.uint8) * 255
    blue_object = (
        (hue >= 98)
        & (hue <= 120)
        & (saturation > 65)
        & (value > 50)
    ).astype(np.uint8) * 255

    vegetation = (
        (hue >= 30) & (hue <= 95) & (saturation > 45)
    ).astype(np.uint8) * 255
    vegetation = cv2.dilate(
        vegetation, np.ones((5, 5), np.uint8), iterations=1
    )
    not_vegetation = cv2.bitwise_not(vegetation)
    bright = cv2.bitwise_and(bright, not_vegetation)
    vivid = cv2.bitwise_and(vivid, not_vegetation)
    blue_object = cv2.bitwise_and(blue_object, not_vegetation)

    def clean(candidate: np.ndarray):
        candidate = cv2.bitwise_and(candidate, focus)
        candidate = cv2.medianBlur(candidate, 3)
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        return cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

    return {
        "primary": clean(cv2.bitwise_or(bright, vivid)),
        "compact_blue": clean(blue_object),
    }, hsv


def contour_detections(
    masks: dict[str, np.ndarray],
    original_width: int,
    original_height: int,
    scale: float,
):
    height, width = next(iter(masks.values())).shape
    frame_area = width * height
    minimum_area = max(55, frame_area * 0.00035)
    ranked = []

    for mask_name, mask in masks.items():
        maximum_ratio = 0.003 if mask_name == "compact_blue" else 0.012
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if not minimum_area <= area <= frame_area * maximum_ratio:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 7 or box_height < 7:
                continue
            aspect_ratio = box_width / box_height
            if not 0.28 <= aspect_ratio <= 3.4:
                continue
            border_x = round(width * 0.015)
            border_y = round(height * 0.015)
            if (
                x <= border_x
                or x + box_width >= width - border_x
                or y + box_height >= height - border_y
            ):
                continue
            rectangularity = area / max(1, box_width * box_height)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / max(1.0, hull_area)
            if rectangularity < 0.32 or solidity < 0.62:
                continue
            area_ratio = area / frame_area
            confidence = min(
                0.90,
                0.42
                + rectangularity * 0.18
                + solidity * 0.16
                + min(area_ratio * 8, 0.12),
            )
            if confidence < MIN_SHAPE_CONFIDENCE:
                continue
            ranked.append(
                (confidence, area, x, y, box_width, box_height)
            )

    detections = []
    for confidence, area, x, y, box_width, box_height in sorted(
        ranked, reverse=True
    )[:MAX_DETECTIONS]:
        x1, y1 = x / scale, y / scale
        x2, y2 = (x + box_width) / scale, (y + box_height) / scale
        detections.append(
            {
                "x1": round(max(0.0, x1 / original_width), 5),
                "y1": round(max(0.0, y1 / original_height), 5),
                "x2": round(min(1.0, x2 / original_width), 5),
                "y2": round(min(1.0, y2 / original_height), 5),
                "confidence": round(float(confidence), 4),
                "label": "visual anomaly",
                "areaRatio": round(float(area / frame_area), 5),
            }
        )
    return detections


def scene_features(
    image: np.ndarray,
    hsv: np.ndarray,
    focus: np.ndarray,
    detections: list[dict],
):
    hue, saturation, value = cv2.split(hsv)
    focused = focus > 0
    pixel_count = max(1, int(np.count_nonzero(focused)))
    brown = (
        focused
        & (hue >= 5)
        & (hue <= 28)
        & (saturation > 45)
        & (value > 35)
    )
    brown_ratio = np.count_nonzero(brown) / pixel_count

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    focused_values = grey[focused]
    contrast = float(np.std(focused_values)) if focused_values.size else 0.0
    turbidity = min(
        1.0, brown_ratio * 1.7 + max(0.0, 18.0 - contrast) / 100.0
    )
    discoloration = min(1.0, brown_ratio * 1.5)
    detected_area = sum(item["areaRatio"] for item in detections)
    debris = min(1.0, len(detections) / 4.0 + detected_area * 4.0)
    quality = min(1.0, max(0.15, contrast / 48.0))
    return turbidity, debris, discoloration, quality


@router.get("/health")
def vision_health():
    return {
        "status": "ready",
        "name": "OpenCV threshold screening",
        "runtime": f"OpenCV {cv2.__version__}",
        "maxAnalysisSide": MAX_ANALYSIS_SIDE,
        "maxDetections": MAX_DETECTIONS,
    }


@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    context: str = Form("stormwater"),
):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG, and WEBP images are supported",
        )

    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="Image exceeds the 12 MB limit"
        )
    frame = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    original_height, original_width = frame.shape[:2]
    analysis_frame, scale = resize_for_analysis(frame)
    height, width = analysis_frame.shape[:2]
    focus = water_focus_mask(height, width)
    candidate_masks, hsv = build_candidate_masks(analysis_frame, focus)
    detections = contour_detections(
        candidate_masks, original_width, original_height, scale
    )
    turbidity, debris, discoloration, quality = scene_features(
        analysis_frame, hsv, focus, detections
    )

    score = round(
        min(
            95,
            max(
                5,
                8
                + turbidity * 34
                + debris * 38
                + discoloration * 20,
            ),
        )
    )
    evidence = max(turbidity, debris, discoloration)
    confidence = round(58 + evidence * 30)

    return {
        "score": score,
        "confidence": confidence,
        "source": "opencv_threshold_v2_conservative",
        "method": "deterministic_opencv_not_trained_ml",
        "context": context,
        "detectionCount": len(detections),
        "detections": detections,
        "factors": {
            "turbidity": round(turbidity, 4),
            "debris": round(debris, 4),
            "discoloration": round(discoloration, 4),
            "quality": round(quality, 4),
        },
        "limitations": (
            "Visual screening only; does not establish water safety or "
            "identify dissolved or microbial contamination."
        ),
    }
