"""Local ONNX waste detector; no cloud image transfer or PyTorch required."""
import os
import threading
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

LABELS = ['paper', 'plastic', 'metal', 'organic', 'other']
_session = None
_lock = threading.Lock()


def model_path():
    return Path(os.getenv('WASTE_MODEL_PATH', str(Path(__file__).parent / 'weights' / 'trashscan8s.onnx')))


def inference_windows(width, height):
    """Full view plus overlapping crops; bounded work on a Raspberry Pi."""
    windows = [(0, 0, width, height)]
    if os.getenv('WASTE_TILES', '1') != '1' or min(width, height) < 480:
        return windows
    side = min(min(width, height), max(384, round(min(width, height) * .70)))
    for y in (0, height - side):
        for x in (0, width - side):
            windows.append((x, y, x + side, y + side))
    return list(dict.fromkeys(windows))


def filter_candidate(box, confidence, label, window, image_size):
    """Reject scene-sized predictions, padding and truncated crop objects.

    These are conservative deployment rules, not evidence that a region is water.
    Large real objects can be rejected too; expose limits as environment settings.
    """
    x1, y1, x2, y2 = box
    ox, oy, ex, ey = window
    width, height = image_size
    cw, ch = ex-ox, ey-oy
    minimum = float(os.getenv('WASTE_CONFIDENCE', '0.40'))
    if label == 'paper':
        minimum = max(minimum, float(os.getenv('WASTE_PAPER_CONFIDENCE', '0.70')))
    if not np.isfinite([*box, confidence]).all() or confidence < minimum:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    # A crop-edge object is evaluated from the full view or an overlapping crop.
    if ((ox > 0 and x1 < 3) or (oy > 0 and y1 < 3)
            or (ex < width and x2 > cw-3) or (ey < height and y2 > ch-3)):
        return None
    x1, y1, x2, y2 = max(0., x1), max(0., y1), min(cw, x2), min(ch, y2)
    bw, bh = x2-x1, y2-y1
    if bw < 4 or bh < 4:
        return None
    area = bw*bh/(width*height)
    if area > float(os.getenv('WASTE_MAX_AREA', '0.18')):
        return None
    if bw*bh/(cw*ch) > .55 or bw/cw > .95 or bh/ch > .95:
        return None
    return dict(x1=(x1+ox)/width, y1=(y1+oy)/height,
                x2=(x2+ox)/width, y2=(y2+oy)/height,
                confidence=float(confidence), label=label, areaRatio=area)


def detect(frame):
    global _session
    import onnxruntime as ort
    with _lock:
        if _session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(os.getenv('WASTE_THREADS', '2')))
            _session = ort.InferenceSession(str(model_path()), sess_options=options, providers=['CPUExecutionProvider'])
    start = perf_counter()
    height, width = frame.shape[:2]
    candidates = []
    # Serialize inference to bound CPU load if several camera clients connect.
    with _lock:
        for window in inference_windows(width, height):
            x, y, right, bottom = window
            candidates.extend(predict_crop(frame[y:bottom, x:right], window, (width, height)))
    if not candidates:
        return [], round((perf_counter()-start)*1000, 1)
    boxes = [[d['x1'], d['y1'], d['x2']-d['x1'], d['y2']-d['y1']] for d in candidates]
    # Cross-class NMS: one physical object should not be counted twice.
    kept = cv2.dnn.NMSBoxes(boxes, [d['confidence'] for d in candidates], 0., .45)
    detections = [candidates[int(i)] for i in np.asarray(kept).reshape(-1)[:30]]
    return detections, round((perf_counter()-start)*1000, 1)


def predict_crop(frame, window, image_size):
    height, width = frame.shape[:2]
    scale = min(640 / width, 640 / height)
    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    left, top = (640 - resized.shape[1]) // 2, (640 - resized.shape[0]) // 2
    padded = np.full((640, 640, 3), 114, np.uint8)
    padded[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    output = _session.run(None, {_session.get_inputs()[0].name: tensor})[0]
    if output.shape != (1, 9, 8400):
        raise ValueError(f'Unsupported model output: {output.shape}')
    rows = output[0].T
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), classes + 4]
    threshold = float(os.getenv('WASTE_CONFIDENCE', '0.40'))
    selected = scores >= threshold
    rows, scores, classes = rows[selected], scores[selected], classes[selected]
    boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2, rows[:, 1] - rows[:, 3] / 2, rows[:, 2], rows[:, 3]))
    kept = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, 0.5) if len(boxes) else []
    detections = []
    for i in np.asarray(kept).reshape(-1):
        x, y, w, h = boxes[i]
        box = tuple(float(v) for v in ((x-left)/scale, (y-top)/scale, (x+w-left)/scale, (y+h-top)/scale))
        candidate = filter_candidate(box, float(scores[i]), LABELS[classes[i]], window, image_size)
        if candidate:
            detections.append(candidate)
    return detections
