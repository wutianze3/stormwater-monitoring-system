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
    kept = []
    for class_id in range(len(LABELS)):
        indices = np.flatnonzero(classes == class_id)
        if len(indices):
            local = cv2.dnn.NMSBoxes(boxes[indices].tolist(), scores[indices].tolist(), threshold, 0.5)
            kept.extend(indices[np.asarray(local).reshape(-1)].tolist())
    detections = []
    for i in sorted(kept, key=lambda j: float(scores[j]), reverse=True)[:30]:
        x, y, w, h = boxes[i]
        x1, y1 = max(0., (x-left)/scale/width), max(0., (y-top)/scale/height)
        x2, y2 = min(1., (x+w-left)/scale/width), min(1., (y+h-top)/scale/height)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(dict(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                               confidence=float(scores[i]), label=LABELS[classes[i]],
                               areaRatio=float((x2-x1)*(y2-y1))))
    return detections, round((perf_counter()-start)*1000, 1)
