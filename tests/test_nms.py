"""Teste de equivalência do NMS próprio contra o cv2.dnn.NMSBoxes do OpenCV."""

import cv2
import numpy as np

from src.detection.yolo_onnx import nms


def test_nms_proprio_contra_opencv():
    # Conjunto sintético de caixas com sobreposição proposital
    caixas = np.array(
        [
            [10.0, 10.0, 100.0, 60.0],  # Caixa A (Score alto)
            [12.0, 11.0, 98.0, 59.0],   # Caixa B (Quase idêntica a A, score menor)
            [200.0, 50.0, 320.0, 110.0], # Caixa C (Distante, não sobrepõe)
            [205.0, 52.0, 318.0, 108.0], # Caixa D (Sobreposta a C)
            [15.0, 14.0, 95.0, 55.0],   # Caixa E (Sobreposta a A)
        ],
        dtype=np.float32,
    )
    scores = np.array([0.92, 0.85, 0.78, 0.65, 0.40], dtype=np.float32)
    iou_thresh = 0.45

    # 1. Nosso NMS
    mantidos_proprio = nms(caixas, scores, iou_max=iou_thresh)

    # 2. OpenCV NMSBoxes (recebe [x, y, w, h])
    caixas_xywh = []
    for c in caixas:
        caixas_xywh.append([float(c[0]), float(c[1]), float(c[2] - c[0]), float(c[3] - c[1])])

    indices_cv = cv2.dnn.NMSBoxes(
        bboxes=caixas_xywh,
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_thresh,
    )

    if hasattr(indices_cv, "flatten"):
        mantidos_cv = indices_cv.flatten().tolist()
    else:
        mantidos_cv = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices_cv]

    assert sorted(mantidos_proprio) == sorted(mantidos_cv)
    # Espera-se manter exatamente a Caixa A (0) e a Caixa C (2)
    assert 0 in mantidos_proprio
    assert 2 in mantidos_proprio
    assert len(mantidos_proprio) == 2


def test_nms_vazio():
    vazio = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    assert nms(vazio, scores) == []
