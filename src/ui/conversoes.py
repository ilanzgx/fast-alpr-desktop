"""Conversões entre formatos de imagem NumPy (OpenCV) e Qt (QImage / QPixmap)."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def numpy_para_pixmap(img: np.ndarray | None) -> QPixmap:
    """Converte uma imagem NumPy (BGR ou monocromática) para QPixmap.
    
    Usa .copy() para garantir que a memória do buffer não seja liberada pelo GC do NumPy
    enquanto o Qt estiver exibindo a imagem.
    """
    if img is None or img.size == 0:
        return QPixmap()

    if img.ndim == 2:  # Tons de cinza ou binária
        img_contig = np.ascontiguousarray(img)
        h, w = img_contig.shape
        qimg = QImage(img_contig.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif img.ndim == 3:  # BGR para RGB
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w, c = rgb.shape
        bytes_por_linha = c * w
        qimg = QImage(rgb.data, w, h, bytes_por_linha, QImage.Format.Format_RGB888)
    else:
        return QPixmap()

    return QPixmap.fromImage(qimg.copy())
