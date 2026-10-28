"""Módulo de detecção de placas veiculares (métodos clássico e YOLO ONNX)."""

from src.detection.contours import detectar_placa_classica
from src.detection.yolo_onnx import DetectorYolo, nms

__all__ = ["DetectorYolo", "detectar_placa_classica", "nms"]
