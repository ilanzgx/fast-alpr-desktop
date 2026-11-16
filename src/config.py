"""Configurações, caminhos e contratos de dados do pipeline ALPR."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Diretórios base do projeto (suporta execução normal e empacotada via PyInstaller)
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ROOT_DIR = Path(sys._MEIPASS).resolve()
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
TOOLS_DIR = ROOT_DIR / "tools"

# Caminhos padrão configuráveis
DEFAULT_ONNX_PATH = MODELS_DIR / "placas.onnx"


def obter_caminho_modelo() -> Path:
    """Retorna o caminho do modelo ONNX, respeitando a variável de ambiente MODELO_ONNX."""
    caminho_env = os.environ.get("MODELO_ONNX")
    if caminho_env:
        return Path(caminho_env)
    return DEFAULT_ONNX_PATH


@dataclass(frozen=True)
class Caixa:
    """Representa a caixa delimitadora (bounding box) da placa."""

    x1: int
    y1: int
    x2: int
    y2: int
    confianca: float = 1.0

    @property
    def largura(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def altura(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass
class ParametrosPipeline:
    """Parâmetros ajustáveis de todas as etapas do processamento."""

    # Detecção (E1)
    metodo: str = "yolo"  # "yolo" | "classico"
    confianca_min: float = 0.25
    iou_nms: float = 0.45
    canny_min: int = 30
    canny_max: int = 200

    # Recorte (E2)
    margem_roi: float = 0.04  # 4% de margem

    # Pré-processamento (E4-E7)
    kernel_gauss: int = 5
    clahe_clip: float = 2.0
    clahe_grade: int = 8
    binarizacao: str = "otsu"  # "otsu" | "adaptativa"
    bloco_adaptativo: int = 31
    constante_adaptativa: int = 10

    # OCR e Pós-processamento (E9-E10)
    ocr_engine: str = "fast_plate_ocr"  # "fast_plate_ocr" | "tesseract"
    ocr_psm: int = 7
    modo_livre: bool = False
    leitura_robusta: bool = False


@dataclass
class ResultadoPipeline:
    """Estrutura com todas as saídas intermediárias e finais do pipeline."""

    original: np.ndarray
    caixa: Caixa | None = None
    anotada: np.ndarray | None = None
    bordas: np.ndarray | None = None
    roi: np.ndarray | None = None
    roi_cinza: np.ndarray | None = None
    roi_clahe: np.ndarray | None = None
    binarizada: np.ndarray | None = None
    caracteres: list[np.ndarray] = field(default_factory=list)
    texto_bruto: str = ""
    texto_final: str = ""
    formato: str = "invalido"  # "antigo" | "mercosul" | "invalido" | "livre"
    confianca_ocr: float | None = None
    substituicoes: int = 0
    tempos_ms: dict[str, float] = field(default_factory=dict)
    erro: str | None = None
