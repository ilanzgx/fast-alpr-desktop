"""Testes unitários para o módulo de OCR (FastPlateOCR e Tesseract)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.ocr import (
    reconhecer_com_fastplate,
    reconhecer_placa,
    verificar_fastplate_disponivel,
)


def test_verificar_fastplate_disponivel() -> None:
    disponivel, msg = verificar_fastplate_disponivel()
    assert disponivel is True
    assert "FastPlateOCR" in msg


def test_reconhecer_fastplate_entradas_invalidas() -> None:
    # None
    texto, conf, erro = reconhecer_com_fastplate(None)  # type: ignore[arg-type]
    assert texto == ""
    assert conf is None
    assert erro is not None

    # Array vazio
    vazio = np.zeros((0, 0, 3), dtype=np.uint8)
    texto, conf, erro = reconhecer_com_fastplate(vazio)
    assert texto == ""
    assert conf is None
    assert erro is not None

    # Imagem muito pequena
    pequena = np.zeros((2, 2, 3), dtype=np.uint8)
    texto, conf, erro = reconhecer_com_fastplate(pequena)
    assert texto == ""
    assert conf is None
    assert erro is not None


def test_reconhecer_fastplate_imagem_real_ou_sintetica() -> None:
    caminho_amostra = Path("data/teste/recortes/ALE-8602.jpg")
    if caminho_amostra.exists():
        img_bgr = cv2.imread(str(caminho_amostra))
        assert img_bgr is not None
        texto, conf, erro = reconhecer_com_fastplate(img_bgr)
        assert erro is None
        assert texto == "ALE8602"
        assert conf is not None
        assert conf > 0.90

        # Testa entrada em tons de cinza (2D)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        texto_g, conf_g, erro_g = reconhecer_com_fastplate(img_gray)
        assert erro_g is None
        assert texto_g == "ALE8602"
        assert conf_g is not None
    else:
        # Imagem sintética mínima
        sintetica = np.full((70, 200, 3), 255, dtype=np.uint8)
        texto, conf, erro = reconhecer_com_fastplate(sintetica)
        assert erro is None


def test_reconhecer_placa_com_engine_fastplate() -> None:
    caminho_amostra = Path("data/teste/recortes/ALE-8602.jpg")
    if caminho_amostra.exists():
        roi = cv2.imread(str(caminho_amostra))
        assert roi is not None
        dummy = np.zeros((70, 200), dtype=np.uint8)
        texto, conf, erro = reconhecer_placa(
            roi_cinza=dummy,
            roi_clahe=dummy,
            binarizada=dummy,
            engine="fast_plate_ocr",
            roi=roi,
        )
        assert erro is None
        assert texto == "ALE8602"
        assert conf is not None
        assert conf > 0.90
