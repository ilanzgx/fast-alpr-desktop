"""Testes unitários de segmentação de caracteres e normalização geométrica para OCR."""

from __future__ import annotations

import cv2
import numpy as np

from src.ocr import _deslantar_caractere, _estimar_inclinacao_caracteres
from src.segmentation import segmentar_caracteres


def test_segmentar_caracteres_isola_mascara_sem_vazamento():
    # Dois componentes desconexos cujos bounding boxes se sobrepõem
    img = np.zeros((100, 300), dtype=np.uint8)

    # Componente A: formato "L" largo na base (x=40..72, y=25..75)
    img[25:75, 40:47] = 255
    img[69:75, 40:72] = 255

    # Componente B: haste vertical dentro do bbox de A sem encostar em A (x=56..64, y=25..60)
    img[25:60, 56:64] = 255

    chars = segmentar_caracteres(img)
    assert len(chars) == 2

    # Ao isolar apenas a máscara do componente A, o quadrante superior direito fica vazio (0)
    char_a = chars[0]
    assert char_a.shape == (60, 40)
    assert np.count_nonzero(char_a[5:32, 22:36]) == 0


def test_segmentar_caracteres_descarta_fragmentos_de_canto():
    img = np.zeros((100, 300), dtype=np.uint8)

    # Artefato grande preso no canto superior esquerdo (toca borda lateral e vertical)
    img[0:48, 0:18] = 255

    # Caractere real no interior da placa
    img[25:75, 110:124] = 255

    chars = segmentar_caracteres(img)
    assert len(chars) == 1


def test_deslantar_caractere_corrige_cisalhamento_sem_cortar_altura():
    # Cria 4 glifos sintéticos (40x60) inclinados com cisalhamento conhecido (~0.25)
    caracteres: list[np.ndarray] = []
    for _ in range(4):
        tile = np.zeros((60, 40), dtype=np.uint8)
        pts = np.array([[22, 10], [27, 10], [17, 50], [12, 50]], dtype=np.int32)
        cv2.fillPoly(tile, [pts], 255)
        caracteres.append(tile)

    skew_estimado = _estimar_inclinacao_caracteres(caracteres)
    assert abs(skew_estimado) > 0.15

    deslantado = _deslantar_caractere(caracteres[0], skew_estimado)
    # Altura original mantida e largura nunca reduzida abaixo de 40px
    assert deslantado.shape[0] == 60
    assert deslantado.shape[1] >= 40
    # Polaridade para Tesseract (texto escuro sobre fundo claro)
    assert float(np.mean(deslantado)) > 200.0

    # Verifica redução expressiva do cisalhamento residual
    fg_pos = (deslantado < 128).astype(np.uint8) * 255
    m_pos = cv2.moments(fg_pos)
    skew_pos = abs(m_pos["mu11"] / m_pos["mu02"])
    assert skew_pos < 0.06
