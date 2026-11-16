"""Testes unitários de segmentação de caracteres."""

from __future__ import annotations

import numpy as np

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
