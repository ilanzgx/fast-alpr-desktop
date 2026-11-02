"""Testes unitários das funções de pré-processamento de imagem."""

import numpy as np

from src.config import Caixa
from src.preprocessing import (
    aplicar_clahe,
    binarizar,
    converter_cinza,
    corrigir_perspectiva,
    recortar_roi,
    suavizar_gauss,
    verificar_distorcao_geometrica,
)


def test_converter_cinza():
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr[:, :] = [100, 150, 200]
    cinza = converter_cinza(bgr)
    assert cinza.ndim == 2
    assert cinza.shape == (100, 100)


def test_suavizar_gauss():
    cinza = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    suave = suavizar_gauss(cinza, k=5)
    assert suave.shape == cinza.shape
    # Testa normalização de kernel par para ímpar
    suave_par = suavizar_gauss(cinza, k=4)
    assert suave_par.shape == cinza.shape


def test_aplicar_clahe():
    cinza = np.full((64, 64), 128, dtype=np.uint8)
    cinza[10:30, 10:30] = 50
    clahe = aplicar_clahe(cinza, clip=2.0, grade=8)
    assert clahe.shape == cinza.shape
    assert clahe.dtype == np.uint8


def test_binarizar_inversao_polaridade():
    # Placa clara (255) com texto escuro (0)
    img_fundo_claro = np.full((60, 120), 220, dtype=np.uint8)
    img_fundo_claro[20:40, 30:90] = 30  # texto escuro

    binaria = binarizar(img_fundo_claro, modo="otsu", inverter_se_fundo_claro=True)
    assert binaria.shape == img_fundo_claro.shape
    # O texto (interior) deve ter virado branco (255)
    assert binaria[30, 60] == 255
    # O fundo deve ter virado preto (0)
    assert binaria[5, 5] == 0


def test_recortar_roi():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    caixa = Caixa(x1=50, y1=50, x2=150, y2=100)
    # Com margem 0.04: pad_x = 4, pad_y = 2
    roi = recortar_roi(img, caixa, margem=0.04)
    assert roi.shape[0] == (50 + 4)  # altura 50 + 2*2
    assert roi.shape[1] == (100 + 8)  # largura 100 + 2*4


def test_verificar_distorcao_geometrica():
    # Retângulo ortogonal perfeito
    pts_reto = np.array([[10, 10], [110, 10], [110, 50], [10, 50]], dtype=np.float32)
    distorcido, inc_h, cis_v = verificar_distorcao_geometrica(pts_reto, threshold_graus=2.5)
    assert not distorcido
    assert inc_h < 0.1
    assert cis_v < 0.1

    # Quadrilátero com cisalhamento e inclinação em diagonal (~10 graus)
    pts_diagonal = np.array([[10, 10], [115, 25], [130, 70], [25, 55]], dtype=np.float32)
    distorcido, inc_h, cis_v = verificar_distorcao_geometrica(pts_diagonal, threshold_graus=2.5)
    assert distorcido
    assert inc_h > 2.5 or cis_v > 2.5


def test_corrigir_perspectiva():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    vertices = np.array([[10, 10], [110, 15], [120, 60], [20, 55]], dtype=np.float32)
    retificada = corrigir_perspectiva(img, vertices, largura=400, altura=130)
    assert retificada.shape == (130, 400, 3)
