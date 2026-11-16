"""Testes unitários das funções de pré-processamento de imagem."""

import cv2
import numpy as np

from src.config import Caixa
from src.preprocessing import (
    aplicar_clahe,
    binarizar,
    converter_cinza,
    corrigir_inclinacao,
    corrigir_perspectiva,
    encontrar_vertices_placa_roi,
    estimar_angulo_inclinacao,
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


def test_corrigir_perspectiva_zoom_out_e_entorno_real():
    # Entorno cinza médio (180) ao redor de placa branca para validar ausência de borda preta
    img = np.full((180, 360, 3), 180, dtype=np.uint8)
    img[40:120, 60:300] = (245, 245, 245)
    vertices = np.array([[60, 40], [300, 40], [300, 120], [60, 120]], dtype=np.float32)

    ret = corrigir_perspectiva(img, vertices, largura=None, altura=130, margem_saida=0.10)
    assert ret.shape[0] > 130
    assert ret.shape[1] > 390
    # Margem zoom-out externa preserva pixels reais (não preenche com preto puro 0)
    assert int(ret[2, 2].min()) > 100
    assert int(ret[ret.shape[0] // 2, ret.shape[1] // 2].mean()) > 230


def test_corrigir_inclinacao_standalone_e_com_contexto():
    base = np.full((260, 480, 3), 140, dtype=np.uint8)
    # Desenha placa branca com faixa azul superior e 7 barras escuras simulando caracteres
    base[88:172, 110:370] = (240, 240, 240)
    base[88:106, 110:370] = (180, 80, 20)
    for idx in range(7):
        cx = 132 + idx * 34
        base[114:162, cx : cx + 14] = (30, 30, 30)

    centro = (240.0, 130.0)
    m_rot = cv2.getRotationMatrix2D(centro, 15.0, 1.0)
    rotacionada = cv2.warpAffine(
        base, m_rot, (480, 260), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    # Localiza bounding box exato da placa rotacionada para simular caixa YOLO + contexto
    mask_placa = np.any(np.abs(rotacionada.astype(np.int16) - 140) > 25, axis=2)
    ys, xs = np.where(mask_placa)
    caixa = Caixa(x1=int(xs.min()), y1=int(ys.min()), x2=int(xs.max()), y2=int(ys.max()))

    corrigida_ctx, ang_ctx = corrigir_inclinacao(
        rotacionada, margem_zoom_out=0.09, caixa_interna=caixa
    )
    assert abs(ang_ctx) > 8.0
    # BORDER_REPLICATE preserva o entorno cinza (140) nos 4 cantos sem gerar triângulos pretos (0)
    cantos_ctx = [
        corrigida_ctx[0, 0],
        corrigida_ctx[0, -1],
        corrigida_ctx[-1, 0],
        corrigida_ctx[-1, -1],
    ]
    assert all(int(c.min()) > 100 for c in cantos_ctx)
    # Inclinação residual corrigida e folga vertical mantendo faixa/caracteres longe das bordas
    assert abs(estimar_angulo_inclinacao(converter_cinza(corrigida_ctx))) < 3.0
    ys_txt = np.where(converter_cinza(corrigida_ctx) < 90)[0]
    assert int(ys_txt.min()) >= 6
    assert (corrigida_ctx.shape[0] - 1 - int(ys_txt.max())) >= 6

    # Modo standalone (recorte sem folga externa prévia)
    sub = rotacionada[caixa.y1 - 8 : caixa.y2 + 8, caixa.x1 - 8 : caixa.x2 + 8]
    corrigida_solo, ang_solo = corrigir_inclinacao(sub, margem_zoom_out=0.09)
    assert abs(ang_solo) > 8.0
    cantos_solo = [
        corrigida_solo[0, 0],
        corrigida_solo[0, -1],
        corrigida_solo[-1, 0],
        corrigida_solo[-1, -1],
    ]
    assert all(int(c.min()) > 100 for c in cantos_solo)
    assert abs(estimar_angulo_inclinacao(converter_cinza(corrigida_solo))) < 3.0

    # Ângulo pequeno (3.5°): valida que a altura estimada com contexto não encolhe a faixa azul
    m_rot_peq = cv2.getRotationMatrix2D(centro, 3.5, 1.0)
    rot_peq = cv2.warpAffine(
        base, m_rot_peq, (480, 260), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    mask_peq = np.any(np.abs(rot_peq.astype(np.int16) - 140) > 25, axis=2)
    ys_p, xs_p = np.where(mask_peq)
    caixa_peq = Caixa(
        x1=int(xs_p.min()), y1=int(ys_p.min()), x2=int(xs_p.max()), y2=int(ys_p.max())
    )
    corr_peq, ang_peq = corrigir_inclinacao(
        rot_peq, margem_zoom_out=0.09, caixa_interna=caixa_peq
    )
    assert abs(ang_peq) > 2.0
    # Canal azul da barra superior e entorno acima dela preservados com folga
    dif_azul = corr_peq[:, :, 0].astype(np.int16) - corr_peq[:, :, 2].astype(np.int16)
    ys_topo = np.where(dif_azul > 80)[0]
    assert int(ys_topo.min()) >= 6


def test_binarizar_faixa_central_com_entorno_escuro():
    # Mais de 55% da imagem é para-choque escuro (20), com placa clara (220) no centro
    img = np.full((120, 400), 20, dtype=np.uint8)
    img[24:96, 55:345] = 220
    img[38:82, 120:136] = 30
    img[38:82, 200:216] = 30

    binaria = binarizar(img, modo="otsu", inverter_se_fundo_claro=True)
    # Caractere vira branco (255) e fundo interno da placa vira preto (0)
    assert binaria[60, 128] == 255
    assert binaria[32, 75] == 0


def test_encontrar_vertices_rejeita_subfaixa_e_paralelogramo():
    img = np.full((220, 520, 3), 60, dtype=np.uint8)
    caixa = Caixa(x1=60, y1=40, x2=460, y2=170)

    # Caso 1: apenas subfaixa branca muito achatada (aspecto > 4.0) dentro da caixa
    img_sub = img.copy()
    img_sub[85:160, 75:445] = (235, 235, 235)
    assert encontrar_vertices_placa_roi(img_sub, caixa_esperada=caixa) is None

    # Caso 2: paralelogramo fortemente cisalhado com ângulos fora de [78°, 102°]
    img_shear = img.copy()
    poly = np.array([[105, 48], [450, 48], [405, 162], [60, 162]], dtype=np.int32)
    cv2.fillPoly(img_shear, [poly], (235, 235, 235))
    assert encontrar_vertices_placa_roi(img_shear, caixa_esperada=caixa) is None

    # Caso 3: placa ortogonal proporcional à caixa esperada deve ser aceita
    img_ok = img.copy()
    img_ok[45:165, 68:452] = (235, 235, 235)
    vertices = encontrar_vertices_placa_roi(img_ok, caixa_esperada=caixa)
    assert vertices is not None
    assert vertices.shape == (4, 2)
