"""Módulo de pré-processamento de imagem: cinza, CLAHE, limiarização e correção geométrica."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import Caixa


def converter_cinza(bgr: np.ndarray) -> np.ndarray:
    """Converte imagem BGR para tons de cinza."""
    if bgr.ndim == 2:
        return bgr.copy()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def suavizar_gauss(cinza: np.ndarray, k: int = 5) -> np.ndarray:
    """Aplica filtro gaussiano 2D com kernel ímpar k x k."""
    if k <= 1:
        return cinza.copy()
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(cinza, (k, k), 0)


def aplicar_clahe(cinza: np.ndarray, clip: float = 2.0, grade: int = 8) -> np.ndarray:
    """Aplica equalização de histograma adaptativa (CLAHE)."""
    grade = max(2, grade)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grade, grade))
    return clahe.apply(cinza)


def binarizar(
    cinza: np.ndarray,
    modo: str = "otsu",
    bloco: int = 31,
    c: int = 10,
    inverter_se_fundo_claro: bool = True,
) -> np.ndarray:
    """Aplica binarização (Otsu ou Adaptativa Gaussiana).
    
    Garante que os caracteres fiquem brancos (255) sobre fundo preto (0)
    para o algoritmo de componentes conexos.
    """
    if modo == "adaptativa":
        if bloco <= 1:
            bloco = 3
        if bloco % 2 == 0:
            bloco += 1
        binaria = cv2.adaptiveThreshold(
            cinza,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bloco,
            c,
        )
    else:  # Otsu
        _, binaria = cv2.threshold(
            cinza,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    # Inversão automática: se a maior parte da placa for branca (texto escuro),
    # inverte para que o texto vire branco (foreground)
    if inverter_se_fundo_claro:
        pixels_brancos = np.count_nonzero(binaria == 255)
        if pixels_brancos > (binaria.size / 2):
            binaria = cv2.bitwise_not(binaria)

    return binaria


def recortar_roi(imagem: np.ndarray, caixa: Caixa, margem: float = 0.04) -> np.ndarray:
    """Recorta a região de interesse (ROI) aplicando uma margem percentual com clipping."""
    h_img, w_img = imagem.shape[:2]

    largura = caixa.largura
    altura = caixa.altura
    pad_x = int(largura * margem)
    pad_y = int(altura * margem)

    x1 = max(0, caixa.x1 - pad_x)
    y1 = max(0, caixa.y1 - pad_y)
    x2 = min(w_img, caixa.x2 + pad_x)
    y2 = min(h_img, caixa.y2 + pad_y)

    recorte = imagem[y1:y2, x1:x2]
    return recorte


def ampliar_roi(roi: np.ndarray, altura_alvo: int = 150) -> np.ndarray:
    """Redimensiona o ROI mantendo a proporção para melhorar o OCR (altura mínima recomendada)."""
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return roi

    if h < altura_alvo:
        fator = altura_alvo / h
        novo_w = max(1, int(w * fator))
        return cv2.resize(roi, (novo_w, altura_alvo), interpolation=cv2.INTER_CUBIC)

    return roi


def corrigir_inclinacao(cinza: np.ndarray) -> tuple[np.ndarray, float]:
    """Estima e corrige pequenas inclinações angulares da placa via minAreaRect nos pixels do texto."""
    _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = binaria.shape

    # Evita as bordas do recorte
    by = max(1, int(h * 0.08))
    bx = max(1, int(w * 0.05))
    interior = binaria[by : h - by, bx : w - bx]

    # Busca pixels escuros (texto)
    pontos = np.column_stack(np.where(interior < 128))
    if len(pontos) < 15:
        return cinza.copy(), 0.0

    pontos_xy = pontos[:, ::-1].astype(np.float32)
    retangulo = cv2.minAreaRect(pontos_xy)
    angulo = retangulo[-1]
    larg_ret, alt_ret = retangulo[1]

    if larg_ret < alt_ret:
        angulo += 90

    if angulo > 45:
        angulo -= 90

    # Ignora rotações insignificantes ou anômalas
    if abs(angulo) < 1.0 or abs(angulo) > 15.0:
        return cinza.copy(), 0.0

    centro = (w / 2, h / 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    corrigida = cv2.warpAffine(
        cinza,
        matriz,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrigida, angulo


def corrigir_perspectiva(
    imagem: np.ndarray, vertices: np.ndarray, largura: int = 400, altura: int = 130
) -> np.ndarray:
    """Aplica transformação de perspectiva a partir de 4 vértices para um retângulo padronizado."""
    # Ordena vértices: sup-esq, sup-dir, inf-dir, inf-esq
    pts = vertices.reshape(4, 2).astype(np.float32)
    soma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    sup_esq = pts[np.argmin(soma)]
    inf_dir = pts[np.argmax(soma)]
    sup_dir = pts[np.argmin(diff)]
    inf_esq = pts[np.argmax(diff)]

    origem = np.array([sup_esq, sup_dir, inf_dir, inf_esq], dtype=np.float32)
    destino = np.array(
        [[0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]],
        dtype=np.float32,
    )

    matriz = cv2.getPerspectiveTransform(origem, destino)
    return cv2.warpPerspective(imagem, matriz, (largura, altura))
