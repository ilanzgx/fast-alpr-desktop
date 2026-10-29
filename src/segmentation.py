"""Segmentação de caracteres da placa via componentes conexos."""

from __future__ import annotations

import cv2
import numpy as np


def segmentar_caracteres(
    binaria: np.ndarray,
    altura_min_pct: float = 0.35,
    altura_max_pct: float = 0.95,
    largura_max_pct: float = 0.25,
) -> list[np.ndarray]:
    """Segmenta caracteres individuais a partir de uma imagem binarizada.
    
    Espera caracteres brancos (255) sobre fundo preto (0).
    Filtra componentes conexos por dimensões plausíveis de caracteres de placa
    e os ordena horizontalmente (esquerda para direita).
    """
    if binaria is None or binaria.size == 0:
        return []

    h_roi, w_roi = binaria.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binaria, connectivity=8
    )

    h_min = int(h_roi * altura_min_pct)
    h_max = int(h_roi * altura_max_pct)
    w_max = int(w_roi * largura_max_pct)
    area_min = int(h_min * (w_roi * 0.03))

    candidatos = []
    for i in range(1, num_labels):  # Ignora label 0 (fundo)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if h_min <= h <= h_max and 3 <= w <= w_max and area >= area_min:
            # Recorta o caractere da máscara binarizada
            char_crop = binaria[y : y + h, x : x + w]
            candidatos.append((x, char_crop))

    # Ordena da esquerda para a direita pela coordenada x
    candidatos.sort(key=lambda item: item[0])

    # Normaliza cada caractere para um tamanho padrão com borda
    caracteres_normalizados = []
    for _, char_crop in candidatos:
        norm = normalizar_caractere(char_crop)
        caracteres_normalizados.append(norm)

    return caracteres_normalizados


def normalizar_caractere(
    char_crop: np.ndarray, largura: int = 32, altura: int = 48, margem: int = 4
) -> np.ndarray:
    """Redimensiona o caractere mantendo a proporção e adiciona margem fixa centralizada."""
    h, w = char_crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((altura, largura), dtype=np.uint8)

    # Redimensiona proporcionalmente para caber na caixa alvo
    escala = min((altura - 2 * margem) / h, (largura - 2 * margem) / w)
    novo_w = max(1, int(w * escala))
    novo_h = max(1, int(h * escala))

    redim = cv2.resize(char_crop, (novo_w, novo_h), interpolation=cv2.INTER_NEAREST)

    tela = np.zeros((altura, largura), dtype=np.uint8)
    topo = (altura - novo_h) // 2
    esq = (largura - novo_w) // 2
    tela[topo : topo + novo_h, esq : esq + novo_w] = redim

    return tela


def montar_faixa_caracteres(
    caracteres: list[np.ndarray],
    espacamento: int = 10,
    margem_v: int = 16,
    margem_h: int = 16,
) -> np.ndarray | None:
    """Monta uma imagem horizontal com os caracteres segmentados dispostos sequencialmente.
    
    Remove qualquer ruído externo (moldura da placa, QR code, parafusos, cabeçalho)
    e apresenta ao Tesseract uma linha limpa com texto preto sobre fundo branco.
    """
    if not caracteres:
        return None

    h_char, w_char = caracteres[0].shape[:2]
    num_chars = len(caracteres)
    largura_total = num_chars * w_char + (num_chars - 1) * espacamento + 2 * margem_h
    altura_total = h_char + 2 * margem_v

    faixa = np.full((altura_total, largura_total), 255, dtype=np.uint8)

    for i, char_img in enumerate(caracteres):
        x = margem_h + i * (w_char + espacamento)
        # Assegura polaridade: texto escuro (0) sobre fundo claro (255)
        if np.mean(char_img) < 128:
            char_proc = cv2.bitwise_not(char_img)
        else:
            char_proc = char_img.copy()

        faixa[margem_v : margem_v + h_char, x : x + w_char] = char_proc

    return faixa

