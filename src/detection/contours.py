"""Detecção clássica de placas por bordas (Canny), contornos e filtro de quadriláteros."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import Caixa


def detectar_placa_classica(
    imagem: np.ndarray,
    canny_min: int = 30,
    canny_max: int = 200,
    aspecto_min: float = 1.5,
    aspecto_max: float = 6.0,
    area_min_pct: float = 0.003,
) -> tuple[Caixa | None, np.ndarray | None, np.ndarray]:
    """Detecta a placa veicular utilizando visão computacional clássica.
    
    Retorna:
        - Caixa: bounding box da melhor placa candidata ou None;
        - Vertices: array (4, 2) dos 4 cantos do quadrilátero para perspectiva ou None;
        - Bordas: imagem binária das bordas Canny geradas para inspeção visual.
    """
    if imagem.ndim == 3:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem.copy()

    h_img, w_img = cinza.shape[:2]
    area_total = h_img * w_img
    area_min = area_total * area_min_pct

    # 1. Filtro bilateral preservando bordas
    suave = cv2.bilateralFilter(cinza, d=11, sigmaColor=17, sigmaSpace=17)

    # 2. Detector de bordas Canny
    bordas = cv2.Canny(suave, canny_min, canny_max)

    # 3. Busca de contornos
    contornos, _ = cv2.findContours(bordas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None, bordas

    # Ordena por área decrescente e seleciona os maiores candidatos
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:15]

    melhor_caixa: Caixa | None = None
    melhores_vertices: np.ndarray | None = None
    maior_area = 0.0

    for c in contornos:
        area = cv2.contourArea(c)
        if area < area_min:
            continue

        perimetro = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.018 * perimetro, True)

        # Procura por polígono de 4 vértices (retângulo/quadrilátero)
        if len(aprox) == 4:
            x, y, w, h = cv2.boundingRect(aprox)
            if h == 0:
                continue

            proporcao = w / float(h)
            if aspecto_min <= proporcao <= aspecto_max:
                if area > maior_area:
                    maior_area = area
                    melhor_caixa = Caixa(x1=x, y1=y, x2=x + w, y2=y + h, confianca=1.0)
                    melhores_vertices = aprox.reshape(4, 2)

    if melhor_caixa is not None:
        bordas_bgr = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
        if melhores_vertices is not None:
            cv2.drawContours(bordas_bgr, [melhores_vertices], -1, (0, 200, 0), 2)
        else:
            cv2.rectangle(
                bordas_bgr,
                (melhor_caixa.x1, melhor_caixa.y1),
                (melhor_caixa.x2, melhor_caixa.y2),
                (0, 200, 0),
                2,
            )
        bordas = bordas_bgr

    return melhor_caixa, melhores_vertices, bordas
