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


def estimar_angulo_inclinacao(cinza: np.ndarray) -> float:
    """Estima a inclinação angular da placa (-45° a +45°) usando Hough Lines e contornos."""
    h, w = cinza.shape[:2]
    if h < 10 or w < 20:
        return 0.0

    # 1. CLAHE para reforçar contraste de bordas mesmo em baixa iluminação
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cinza_clahe = clahe.apply(cinza)

    # 2. Detecção via Hough Lines Ponderadas por Comprimento
    suave = cv2.GaussianBlur(cinza_clahe, (5, 5), 0)
    bordas = cv2.Canny(suave, 50, 150)
    linhas = cv2.HoughLinesP(
        bordas,
        rho=1,
        theta=np.pi / 180,
        threshold=max(20, int(w * 0.15)),
        minLineLength=max(20, int(w * 0.20)),
        maxLineGap=max(5, int(w * 0.05)),
    )

    angulos: list[float] = []
    pesos: list[float] = []
    if linhas is not None:
        for linha in linhas:
            lx1, ly1, lx2, ly2 = linha[0]
            comp = float(np.hypot(lx2 - lx1, ly2 - ly1))
            ang = float(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
            if -45.0 <= ang <= 45.0:
                angulos.append(ang)
                pesos.append(comp)

    if angulos and sum(pesos) > 0:
        return float(np.average(angulos, weights=pesos))

    # 3. Fallback: minAreaRect no contorno claro da placa
    _, thr = cv2.threshold(cinza_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contornos, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) > (h * w * 0.10):
            rect = cv2.minAreaRect(c)
            (rx, ry), (rw, rh), rang = rect
            if rw < rh:
                rang += 90.0
            if rang > 45.0:
                rang -= 90.0
            if abs(rang) <= 45.0:
                return float(rang)

    # 4. Fallback clássico: minAreaRect nos pixels escuros do texto
    by = max(1, int(h * 0.08))
    bx = max(1, int(w * 0.05))
    interior = thr[by : h - by, bx : w - bx]
    pontos = np.column_stack(np.where(interior < 128))
    if len(pontos) >= 15:
        pontos_xy = pontos[:, ::-1].astype(np.float32)
        retangulo = cv2.minAreaRect(pontos_xy)
        angulo = retangulo[-1]
        larg_ret, alt_ret = retangulo[1]
        if larg_ret < alt_ret:
            angulo += 90.0
        if angulo > 45.0:
            angulo -= 90.0
        if abs(angulo) <= 45.0:
            return float(angulo)

    return 0.0


def corrigir_inclinacao(
    cinza: np.ndarray, angulo_maximo: float = 45.0
) -> tuple[np.ndarray, float]:
    """Estima e corrige inclinações da placa rotacionando e expandindo o canvas para não podar bordas."""
    h, w = cinza.shape[:2]
    if h < 10 or w < 20:
        return cinza.copy(), 0.0

    angulo = estimar_angulo_inclinacao(cinza)

    # Ignora rotações insignificantes (< 0.8°) ou além do limite aceitável
    if abs(angulo) < 0.8 or abs(angulo) > angulo_maximo:
        return cinza.copy(), 0.0

    centro = (w / 2.0, h / 2.0)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    # Expande o canvas para que nenhum canto da placa seja cortado
    cos = np.abs(matriz[0, 0])
    sin = np.abs(matriz[0, 1])
    novo_w = int((h * sin) + (w * cos))
    novo_h = int((h * cos) + (w * sin))
    matriz[0, 2] += (novo_w / 2.0) - centro[0]
    matriz[1, 2] += (novo_h / 2.0) - centro[1]

    corrigida = cv2.warpAffine(
        cinza,
        matriz,
        (novo_w, novo_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Se a rotação foi expressiva (> 2°), isola a caixa da placa para remover excesso de para-choque
    if abs(angulo) >= 2.0:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(corrigida)
        _, thr = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []
        for c in contornos:
            x, y, cw, ch = cv2.boundingRect(c)
            aspecto = cw / max(1, ch)
            area = cv2.contourArea(c)
            if aspecto >= 2.0 and cw > (novo_w * 0.35) and ch > 15:
                candidatos.append((area, (x, y, cw, ch)))

        if candidatos:
            candidatos.sort(reverse=True)
            _, (bx, by, bw, bh) = candidatos[0]
            pad_x = int(bw * 0.03)
            pad_y = int(bh * 0.05)
            rx1 = max(0, bx - pad_x)
            ry1 = max(0, by - pad_y)
            rx2 = min(novo_w, bx + bw + pad_x)
            ry2 = min(novo_h, by + bh + pad_y)
            corrigida = corrigida[ry1:ry2, rx1:rx2]

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
