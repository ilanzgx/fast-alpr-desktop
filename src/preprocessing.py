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

    # Inversão automática avaliada na região central da placa (evita que bordas escuras
    # adicionadas pelo zoom-out ou grade do carro invertam a polaridade correta)
    if inverter_se_fundo_claro:
        h_bin, w_bin = binaria.shape[:2]
        if h_bin >= 20 and w_bin >= 40:
            regiao_ref = binaria[
                int(h_bin * 0.25) : int(h_bin * 0.85),
                int(w_bin * 0.15) : int(w_bin * 0.85),
            ]
        else:
            regiao_ref = binaria
        pixels_brancos = np.count_nonzero(regiao_ref == 255)
        if pixels_brancos > (regiao_ref.size / 2):
            binaria = cv2.bitwise_not(binaria)

    return binaria


def recortar_roi_com_offset(
    imagem: np.ndarray, caixa: Caixa, margem: float = 0.04
) -> tuple[np.ndarray, tuple[int, int]]:
    """Recorta o ROI aplicando margem percentual e retorna também o offset (x1, y1)."""
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
    return recorte, (x1, y1)


def recortar_roi(imagem: np.ndarray, caixa: Caixa, margem: float = 0.04) -> np.ndarray:
    """Recorta a região de interesse (ROI) aplicando uma margem percentual com clipping."""
    recorte, _ = recortar_roi_com_offset(imagem, caixa, margem=margem)
    return recorte


def ampliar_roi_com_escala(roi: np.ndarray, altura_alvo: int = 150) -> tuple[np.ndarray, float]:
    """Redimensiona o ROI mantendo a proporção e retorna também o fator de escala aplicado."""
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return roi, 1.0

    if h < altura_alvo:
        fator = altura_alvo / h
        novo_w = max(1, int(round(w * fator)))
        # Recalcula fator efetivo com base nas dimensões inteiras
        fator_efetivo_h = altura_alvo / h
        redim = cv2.resize(roi, (novo_w, altura_alvo), interpolation=cv2.INTER_CUBIC)
        return redim, fator_efetivo_h

    return roi, 1.0


def ampliar_roi(roi: np.ndarray, altura_alvo: int = 150) -> np.ndarray:
    """Redimensiona o ROI mantendo a proporção para melhorar o OCR (altura mínima recomendada)."""
    redim, _ = ampliar_roi_com_escala(roi, altura_alvo=altura_alvo)
    return redim


def estimar_angulo_inclinacao(cinza: np.ndarray) -> float:
    """Estima a inclinação angular da placa (-45° a +45°) usando pico de histograma Hough e contornos."""
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
        threshold=max(14, int(w * 0.12)),
        minLineLength=max(14, int(w * 0.14)),
        maxLineGap=max(5, int(w * 0.05)),
    )

    angulos: list[float] = []
    pesos: list[float] = []
    if linhas is not None:
        for linha in linhas:
            lx1, ly1, lx2, ly2 = linha[0]
            # Ignora bordas artificiais do recorte da imagem
            if (ly1 <= 3 and ly2 <= 3) or (ly1 >= h - 4 and ly2 >= h - 4):
                continue
            if (lx1 <= 3 and lx2 <= 3) or (lx1 >= w - 4 and lx2 >= w - 4):
                continue

            comp = float(np.hypot(lx2 - lx1, ly2 - ly1))
            ang = float(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
            if ang > 45.0:
                ang -= 90.0
            elif ang < -45.0:
                ang += 90.0

            if -45.0 <= ang <= 45.0:
                angulos.append(ang)
                pesos.append(comp)

    if angulos and sum(pesos) > 0:
        # Agrupa em bins de 2 graus para encontrar a moda (orientação dominante)
        # Evita cancelamento mútuo entre bordas da placa e traços inclinados de caracteres
        bins = np.arange(-45, 47, 2)
        hist, bin_edges = np.histogram(angulos, bins=bins, weights=pesos)
        idx_pico = int(np.argmax(hist))
        ang_pico = (bin_edges[idx_pico] + bin_edges[idx_pico + 1]) / 2.0

        proximas = [a for a, p in zip(angulos, pesos) if abs(a - ang_pico) <= 4.0]
        pesos_prox = [p for a, p in zip(angulos, pesos) if abs(a - ang_pico) <= 4.0]
        if proximas and sum(pesos_prox) > 0:
            return float(np.average(proximas, weights=pesos_prox))
        return float(ang_pico)

    # 3. Fallback: minAreaRect no contorno claro da placa
    _, thr = cv2.threshold(cinza_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contornos, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) > (h * w * 0.10):
            rect = cv2.minAreaRect(c)
            (rx, ry), (rw, rh), rang = rect
            aspecto_rect = max(rw, rh) / max(1.0, min(rw, rh))
            if rw < rh:
                rang += 90.0
            if rang > 45.0:
                rang -= 90.0
            if abs(rang) >= 35.0 and aspecto_rect < 1.35:
                continue
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
        aspecto_txt = max(larg_ret, alt_ret) / max(1.0, min(larg_ret, alt_ret))
        if larg_ret < alt_ret:
            angulo += 90.0
        if angulo > 45.0:
            angulo -= 90.0
        if abs(angulo) <= 45.0 and not (abs(angulo) >= 35.0 and aspecto_txt < 1.35):
            return float(angulo)

    return 0.0


def _ordenar_vertices(vertices: np.ndarray) -> np.ndarray:
    """Ordena os 4 vértices de um quadrilátero em [sup-esq, sup-dir, inf-dir, inf-esq]."""
    pts = vertices.reshape(4, 2).astype(np.float32)
    soma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    idx_se = int(np.argmin(soma))
    idx_id = int(np.argmax(soma))
    idx_sd = int(np.argmin(diff))
    idx_ie = int(np.argmax(diff))

    if len({idx_se, idx_id, idx_sd, idx_ie}) == 4:
        return np.array([pts[idx_se], pts[idx_sd], pts[idx_id], pts[idx_ie]], dtype=np.float32)

    # Fallback para rotações extremas onde soma/diff podem empatar entre cantos
    ordenados_y = pts[np.argsort(pts[:, 1])]
    topo = ordenados_y[:2][np.argsort(ordenados_y[:2, 0])]
    base = ordenados_y[2:][np.argsort(ordenados_y[2:, 0])]
    return np.array([topo[0], topo[1], base[1], base[0]], dtype=np.float32)


def _angulos_internos_validos(
    pts: np.ndarray, ang_min: float = 42.0, ang_max: float = 138.0
) -> bool:
    """Verifica se todos os ângulos internos de um polígono de 4 lados estão na faixa aceitável."""
    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom < 1e-6:
            return False
        cos_ang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cos_ang)))
        if ang < ang_min or ang > ang_max:
            return False
    return True


def _calcular_iou_caixa(bbox_rect: tuple[int, int, int, int], caixa: Caixa) -> float:
    """Calcula a interseção sobre união (IoU) entre (x, y, w, h) e uma Caixa."""
    qx1, qy1, qw, qh = bbox_rect
    qx2, qy2 = qx1 + qw, qy1 + qh

    ix1 = max(qx1, caixa.x1)
    iy1 = max(qy1, caixa.y1)
    ix2 = min(qx2, caixa.x2)
    iy2 = min(qy2, caixa.y2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_q = max(0, qw * qh)
    area_c = caixa.largura * caixa.altura
    uniao = area_q + area_c - inter
    return inter / max(1.0, float(uniao))


def _recortar_com_zoom_out(
    imagem: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad_x: int, pad_y: int
) -> np.ndarray:
    """Recorta a região [x1:x2, y1:y2] aplicando margem zoom-out garantida via replicação."""
    h, w = imagem.shape[:2]
    req_x1, req_y1 = x1 - pad_x, y1 - pad_y
    req_x2, req_y2 = x2 + pad_x, y2 + pad_y

    cl_x1 = max(0, req_x1)
    cl_y1 = max(0, req_y1)
    cl_x2 = min(w, req_x2)
    cl_y2 = min(h, req_y2)

    recorte = imagem[cl_y1:cl_y2, cl_x1:cl_x2]
    if recorte.size == 0:
        return imagem.copy()

    borda_esq = cl_x1 - req_x1
    borda_topo = cl_y1 - req_y1
    borda_dir = req_x2 - cl_x2
    borda_base = req_y2 - cl_y2

    if borda_esq > 0 or borda_topo > 0 or borda_dir > 0 or borda_base > 0:
        return cv2.copyMakeBorder(
            recorte,
            borda_topo,
            borda_base,
            borda_esq,
            borda_dir,
            borderType=cv2.BORDER_REPLICATE,
        )
    return recorte


def _rotacionar_com_matriz(
    imagem: np.ndarray, angulo: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotaciona a imagem expandindo o canvas com replicação de borda e retorna matriz 2x3."""
    h, w = imagem.shape[:2]
    centro = (w / 2.0, h / 2.0)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    cos = float(np.abs(matriz[0, 0]))
    sin = float(np.abs(matriz[0, 1]))
    novo_w = int((h * sin) + (w * cos))
    novo_h = int((h * cos) + (w * sin))
    matriz[0, 2] += (novo_w / 2.0) - centro[0]
    matriz[1, 2] += (novo_h / 2.0) - centro[1]

    # Usa BORDER_REPLICATE para evitar cantos pretos que viram blocos brancos após binarização
    rotacionada = cv2.warpAffine(
        imagem,
        matriz,
        (novo_w, novo_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotacionada, matriz


def _estimar_caixa_desinclinada(
    caixa_interna: Caixa, angulo: float, matriz: np.ndarray
) -> tuple[float, float, float, float]:
    """Projeta caixa para o espaço rotacionado e estima (cx, cy, largura, altura) sem inclinação."""
    cantos = np.array(
        [
            [caixa_interna.x1, caixa_interna.y1, 1.0],
            [caixa_interna.x2, caixa_interna.y1, 1.0],
            [caixa_interna.x2, caixa_interna.y2, 1.0],
            [caixa_interna.x1, caixa_interna.y2, 1.0],
        ],
        dtype=np.float32,
    )
    cantos_rot = (matriz @ cantos.T).T
    cx_rot = float(np.mean(cantos_rot[:, 0]))
    cy_rot = float(np.mean(cantos_rot[:, 1]))

    w_box = float(caixa_interna.largura)
    h_box = float(caixa_interna.altura)
    rad = float(np.radians(abs(angulo)))
    c, s = abs(float(np.cos(rad))), abs(float(np.sin(rad)))
    denom = c * c - s * s
    if denom > 0.25:
        w_est = (w_box * c - h_box * s) / denom
        h_est = (h_box * c - w_box * s) / denom
        w_est = float(np.clip(w_est, w_box * 0.55, w_box * 1.05))
        h_est = float(np.clip(h_est, min(h_box * 0.40, w_est / 4.2), h_box))
        # Evita contração artificial da altura em ângulos pequenos ou caixas justas do YOLO
        if abs(angulo) < 5.0:
            h_est = max(h_est, h_box * 0.95)
        h_est = max(h_est, min(h_box, w_est / 3.45))
    else:
        w_est = w_box * 0.85
        h_est = max(h_box * 0.55, min(h_box, w_est / 3.45))

    return cx_rot, cy_rot, w_est, h_est


def detectar_vertices_pos_rotacao(
    roi_ctx_amp: np.ndarray,
    caixa_ctx_amp: Caixa,
    escala_ctx: float,
    ox: int,
    oy: int,
    angulo_maximo: float = 45.0,
) -> np.ndarray | None:
    """Desinclina placas escorçadas e busca os 4 vértices, mapeando-os para a imagem original."""
    h, w = roi_ctx_amp.shape[:2]
    if h < 10 or w < 20:
        return None

    rx1 = max(0, caixa_ctx_amp.x1)
    ry1 = max(0, caixa_ctx_amp.y1)
    rx2 = min(w, caixa_ctx_amp.x2)
    ry2 = min(h, caixa_ctx_amp.y2)
    nucleo = roi_ctx_amp[ry1:ry2, rx1:rx2]
    if nucleo.size == 0:
        return None

    angulo = estimar_angulo_inclinacao(converter_cinza(nucleo))
    if abs(angulo) >= 0.8 and abs(angulo) <= angulo_maximo:
        corrigida, matriz = _rotacionar_com_matriz(roi_ctx_amp, angulo)
        cx_rot, cy_rot, w_est, h_est = _estimar_caixa_desinclinada(caixa_ctx_amp, angulo, matriz)
    else:
        corrigida = roi_ctx_amp
        matriz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        cx_rot = (caixa_ctx_amp.x1 + caixa_ctx_amp.x2) / 2.0
        cy_rot = (caixa_ctx_amp.y1 + caixa_ctx_amp.y2) / 2.0
        w_est = float(caixa_ctx_amp.largura)
        h_est = float(caixa_ctx_amp.altura)

    # Só necessita retificação em perspectiva se a placa ainda estiver escorçada/diagonal
    if (w_est / max(1.0, h_est)) >= 2.2:
        return None

    nh, nw = corrigida.shape[:2]
    tx1 = max(0, int(round(cx_rot - w_est * 0.54)))
    ty1 = max(0, int(round(cy_rot - h_est * 0.54)))
    tx2 = min(nw, int(round(cx_rot + w_est * 0.54)))
    ty2 = min(nh, int(round(cy_rot + h_est * 0.54)))
    sub = corrigida[ty1:ty2, tx1:tx2]
    vt = encontrar_vertices_placa_roi(sub)
    if vt is None or cv2.contourArea(vt) < (w_est * h_est * 0.45):
        return None

    # Reprojeta os vértices detectados do recorte rotacionado para a imagem original inteira
    v_corr = vt + np.array([tx1, ty1], dtype=np.float32)
    m_inv = cv2.invertAffineTransform(matriz)
    v_homog = np.hstack([v_corr, np.ones((4, 1), dtype=np.float32)])
    v_ctx_amp = (m_inv @ v_homog.T).T.astype(np.float32)
    return (v_ctx_amp / escala_ctx) + np.array([ox, oy], dtype=np.float32)


def corrigir_inclinacao(
    imagem: np.ndarray,
    angulo_maximo: float = 45.0,
    margem_zoom_out: float = 0.09,
    caixa_interna: Caixa | None = None,
) -> tuple[np.ndarray, float]:
    """Estima e corrige inclinações da placa mantendo margem zoom-out para não cortar caracteres."""
    h, w = imagem.shape[:2]
    if h < 10 or w < 20:
        return imagem.copy(), 0.0

    # Estima o ângulo na região central da placa para evitar interferência da grade do veículo
    if caixa_interna is not None:
        rx1 = max(0, caixa_interna.x1)
        ry1 = max(0, caixa_interna.y1)
        rx2 = min(w, caixa_interna.x2)
        ry2 = min(h, caixa_interna.y2)
        nucleo = imagem[ry1:ry2, rx1:rx2]
        cinza_est = converter_cinza(nucleo) if nucleo.size > 0 else converter_cinza(imagem)
    else:
        cinza_est = converter_cinza(imagem)

    angulo = estimar_angulo_inclinacao(cinza_est)

    # Ignora rotações insignificantes (< 0.8°) ou além do limite aceitável
    if abs(angulo) < 0.8 or abs(angulo) > angulo_maximo:
        return imagem.copy(), 0.0

    corrigida, matriz = _rotacionar_com_matriz(imagem, angulo)
    novo_h, novo_w = corrigida.shape[:2]

    if caixa_interna is not None:
        cx_rot, cy_rot, w_est, h_est = _estimar_caixa_desinclinada(caixa_interna, angulo, matriz)
        w_box = float(caixa_interna.largura)
        h_box = float(caixa_interna.altura)

        px1 = int(round(cx_rot - w_est / 2.0))
        py1 = int(round(cy_rot - h_est / 2.0))
        px2 = int(round(cx_rot + w_est / 2.0))
        py2 = int(round(cy_rot + h_est / 2.0))

        # Refina altura/largura via contorno claro apenas se não podar a largura estimada
        if abs(angulo) >= 2.0:
            busca_x1 = max(0, int(cx_rot - max(w_est, w_box) * 0.55))
            busca_y1 = max(0, int(cy_rot - h_box * 0.55))
            busca_x2 = min(novo_w, int(cx_rot + max(w_est, w_box) * 0.55))
            busca_y2 = min(novo_h, int(cy_rot + h_box * 0.55))
            janela = converter_cinza(corrigida[busca_y1:busca_y2, busca_x1:busca_x2])
            if janela.size > 0:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(janela)
                _, thr = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contornos, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidatos = []
                for c_cont in contornos:
                    bx, by, bw, bh = cv2.boundingRect(c_cont)
                    aspecto = bw / max(1, bh)
                    area = cv2.contourArea(c_cont)
                    cy_cand = busca_y1 + by + (bh / 2.0)
                    if (
                        aspecto >= 1.8
                        and bw >= (w_est * 0.78)
                        and abs(cy_cand - cy_rot) <= (h_est * 0.22)
                        and max(15, int(h_est * 0.80)) <= bh <= max(25, int(h_est * 1.25))
                    ):
                        candidatos.append((area, (bx + busca_x1, by + busca_y1, bw, bh)))
                if candidatos:
                    candidatos.sort(reverse=True)
                    _, (bx, by, bw, bh) = candidatos[0]
                    px1 = min(px1, bx)
                    px2 = max(px2, bx + bw)
                    py1 = min(py1, by)
                    py2 = max(py2, by + bh)

        pw = max(20, px2 - px1)
        ph = max(12, py2 - py1)
        pad_x = max(8, int(round(pw * min(margem_zoom_out, 0.08))))
        pad_y = max(10, int(round(ph * max(margem_zoom_out, 0.11))))
        return _recortar_com_zoom_out(corrigida, px1, py1, px2, py2, pad_x, pad_y), angulo

    # Modo standalone (ROI sem contexto extra externo)
    if abs(angulo) >= 2.0:
        cinza_rot = converter_cinza(corrigida)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cinza_rot)
        _, thr = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []
        for c_cont in contornos:
            x, y, cw, ch = cv2.boundingRect(c_cont)
            aspecto = cw / max(1, ch)
            area = cv2.contourArea(c_cont)
            if aspecto >= 1.8 and cw > (novo_w * 0.50) and ch > 15:
                candidatos.append((area, (x, y, cw, ch)))

        if candidatos:
            candidatos.sort(reverse=True)
            _, (bx, by, bw, bh) = candidatos[0]
            pad_x = max(8, int(round(bw * min(margem_zoom_out, 0.08))))
            pad_y = max(8, int(round(bh * max(margem_zoom_out, 0.10))))
            return _recortar_com_zoom_out(corrigida, bx, by, bx + bw, by + bh, pad_x, pad_y), angulo

    if margem_zoom_out > 0:
        pad_x = max(4, int(round(novo_w * min(margem_zoom_out, 0.06))))
        pad_y = max(4, int(round(novo_h * min(margem_zoom_out, 0.08))))
        corrigida = cv2.copyMakeBorder(
            corrigida, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
        )

    return corrigida, angulo


def encontrar_vertices_placa_roi(
    roi: np.ndarray, caixa_esperada: Caixa | None = None
) -> np.ndarray | None:
    """Detecta os 4 vértices do contorno quadrilátero da placa veicular dentro do ROI recortado."""
    if roi is None or roi.size == 0:
        return None
    h, w = roi.shape[:2]
    if h < 20 or w < 40:
        return None

    cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cinza)

    area_ref = float(h * w)
    tem_margem = False
    if caixa_esperada is not None:
        area_ref = max(1.0, float(caixa_esperada.largura * caixa_esperada.altura))
        min_pad = max(6, int(min(w, h) * 0.06))
        tem_margem = (
            caixa_esperada.x1 >= min_pad
            and caixa_esperada.y1 >= min_pad
            and (w - caixa_esperada.x2) >= min_pad
            and (h - caixa_esperada.y2) >= min_pad
        )

    # Teste de parâmetros para detecção robusta de contornos quadriláteros
    tentativas = [(50, 150, (5, 5)), (30, 180, (3, 3)), (20, 100, (5, 5))]
    melhores_candidatos: list[tuple[float, float, np.ndarray]] = []
    for c_min, c_max, k_size in tentativas:
        edges = cv2.Canny(clahe, c_min, c_max)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contornos, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in sorted(contornos, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if caixa_esperada is not None:
                if area < (area_ref * 0.22) or area > (area_ref * 1.25):
                    continue
            elif area < (h * w * 0.20):
                continue

            peri = cv2.arcLength(c, True)
            for eps in [0.04, 0.03, 0.05, 0.025]:
                aprox = cv2.approxPolyDP(c, eps * peri, True)
                if len(aprox) == 4 and cv2.isContourConvex(aprox):
                    pts = aprox.reshape(4, 2).astype(np.float32)
                    if not _angulos_internos_validos(pts, ang_min=42.0, ang_max=138.0):
                        continue

                    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
                    area_min_rect = max(1.0, float(rw * rh))
                    if (area / area_min_rect) < 0.72:
                        continue

                    lados = [
                        float(
                            np.hypot(
                                pts[i][0] - pts[(i + 1) % 4][0],
                                pts[i][1] - pts[(i + 1) % 4][1],
                            )
                        )
                        for i in range(4)
                    ]
                    larg_media = (lados[0] + lados[2]) / 2.0
                    alt_media = (lados[1] + lados[3]) / 2.0
                    if larg_media < alt_media:
                        larg_media, alt_media = alt_media, larg_media
                    aspecto = larg_media / max(1.0, alt_media)
                    if not (1.2 <= aspecto <= 5.5):
                        continue

                    if caixa_esperada is None:
                        return pts

                    # Rejeita subfaixas internas brancas (sem a barra azul Mercosul) e
                    # paralelogramos distorcidos pelo QR code/cantos em placas não escorçadas
                    if aspecto > 3.65:
                        continue
                    if aspecto >= 2.2 and not _angulos_internos_validos(
                        pts, ang_min=78.0, ang_max=102.0
                    ):
                        continue

                    bbox = cv2.boundingRect(aprox)
                    iou = _calcular_iou_caixa(bbox, caixa_esperada)
                    if iou < 0.35:
                        continue
                    if tem_margem:
                        na_borda = sum(
                            1
                            for px, py in pts
                            if px <= 2 or px >= (w - 3) or py <= 2 or py >= (h - 3)
                        )
                        if na_borda >= 2:
                            continue

                    # Prefere contornos alinhados à altura total da caixa (evita subfaixa parcial)
                    _, _, _, bh = bbox
                    fator_alt = min(1.0, bh / max(1.0, float(caixa_esperada.altura)))
                    score = iou * (0.85 if (aspecto > 3.65 and fator_alt < 0.78) else 1.0)
                    melhores_candidatos.append((score, area, pts))
                    break

    if melhores_candidatos:
        melhores_candidatos.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return melhores_candidatos[0][2]
    return None


def verificar_distorcao_geometrica(
    pts: np.ndarray, threshold_graus: float = 2.5
) -> tuple[bool, float, float]:
    """Verifica se o quadrilátero possui inclinação ou distorção de perspectiva significativa."""
    sup_esq, sup_dir, inf_dir, inf_esq = _ordenar_vertices(pts)

    ang_topo = np.degrees(np.arctan2(sup_dir[1] - sup_esq[1], sup_dir[0] - sup_esq[0]))
    ang_base = np.degrees(np.arctan2(inf_dir[1] - inf_esq[1], inf_dir[0] - inf_esq[0]))
    inc_h = (abs(ang_topo) + abs(ang_base)) / 2.0

    ang_esq = np.degrees(np.arctan2(inf_esq[0] - sup_esq[0], inf_esq[1] - sup_esq[1]))
    ang_dir = np.degrees(np.arctan2(inf_dir[0] - sup_dir[0], inf_dir[1] - sup_dir[1]))
    cis_v = (abs(ang_esq) + abs(ang_dir)) / 2.0

    distorcido = (inc_h >= threshold_graus) or (cis_v >= threshold_graus)
    return distorcido, float(inc_h), float(cis_v)


def corrigir_perspectiva(
    imagem: np.ndarray,
    vertices: np.ndarray,
    largura: int | None = None,
    altura: int = 130,
    margem_saida: float = 0.09,
) -> np.ndarray:
    """Aplica transformação de perspectiva com margem zoom-out para evitar cortes nas bordas."""
    sup_esq, sup_dir, inf_dir, inf_esq = _ordenar_vertices(vertices)
    margem = max(0.0, min(float(margem_saida), 0.25))

    largura_explicita = largura is not None and largura > 0
    if not largura_explicita:
        larg_topo = np.hypot(sup_dir[0] - sup_esq[0], sup_dir[1] - sup_esq[1])
        larg_base = np.hypot(inf_dir[0] - inf_esq[0], inf_dir[1] - inf_esq[1])
        larg_max = max(larg_topo, larg_base)

        alt_esq = np.hypot(inf_esq[0] - sup_esq[0], inf_esq[1] - sup_esq[1])
        alt_dir = np.hypot(inf_dir[0] - sup_dir[0], inf_dir[1] - sup_dir[1])
        alt_max = max(alt_esq, alt_dir)

        aspecto = larg_max / max(1.0, alt_max)
        # Se aspecto < 2.2, houve escorço acentuado por ângulo lateral.
        # Restaura proporção padrão de placas veiculares brasileiras (~3.08)
        if aspecto < 2.2:
            aspecto = 3.08
        larg_placa = max(1, int(round(altura * aspecto)))
        pad_x = int(round(larg_placa * min(margem, 0.08)))
        pad_y = int(round(altura * margem))
        largura_out = larg_placa + 2 * pad_x
        altura_out = altura + 2 * pad_y
    else:
        largura_out = int(largura)
        altura_out = int(altura)
        pad_x = min(largura_out // 4, int(round(largura_out * min(margem, 0.08))))
        pad_y = min(altura_out // 4, int(round(altura_out * margem)))

    origem = np.array([sup_esq, sup_dir, inf_dir, inf_esq], dtype=np.float32)
    destino = np.array(
        [
            [pad_x, pad_y],
            [largura_out - 1 - pad_x, pad_y],
            [largura_out - 1 - pad_x, altura_out - 1 - pad_y],
            [pad_x, altura_out - 1 - pad_y],
        ],
        dtype=np.float32,
    )

    matriz = cv2.getPerspectiveTransform(origem, destino)
    return cv2.warpPerspective(
        imagem,
        matriz,
        (largura_out, altura_out),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
