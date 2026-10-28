"""Inferência de detecção de placas com modelo YOLOv8 exportado para ONNX."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.config import Caixa


def letterbox(
    img: np.ndarray, tamanho: int = 640, cor: int = 114
) -> tuple[np.ndarray, float, int, int]:
    """Redimensiona a imagem mantendo a proporção e preenche com bordas até o tamanho de entrada."""
    h, w = img.shape[:2]
    r = min(tamanho / h, tamanho / w)
    nw, nh = round(w * r), round(h * r)

    redim = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    topo = (tamanho - nh) // 2
    esq = (tamanho - nw) // 2

    tela = np.full((tamanho, tamanho, 3), cor, dtype=np.uint8)
    tela[topo : topo + nh, esq : esq + nw] = redim
    return tela, r, esq, topo


def preparar(img: np.ndarray, tamanho: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Prepara a imagem para inferência ONNX: letterbox, normalização e layout NCHW."""
    tela, r, esq, topo = letterbox(img, tamanho=tamanho)
    x = cv2.cvtColor(tela, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]  # HWC -> NCHW (com batch=1)
    return x, r, esq, topo


def decodificar(
    saida: np.ndarray,
    r: float,
    esq: int,
    topo: int,
    conf_min: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Decodifica as previsões do modelo ONNX e projeta as caixas na imagem original.
    
    Suporta saídas transpostas do YOLOv8 [1, 5, 8400] ou [1, 8400, 5], bem como YOLOv5 [1, N, 6].
    """
    preds = saida[0]
    # Se shape estiver transposto (ex: [5, 8400]), transpõe para [8400, 5]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    # YOLOv8 com 1 classe: [cx, cy, w, h, score]
    if preds.shape[1] == 5:
        mascara = preds[:, 4] >= conf_min
        p = preds[mascara]
        if len(p) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        cx, cy, w, h, scores = p.T
    elif preds.shape[1] >= 6:
        # YOLOv5 / formatos com objectness: [cx, cy, w, h, obj, class_score...]
        obj_scores = preds[:, 4]
        class_scores = preds[:, 5:].max(axis=1) if preds.shape[1] > 6 else preds[:, 5]
        scores = obj_scores * class_scores
        mascara = scores >= conf_min
        p = preds[mascara]
        if len(p) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        scores = scores[mascara]
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    else:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Reverte o letterbox para o sistema de coordenadas original da imagem
    x1 = (cx - w / 2 - esq) / r
    y1 = (cy - h / 2 - topo) / r
    x2 = (cx + w / 2 - esq) / r
    y2 = (cy + h / 2 - topo) / r

    caixas = np.stack([x1, y1, x2, y2], axis=1)
    return caixas, scores


def nms(caixas: np.ndarray, scores: np.ndarray, iou_max: float = 0.45) -> list[int]:
    """Supressão de Não-Máximos (NMS) própria implementada em NumPy puro.
    
    Remove caixas redundantes de alta sobreposição sobre o mesmo objeto.
    """
    if len(caixas) == 0:
        return []

    ordem = scores.argsort()[::-1]
    manter = []

    while ordem.size > 0:
        i = ordem[0]
        manter.append(int(i))
        if ordem.size == 1:
            break

        resto = ordem[1:]
        xx1 = np.maximum(caixas[i, 0], caixas[resto, 0])
        yy1 = np.maximum(caixas[i, 1], caixas[resto, 1])
        xx2 = np.minimum(caixas[i, 2], caixas[resto, 2])
        yy2 = np.minimum(caixas[i, 3], caixas[resto, 3])

        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (caixas[i, 2] - caixas[i, 0]) * (caixas[i, 3] - caixas[i, 1])
        area_r = (caixas[resto, 2] - caixas[resto, 0]) * (caixas[resto, 3] - caixas[resto, 1])

        iou = inter / (area_i + area_r - inter + 1e-9)
        ordem = resto[iou <= iou_max]

    return manter


class DetectorYolo:
    """Carrega uma única vez a sessão ONNX e executa a inferência de placas."""

    def __init__(self, caminho_modelo: str | Path) -> None:
        caminho = Path(caminho_modelo)
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo ONNX não encontrado em: {caminho}\n"
                "Para testar o modelo YOLO, baixe o modelo de referência com:\n"
                "  uv run python scripts/baixar_modelo_referencia.py\n"
                "ou treine no Colab usando notebooks/02_treino_yolo.ipynb"
            )

        self.sess = ort.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])
        self.nome_entrada = self.sess.get_inputs()[0].name
        self.nome_saida = self.sess.get_outputs()[0].name

    def detectar(
        self, img: np.ndarray, conf_min: float = 0.25, iou_nms: float = 0.45
    ) -> list[Caixa]:
        """Localiza placas na imagem e devolve a lista de caixas detectadas."""
        h_orig, w_orig = img.shape[:2]
        tensor, r, esq, topo = preparar(img)

        saidas = self.sess.run([self.nome_saida], {self.nome_entrada: tensor})
        saida_tensor = saidas[0]

        caixas_raw, scores = decodificar(saida_tensor, r, esq, topo, conf_min=conf_min)
        if len(caixas_raw) == 0:
            return []

        indices_mantidos = nms(caixas_raw, scores, iou_max=iou_nms)

        resultados = []
        for idx in indices_mantidos:
            x1, y1, x2, y2 = caixas_raw[idx]
            # Limita às bordas da imagem original
            x1 = max(0, min(int(round(x1)), w_orig))
            y1 = max(0, min(int(round(y1)), h_orig))
            x2 = max(0, min(int(round(x2)), w_orig))
            y2 = max(0, min(int(round(y2)), h_orig))

            if x2 > x1 and y2 > y1:
                conf = float(scores[idx])
                resultados.append(Caixa(x1=x1, y1=y1, x2=x2, y2=y2, confianca=conf))

        return resultados
