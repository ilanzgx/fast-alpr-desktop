"""Orquestrador do pipeline de reconhecimento de placas (ALPR)."""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.config import Caixa, ParametrosPipeline, ResultadoPipeline, obter_caminho_modelo
from src.detection.contours import detectar_placa_classica
from src.detection.yolo_onnx import DetectorYolo
from src.ocr import reconhecer_placa
from src.postprocess import corrigir_por_posicao
from src.preprocessing import (
    ampliar_roi,
    ampliar_roi_com_escala,
    aplicar_clahe,
    binarizar,
    converter_cinza,
    corrigir_inclinacao,
    corrigir_perspectiva,
    detectar_vertices_pos_rotacao,
    encontrar_vertices_placa_roi,
    recortar_roi,
    recortar_roi_com_offset,
    suavizar_gauss,
    verificar_distorcao_geometrica,
)
from src.segmentation import segmentar_caracteres
from src.utils import desenhar_caixa

# Cache global do detector YOLO para evitar recarregar a sessão ONNX a cada frame
_DETECTOR_YOLO_CACHE: DetectorYolo | None = None
_CAMINHO_YOLO_CACHE: str | None = None


def _obter_detector_yolo() -> DetectorYolo:
    global _DETECTOR_YOLO_CACHE, _CAMINHO_YOLO_CACHE
    caminho = str(obter_caminho_modelo())
    if _DETECTOR_YOLO_CACHE is None or _CAMINHO_YOLO_CACHE != caminho:
        _DETECTOR_YOLO_CACHE = DetectorYolo(caminho)
        _CAMINHO_YOLO_CACHE = caminho
    return _DETECTOR_YOLO_CACHE


def executar_pipeline(imagem: np.ndarray, params: ParametrosPipeline) -> ResultadoPipeline:
    """Executa o pipeline completo de processamento e reconhecimento."""
    t0_total = time.perf_counter()
    tempos: dict[str, float] = {}

    caixa: Caixa | None = None
    vertices: np.ndarray | None = None
    bordas: np.ndarray | None = None
    erro: str | None = None

    # -------------------------------------------------------------
    # ETAPA 1: DETECÇÃO DA PLACA (E1)
    # -------------------------------------------------------------
    t0_det = time.perf_counter()
    if params.metodo == "classico":
        caixa, vertices, bordas = detectar_placa_classica(
            imagem,
            canny_min=params.canny_min,
            canny_max=params.canny_max,
        )
    else:  # YOLO ONNX
        try:
            detector = _obter_detector_yolo()
            caixas_detectadas = detector.detectar(
                imagem,
                conf_min=params.confianca_min,
                iou_nms=params.iou_nms,
            )
            if caixas_detectadas:
                # Seleciona a detecção com maior confiança
                caixa = max(caixas_detectadas, key=lambda c: c.confianca)
        except FileNotFoundError:
            erro = "Modelo ONNX não encontrado. Execute scripts/baixar_modelo_referencia.py ou use o método clássico."
        except Exception as det_err:
            erro = f"Falha na inferência YOLO: {det_err}"

        # Gera a visualização de bordas / detecção para o painel "1. Bordas / Detecção"
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if imagem.ndim == 3 else imagem.copy()
        suave = cv2.GaussianBlur(cinza, (5, 5), 0)
        bordas_canny = cv2.Canny(suave, params.canny_min, params.canny_max)
        if caixa is not None:
            bordas_bgr = cv2.cvtColor(bordas_canny, cv2.COLOR_GRAY2BGR)
            bordas = desenhar_caixa(
                bordas_bgr, caixa, cor=(0, 200, 0), rotulo=f"Placa {caixa.confianca:.0%}"
            )
        else:
            bordas = bordas_canny

    tempos["deteccao"] = (time.perf_counter() - t0_det) * 1000.0

    if caixa is None:
        if erro is None:
            erro = "Nenhuma placa localizada. Tente ajustar os parâmetros de detecção ou alternar o método."
        tempos["total"] = (time.perf_counter() - t0_total) * 1000.0
        return ResultadoPipeline(
            original=imagem,
            caixa=None,
            anotada=imagem.copy(),
            bordas=bordas,
            tempos_ms=tempos,
            erro=erro,
        )

    # Imagem original com caixa anotada
    rotulo = f"Placa {caixa.confianca:.0%}" if params.metodo == "yolo" else "Placa"
    anotada = desenhar_caixa(imagem, caixa, cor=(0, 200, 0), rotulo=rotulo)

    # -------------------------------------------------------------
    # ETAPA 2 e 3: RECORTE DO ROI E CORREÇÃO GEOMÉTRICA (E2 e E3)
    # -------------------------------------------------------------
    t0_prep = time.perf_counter()
    if params.metodo == "classico" and vertices is not None:
        roi = corrigir_perspectiva(
            imagem, vertices, largura=400, altura=130, margem_saida=0.09
        )
    else:
        # Extrai ROI com contexto expandido para permitir zoom-out real após retificação/rotação
        margem_ctx = max(params.margem_roi, 0.20)
        roi_ctx, (ox, oy) = recortar_roi_com_offset(imagem, caixa, margem=margem_ctx)
        caixa_em_ctx = Caixa(
            x1=caixa.x1 - ox,
            y1=caixa.y1 - oy,
            x2=caixa.x2 - ox,
            y2=caixa.y2 - oy,
            confianca=caixa.confianca,
        )
        roi_ctx_amp, escala_ctx = ampliar_roi_com_escala(roi_ctx, altura_alvo=160)
        caixa_ctx_amp = Caixa(
            x1=int(round(caixa_em_ctx.x1 * escala_ctx)),
            y1=int(round(caixa_em_ctx.y1 * escala_ctx)),
            x2=int(round(caixa_em_ctx.x2 * escala_ctx)),
            y2=int(round(caixa_em_ctx.y2 * escala_ctx)),
            confianca=caixa.confianca,
        )

        perspectiva_aplicada = False
        vertices_ctx = encontrar_vertices_placa_roi(roi_ctx_amp, caixa_esperada=caixa_ctx_amp)
        vertices_orig: np.ndarray | None = None
        if vertices_ctx is not None:
            vertices_orig = (vertices_ctx / escala_ctx) + np.array([ox, oy], dtype=np.float32)
        else:
            # Desinclina primeiro placas escorçadas para detectar os 4 cantos sem corte da caixa
            vertices_orig = detectar_vertices_pos_rotacao(
                roi_ctx_amp, caixa_ctx_amp, escala_ctx, ox, oy
            )

        if vertices_orig is not None:
            distorcido, _, _ = verificar_distorcao_geometrica(vertices_orig, threshold_graus=2.5)
            if distorcido:
                # Warp único na imagem original preserva pixels reais na margem zoom-out
                roi = corrigir_perspectiva(
                    imagem, vertices_orig, largura=None, altura=130, margem_saida=0.09
                )
                perspectiva_aplicada = True

        if not perspectiva_aplicada:
            roi_rot, angulo = corrigir_inclinacao(
                roi_ctx_amp,
                margem_zoom_out=0.09,
                caixa_interna=caixa_ctx_amp,
            )
            if abs(angulo) >= 0.8:
                roi = roi_rot
            else:
                roi = recortar_roi(imagem, caixa, margem=params.margem_roi)

    # Ampliação do ROI para dimensões confortáveis para OCR
    roi = ampliar_roi(roi, altura_alvo=130)

    # -------------------------------------------------------------
    # ETAPAS 4 a 7: PRÉ-PROCESSAMENTO (E4 a E7)
    # -------------------------------------------------------------
    roi_cinza = converter_cinza(roi)

    # Suavização Gaussiana
    suave = suavizar_gauss(roi_cinza, k=params.kernel_gauss)

    # CLAHE
    roi_clahe = aplicar_clahe(suave, clip=params.clahe_clip, grade=params.clahe_grade)

    # Binarização com inversão automática
    binarizada = binarizar(
        roi_clahe,
        modo=params.binarizacao,
        bloco=params.bloco_adaptativo,
        c=params.constante_adaptativa,
        inverter_se_fundo_claro=True,
    )
    tempos["preproc"] = (time.perf_counter() - t0_prep) * 1000.0

    # -------------------------------------------------------------
    # ETAPA 8: SEGMENTAÇÃO DE CARACTERES (E8)
    # -------------------------------------------------------------
    caracteres = segmentar_caracteres(binarizada)

    # -------------------------------------------------------------
    # ETAPAS 9 e 10: OCR E PÓS-PROCESSAMENTO (E9 e E10)
    # -------------------------------------------------------------
    t0_ocr = time.perf_counter()
    texto_bruto, confianca_ocr, erro_ocr = reconhecer_placa(
        roi_cinza=roi_cinza,
        roi_clahe=roi_clahe,
        binarizada=binarizada,
        caracteres=caracteres,
        psm_padrao=params.ocr_psm,
        robusta=params.leitura_robusta,
        modo_livre=params.modo_livre,
        engine=params.ocr_engine,
        roi=roi,
    )
    tempos["ocr"] = (time.perf_counter() - t0_ocr) * 1000.0

    # Pós-processamento e regras de formato
    pos = corrigir_por_posicao(texto_bruto, modo_livre=params.modo_livre)

    if erro_ocr and not erro:
        erro = erro_ocr

    tempos["total"] = (time.perf_counter() - t0_total) * 1000.0

    return ResultadoPipeline(
        original=imagem,
        caixa=caixa,
        anotada=anotada,
        bordas=bordas,
        roi=roi,
        roi_cinza=roi_cinza,
        roi_clahe=roi_clahe,
        binarizada=binarizada,
        caracteres=caracteres,
        texto_bruto=texto_bruto,
        texto_final=pos.texto_final,
        formato=pos.formato,
        confianca_ocr=confianca_ocr,
        substituicoes=pos.substituicoes,
        tempos_ms=tempos,
        erro=erro,
    )
