"""Módulo de Reconhecimento Óptico de Caracteres (OCR) com FastPlateOCR (ONNX)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.postprocess import limpar_texto


def recortar_miolo_placa(
    roi: np.ndarray,
    corte_topo_pct: float = 0.15,
    corte_base_pct: float = 0.05,
    corte_lat_pct: float = 0.08,
) -> np.ndarray:
    """Recorta a área central útil da placa, removendo a moldura externa e cabeçalho BRASIL/estado."""
    if roi is None or roi.size == 0:
        return roi
    h, w = roi.shape[:2]
    y1 = int(h * corte_topo_pct)
    y2 = max(y1 + 10, int(h * (1.0 - corte_base_pct)))
    x1 = int(w * corte_lat_pct)
    x2 = max(x1 + 20, int(w * (1.0 - corte_lat_pct)))
    return roi[y1:y2, x1:x2]


# Cache global para o modelo ONNX FastPlateOCR
_RECOGNIZER_FASTPLATE: Any | None = None


def obter_recognizer_fastplate() -> Any:
    """Retorna uma instância singleton de LicensePlateRecognizer para inferência rápida."""
    global _RECOGNIZER_FASTPLATE
    if _RECOGNIZER_FASTPLATE is None:
        from fast_plate_ocr import LicensePlateRecognizer

        from src.config import ROOT_DIR

        local_onnx = ROOT_DIR / "models" / "ocr" / "cct_s_v2_global.onnx"
        local_yaml = ROOT_DIR / "models" / "ocr" / "cct_s_v2_global_plate_config.yaml"
        if local_onnx.exists() and local_yaml.exists():
            _RECOGNIZER_FASTPLATE = LicensePlateRecognizer(
                onnx_model_path=str(local_onnx),
                plate_config_path=str(local_yaml),
                device="cpu",
            )
        else:
            _RECOGNIZER_FASTPLATE = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")
    return _RECOGNIZER_FASTPLATE


def verificar_fastplate_disponivel() -> tuple[bool, str]:
    """Verifica se o pacote fast_plate_ocr está disponível no ambiente."""
    try:
        from fast_plate_ocr import LicensePlateRecognizer  # noqa: F401

        return True, "FastPlateOCR disponível"
    except ImportError:
        return False, "Pacote 'fast_plate_ocr' não instalado. Execute: uv add fast-plate-ocr"
    except Exception as e:
        return False, f"Falha ao carregar FastPlateOCR: {e}"


def reconhecer_com_fastplate(
    roi: np.ndarray,
    modo_livre: bool = False,
) -> tuple[str, float | None, str | None]:
    """Executa o reconhecimento de placa utilizando a rede neural FastPlateOCR (ONNX).

    Espera a imagem recortada da placa (ROI BGR ou tons de cinza).
    Garante conversão para 3 canais e calcula confiança média dos caracteres.

    Retorna: (texto_bruto, confianca_estimada, mensagem_de_erro)
    """
    if roi is None or roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 10:
        return "", None, "ROI da placa inválido ou muito pequeno para FastPlateOCR"

    try:
        recognizer = obter_recognizer_fastplate()
    except Exception as e:
        return "", None, f"Erro ao inicializar FastPlateOCR: {e}"

    # FastPlateOCR exige imagem com 3 canais (H, W, 3)
    img = roi
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    try:
        preds = recognizer.run(img, return_confidence=True)
        if not preds:
            return "", None, "Nenhum resultado retornado pelo FastPlateOCR"

        pred = preds[0]
        texto = limpar_texto(pred.plate)

        confianca: float | None = None
        if pred.has_confidence and pred.char_probs is not None and len(pred.char_probs) > 0:
            confianca = float(np.mean(pred.char_probs))
        elif texto:
            confianca = 0.95

        return texto, confianca, None
    except Exception as e:
        return "", None, f"Erro na inferência FastPlateOCR: {e}"


def reconhecer_placa(
    roi_cinza: np.ndarray,
    roi_clahe: np.ndarray,
    binarizada: np.ndarray,
    caracteres: list[np.ndarray] | None = None,
    psm_padrao: int = 7,
    robusta: bool = False,
    modo_livre: bool = False,
    engine: str = "fast_plate_ocr",
    roi: np.ndarray | None = None,
) -> tuple[str, float | None, str | None]:
    """Executa o OCR na placa recortada utilizando a rede neural FastPlateOCR (ONNX).

    Retorna: (texto_bruto, confianca_estimada, mensagem_de_erro)
    """
    img_alvo = roi if roi is not None else roi_clahe
    return reconhecer_com_fastplate(img_alvo, modo_livre=modo_livre)
