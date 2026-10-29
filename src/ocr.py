"""Módulo de Reconhecimento Óptico de Caracteres (OCR) com Tesseract."""

from __future__ import annotations

import collections

import cv2
import numpy as np
import pytesseract

from src.config import obter_caminho_tesseract
from src.postprocess import limpar_texto, validar_formato
from src.segmentation import montar_faixa_caracteres

# Configuração base de caracteres permitidos para placas
CONFIG_BASE = "--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


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


def verificar_tesseract_disponivel() -> tuple[bool, str]:
    """Verifica se o executável do Tesseract está disponível no sistema ou na pasta local."""
    caminho = obter_caminho_tesseract()
    if caminho:
        pytesseract.pytesseract.tesseract_cmd = caminho
        return True, caminho
    return False, "Tesseract OCR não encontrado no sistema ou em tools/tesseract/"


def executar_tesseract(imagem: np.ndarray, psm: int = 7) -> str:
    """Executa a leitura do Tesseract sobre uma única imagem com whitelist alfanumérica.
    
    Garante que a imagem seja apresentada com texto escuro sobre fundo claro,
    que é o formato ótimo esperado pelo motor Tesseract.
    """
    disponivel, caminho_ou_erro = verificar_tesseract_disponivel()
    if not disponivel:
        raise RuntimeError(caminho_ou_erro)

    img_proc = imagem.copy()

    # Normalização de polaridade: Tesseract exige texto escuro sobre fundo claro
    if img_proc.ndim == 2:
        # Se mais de 50% dos pixels forem escuros, inverte para fundo claro
        if np.count_nonzero(img_proc < 128) > (img_proc.size / 2):
            img_proc = cv2.bitwise_not(img_proc)
        img_borda = cv2.copyMakeBorder(img_proc, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    else:
        img_borda = cv2.copyMakeBorder(img_proc, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    cfg = CONFIG_BASE.format(psm=psm)
    texto = pytesseract.image_to_string(img_borda, config=cfg)
    return limpar_texto(texto)


def reconhecer_placa(
    roi_cinza: np.ndarray,
    roi_clahe: np.ndarray,
    binarizada: np.ndarray,
    caracteres: list[np.ndarray] | None = None,
    psm_padrao: int = 7,
    robusta: bool = False,
    modo_livre: bool = False,
) -> tuple[str, float | None, str | None]:
    """Executa o OCR na placa recortada, com suporte a modo simples ou multi-tentativa (robusta).
    
    Aproveita os caracteres previamente segmentados na Etapa 8 para montar uma faixa
    limpa isenta de bordas, QR code e parafusos, garantindo máxima acurácia.
    
    Retorna: (texto_bruto, confianca_estimada, mensagem_de_erro)
    """
    disponivel, info = verificar_tesseract_disponivel()
    if not disponivel:
        return "", None, info

    try:
        faixa = (
            montar_faixa_caracteres(caracteres)
            if (caracteres and len(caracteres) == 7)
            else None
        )
        roi_miolo = recortar_miolo_placa(roi_clahe)
        roi_miolo_cinza = recortar_miolo_placa(roi_cinza)

        if not robusta:
            candidatos_diretos: list[str] = []

            # 1. Prioridade Absoluta: Busca por formato diretamente válido (Mercosul ou Antigo)
            # Avaliamos o miolo (que preserva o anti-aliasing nativo da placa sem molduras ou QR code)
            for img_cand, psm in [
                (roi_miolo, 8),
                (roi_miolo, psm_padrao),
                (roi_miolo_cinza, 8),
                (roi_miolo, 13),
            ]:
                txt = executar_tesseract(img_cand, psm=psm)
                if txt:
                    if not modo_livre and validar_formato(txt) in ("mercosul", "antigo"):
                        return txt, 0.99, None
                    candidatos_diretos.append(txt)

            # Avalia também a faixa dos caracteres segmentados (caso existam 7)
            if faixa is not None:
                for psm in [8, 7]:
                    txt = executar_tesseract(faixa, psm=psm)
                    if txt:
                        if not modo_livre and validar_formato(txt) in ("mercosul", "antigo"):
                            return txt, 0.98, None
                        candidatos_diretos.append(txt)

            # 2. Se nenhum foi 100% perfeito de primeira, seleciona candidato de tamanho 7
            candidatos_7 = [c for c in candidatos_diretos if len(c) == 7]
            if candidatos_7:
                return candidatos_7[0], 0.95, None

            # 3. Fallback: Leitura clássica do ROI integral
            texto = executar_tesseract(roi_clahe, psm=psm_padrao)
            if not texto or len(texto) < 4:
                alt_bin = executar_tesseract(binarizada, psm=psm_padrao)
                if len(alt_bin) >= 4:
                    return alt_bin, 0.85, None

                alt_cinza = executar_tesseract(roi_cinza, psm=psm_padrao)
                if len(alt_cinza) >= 4:
                    return alt_cinza, 0.80, None

            return texto, 0.85 if texto else None, None

        # Modo Robusto (Multi-tentativa): gera variações de tratamentos, recortes e PSMs
        leituras: list[str] = []

        # Tentativas com faixa segmentada
        if faixa is not None:
            for psm in [8, 7, 13]:
                try:
                    res = executar_tesseract(faixa, psm=psm)
                    if res:
                        leituras.append(res)
                except Exception:
                    pass

        # Tentativas com miolo e variações do ROI
        variacoes = [
            ("Miolo", roi_miolo),
            ("CLAHE", roi_clahe),
            ("Binarizada", binarizada),
            ("Cinza", roi_cinza),
        ]

        for _, img_var in variacoes:
            for psm in [8, 7, 13]:
                try:
                    res = executar_tesseract(img_var, psm=psm)
                    if res:
                        leituras.append(res)
                except Exception:
                    continue

        if not leituras:
            return "", None, "Nenhum caractere reconhecido nas tentativas"

        # 1. Prioridade: se alguma leitura já for formato brasileiro válido
        if not modo_livre:
            for leitura in leituras:
                if validar_formato(leitura) in ("mercosul", "antigo"):
                    return leitura, 1.0, None

        # 2. Prioridade: candidato de tamanho 7 mais frequente (consenso)
        candidatos_7 = [cand for cand in leituras if len(cand) == 7]
        if candidatos_7:
            mais_comum = collections.Counter(candidatos_7).most_common(1)[0][0]
            return mais_comum, 0.9, None

        # 3. Fallback: leitura mais frequente de qualquer tamanho plausível (4 a 8 caracteres)
        plausiveis = [cand for cand in leituras if 4 <= len(cand) <= 8]
        if plausiveis:
            mais_comum = collections.Counter(plausiveis).most_common(1)[0][0]
            return mais_comum, 0.7, None

        # Último recurso
        return leituras[0], 0.5, None

    except Exception as e:
        return "", None, f"Erro no motor Tesseract: {e}"
