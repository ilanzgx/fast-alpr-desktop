"""Módulo de Reconhecimento Óptico de Caracteres (OCR) com Tesseract."""

from __future__ import annotations

import collections

import cv2
import numpy as np
import pytesseract

from src.config import obter_caminho_tesseract
from src.postprocess import limpar_texto, validar_formato

# Configuração base de caracteres permitidos para placas
CONFIG_BASE = "--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def verificar_tesseract_disponivel() -> tuple[bool, str]:
    """Verifica se o executável do Tesseract está disponível no sistema ou na pasta local."""
    caminho = obter_caminho_tesseract()
    if caminho:
        pytesseract.pytesseract.tesseract_cmd = caminho
        return True, caminho
    return False, "Tesseract OCR não encontrado no sistema ou em tools/tesseract/"


def executar_tesseract(imagem: np.ndarray, psm: int = 7) -> str:
    """Executa a leitura do Tesseract sobre uma única imagem com whitelist alfanumérica."""
    disponivel, caminho_ou_erro = verificar_tesseract_disponivel()
    if not disponivel:
        raise RuntimeError(caminho_ou_erro)

    # Adiciona borda branca de 10px para evitar falhas do Tesseract com caracteres colados na margem
    if imagem.ndim == 2:
        img_borda = cv2.copyMakeBorder(imagem, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    else:
        img_borda = cv2.copyMakeBorder(imagem, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    cfg = CONFIG_BASE.format(psm=psm)
    texto = pytesseract.image_to_string(img_borda, config=cfg)
    return limpar_texto(texto)


def reconhecer_placa(
    roi_cinza: np.ndarray,
    roi_clahe: np.ndarray,
    binarizada: np.ndarray,
    psm_padrao: int = 7,
    robusta: bool = False,
    modo_livre: bool = False,
) -> tuple[str, float | None, str | None]:
    """Executa o OCR na placa recortada, com suporte a modo simples ou multi-tentativa (robusta).
    
    Retorna: (texto_bruto, confianca_estimada, mensagem_de_erro)
    """
    disponivel, info = verificar_tesseract_disponivel()
    if not disponivel:
        return "", None, info

    try:
        if not robusta:
            # Modo Simples: executa na versão com CLAHE (que costuma ter o melhor contraste)
            texto = executar_tesseract(roi_clahe, psm=psm_padrao)
            return texto, 1.0 if texto else None, None

        # Modo Robusto (Multi-tentativa): gera variações de tratamentos e PSMs
        leituras: list[str] = []
        variacoes = [
            ("CLAHE", roi_clahe),
            ("Cinza", roi_cinza),
            ("Binarizada", binarizada),
        ]

        for _, img_var in variacoes:
            for psm in [7, 13]:
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

        # 3. Fallback: leitura mais frequente de qualquer tamanho plausível
        plausiveis = [cand for cand in leituras if 4 <= len(cand) <= 8]
        if plausiveis:
            mais_comum = collections.Counter(plausiveis).most_common(1)[0][0]
            return mais_comum, 0.7, None

        # Último recurso
        return leituras[0], 0.5, None

    except Exception as e:
        return "", None, f"Erro no motor Tesseract: {e}"
