"""Módulo de Reconhecimento Óptico de Caracteres (OCR) com FastPlateOCR e Tesseract."""

from __future__ import annotations

import collections
from typing import Any

import cv2
import numpy as np
import pytesseract

from src.config import obter_caminho_tesseract
from src.postprocess import corrigir_por_posicao, limpar_texto, validar_formato
from src.segmentation import montar_faixa_caracteres

# Configuração base do Tesseract sem heurísticas de dicionário linguístico (avaliação puramente visual)
CONFIG_BASE = (
    "--oem 3 --psm {psm} "
    "-c load_system_dawg=0 "
    "-c load_freq_dawg=0 "
    "-c load_punc_dawg=0 "
    "-c load_number_dawg=0 "
    "-c load_unambig_dawg=0 "
    "-c load_bigram_dawg=0 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def combinar_por_votacao(candidatos: list[str]) -> str | None:
    """Aplica votação por maioria caractere a caractere para candidatos de 7 caracteres."""
    validos = [c for c in candidatos if len(c) == 7]
    if not validos:
        return None
    if len(validos) == 1:
        return validos[0]

    resultado = []
    for pos in range(7):
        votos = [c[pos] for c in validos]
        mais_votado = collections.Counter(votos).most_common(1)[0][0]
        resultado.append(mais_votado)
    return "".join(resultado)


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


def _estimar_inclinacao_caracteres(caracteres: list[np.ndarray]) -> float:
    """Estima o cisalhamento (skew) mediano dos caracteres via momentos centrais."""
    skews: list[float] = []
    for char_img in caracteres:
        if char_img is None or char_img.size == 0:
            continue
        eh_fundo_escuro = np.mean(char_img) < 128
        fg = (char_img > 127 if eh_fundo_escuro else char_img < 128).astype(np.uint8) * 255
        momentos = cv2.moments(fg)
        if momentos["mu02"] > 1e-2:
            skews.append(float(momentos["mu11"] / momentos["mu02"]))
    if not skews:
        return 0.0
    return float(np.clip(np.median(skews), -0.35, 0.35))


def _deslantar_caractere(char_img: np.ndarray, skew: float) -> np.ndarray:
    """Corrige o cisalhamento do caractere sem cortar suas extremidades, em fundo claro."""
    p = cv2.bitwise_not(char_img) if np.mean(char_img) < 128 else char_img.copy()
    if abs(skew) < 0.04:
        return p

    hc, wc = p.shape[:2]
    desloc = int(np.ceil(0.5 * hc * abs(skew)))
    tela_w = wc + 2 * desloc
    amplo = cv2.copyMakeBorder(p, 0, 0, desloc, desloc, cv2.BORDER_CONSTANT, value=255)
    matriz = np.float32([[1.0, -skew, 0.5 * hc * skew], [0.0, 1.0, 0.0]])
    rot = cv2.warpAffine(
        amplo,
        matriz,
        (tela_w, hc),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    cols = np.where(np.min(rot, axis=0) < 200)[0]
    if len(cols) > 0:
        c1 = max(0, int(cols[0]) - 5)
        c2 = min(tela_w, int(cols[-1]) + 6)
        if (c2 - c1) >= wc:
            return rot[:, c1:c2]
        pad_tot = wc - (c2 - c1)
        pl = pad_tot // 2
        return cv2.copyMakeBorder(
            rot[:, c1:c2], 0, 0, pl, pad_tot - pl, cv2.BORDER_CONSTANT, value=255
        )
    return rot


def _montar_faixa_deslantada(
    caracteres: list[np.ndarray], skew: float, escala: float = 0.7
) -> np.ndarray | None:
    """Monta faixa horizontal com caracteres deslantados na altura-alvo ideal ao LSTM."""
    if not caracteres:
        return None

    des = [_deslantar_caractere(c, skew) for c in caracteres]
    espacamento = 12
    altura_c = des[0].shape[0]
    largura_tot = sum(d.shape[1] for d in des) + (len(des) - 1) * espacamento + 48
    faixa = np.full((altura_c + 40, largura_tot), 255, dtype=np.uint8)

    x = 24
    for d in des:
        w = d.shape[1]
        faixa[20 : 20 + altura_c, x : x + w] = d
        x += w + espacamento

    if escala != 1.0:
        faixa = cv2.resize(faixa, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    return faixa


def _ler_caracteres_isolados(
    caracteres: list[np.ndarray], skew: float, escala: float = 0.7
) -> str:
    """Lê cada caractere segmentado isoladamente (PSM 10, fallback PSM 8) para evitar viés."""
    saida: list[str] = []
    for char_img in caracteres:
        glifo = _deslantar_caractere(char_img, skew)
        if escala != 1.0:
            glifo = cv2.resize(glifo, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        txt = executar_tesseract(glifo, psm=10)
        if not txt:
            txt = executar_tesseract(glifo, psm=8)
        saida.append(txt[:1] if txt else "?")
    return "".join(saida)


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
    """Executa o OCR na placa recortada, com suporte a FastPlateOCR (ONNX) ou Tesseract.

    Retorna: (texto_bruto, confianca_estimada, mensagem_de_erro)
    """
    if engine == "fast_plate_ocr":
        img_alvo = roi if roi is not None else roi_clahe
        return reconhecer_com_fastplate(img_alvo, modo_livre=modo_livre)

    disponivel, info = verificar_tesseract_disponivel()
    if not disponivel:
        return "", None, info

    try:
        tem_sete_chars = bool(caracteres and len(caracteres) == 7)
        faixa = montar_faixa_caracteres(caracteres) if tem_sete_chars else None
        skew_chars = (
            _estimar_inclinacao_caracteres(caracteres)
            if (caracteres and len(caracteres) >= 4)
            else 0.0
        )
        faixa_norm = (
            _montar_faixa_deslantada(caracteres, skew_chars, escala=0.7)
            if tem_sete_chars
            else None
        )
        roi_miolo = recortar_miolo_placa(roi_clahe)
        roi_miolo_cinza = recortar_miolo_placa(roi_cinza)

        txt_faixa_norm = ""
        if faixa_norm is not None:
            try:
                txt_faixa_norm = executar_tesseract(faixa_norm, psm=psm_padrao)
            except Exception:
                txt_faixa_norm = ""

        # Leitura por glifo isolado quando os 7 caracteres foram segmentados no modo BR:
        # evita alucinação sequencial (ex: 4G21 -> 4621 ou G -> GE)
        if tem_sete_chars and not modo_livre and caracteres is not None:
            txt_iso = _ler_caracteres_isolados(caracteres, skew_chars, escala=0.7)
            if "?" in txt_iso and len(txt_faixa_norm) == 7:
                mesclado = [
                    (txt_faixa_norm[idx] if ch == "?" else ch)
                    for idx, ch in enumerate(txt_iso)
                ]
                txt_iso = "".join(mesclado)

            if "?" not in txt_iso:
                pos_iso = corrigir_por_posicao(txt_iso, modo_livre=False)
                if pos_iso.valido:
                    if (
                        len(txt_faixa_norm) == 7
                        and validar_formato(txt_faixa_norm) in ("mercosul", "antigo")
                        and txt_faixa_norm == pos_iso.texto_final
                    ):
                        return txt_faixa_norm, 1.0 if robusta else 0.99, None
                    return txt_iso, 0.99 if robusta else 0.98, None

        if not robusta:
            candidatos_validos: list[str] = []
            todos_candidatos_7: list[str] = []

            for cand_txt in ([txt_faixa_norm] if txt_faixa_norm else []):
                if not modo_livre and validar_formato(cand_txt) in ("mercosul", "antigo"):
                    candidatos_validos.append(cand_txt)
                if len(cand_txt) == 7:
                    todos_candidatos_7.append(cand_txt)

            # 1. Faixa de Caracteres Segmentados
            if faixa is not None:
                for psm in [psm_padrao, 6, 11]:
                    try:
                        txt = executar_tesseract(faixa, psm=psm)
                        if txt:
                            if not modo_livre and validar_formato(txt) in ("mercosul", "antigo"):
                                candidatos_validos.append(txt)
                            if len(txt) == 7:
                                todos_candidatos_7.append(txt)
                    except Exception:
                        pass

            # 2. Recortes de Miolo (Cinza nativo primeiro, depois CLAHE)
            for img_cand, psm in [
                (roi_miolo_cinza, psm_padrao),
                (roi_miolo_cinza, 8),
                (roi_miolo, psm_padrao),
                (roi_miolo, 8),
            ]:
                try:
                    txt = executar_tesseract(img_cand, psm=psm)
                    if txt:
                        if not modo_livre and validar_formato(txt) in ("mercosul", "antigo"):
                            candidatos_validos.append(txt)
                        if len(txt) == 7:
                            todos_candidatos_7.append(txt)
                except Exception:
                    pass

            if candidatos_validos:
                consenso = combinar_por_votacao(candidatos_validos)
                if consenso and (modo_livre or validar_formato(consenso) in ("mercosul", "antigo")):
                    return consenso, 0.99, None
                return candidatos_validos[0], 0.98, None

            if todos_candidatos_7:
                consenso_7 = combinar_por_votacao(todos_candidatos_7)
                if consenso_7:
                    return consenso_7, 0.95, None
                return todos_candidatos_7[0], 0.92, None

            if modo_livre and txt_faixa_norm:
                return txt_faixa_norm, 0.90, None

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

        # Modo Robusto (Multi-tentativa): avalia múltiplas faixas, PSMs e variações do ROI
        leituras: list[str] = []
        leituras_segmentadas: list[str] = []

        if txt_faixa_norm:
            leituras_segmentadas.append(txt_faixa_norm)
            leituras.append(txt_faixa_norm)

        if faixa_norm is not None:
            for psm in [6, 7, 8]:
                if psm == psm_padrao and txt_faixa_norm:
                    continue
                try:
                    res = executar_tesseract(faixa_norm, psm=psm)
                    if res:
                        leituras_segmentadas.append(res)
                        leituras.append(res)
                except Exception:
                    pass

        if faixa is not None:
            for psm in [psm_padrao, 6, 7, 8, 11, 13]:
                try:
                    res = executar_tesseract(faixa, psm=psm)
                    if res:
                        leituras_segmentadas.append(res)
                        leituras.append(res)
                except Exception:
                    pass

        if not modo_livre:
            validas_seg = [
                cand
                for cand in leituras_segmentadas
                if validar_formato(cand) in ("mercosul", "antigo")
            ]
            if validas_seg and tem_sete_chars:
                consenso_seg = combinar_por_votacao(validas_seg)
                if consenso_seg and validar_formato(consenso_seg) in ("mercosul", "antigo"):
                    return consenso_seg, 1.0, None
                mais_comum_seg = collections.Counter(validas_seg).most_common(1)[0][0]
                return mais_comum_seg, 0.99, None

        # Tentativas com miolo e variações do ROI
        variacoes = [
            ("MioloCinza", roi_miolo_cinza),
            ("MioloCLAHE", roi_miolo),
            ("CLAHE", roi_clahe),
            ("Binarizada", binarizada),
            ("Cinza", roi_cinza),
        ]

        for _, img_var in variacoes:
            for psm in [psm_padrao, 6, 7, 8, 13]:
                try:
                    res = executar_tesseract(img_var, psm=psm)
                    if res:
                        leituras.append(res)
                except Exception:
                    continue

        if not leituras:
            return "", None, "Nenhum caractere reconhecido nas tentativas"

        # 1. Prioridade: candidatos diretamente válidos combinados por consenso
        if not modo_livre:
            validas = [cand for cand in leituras if validar_formato(cand) in ("mercosul", "antigo")]
            if validas:
                consenso_valido = combinar_por_votacao(validas)
                if consenso_valido and validar_formato(consenso_valido) in ("mercosul", "antigo"):
                    return consenso_valido, 1.0, None
                mais_comum = collections.Counter(validas).most_common(1)[0][0]
                return mais_comum, 0.99, None

        # 2. Prioridade: candidato de tamanho 7 com votação por maioria
        candidatos_7 = [cand for cand in leituras if len(cand) == 7]
        if candidatos_7:
            consenso_7 = combinar_por_votacao(candidatos_7)
            if consenso_7:
                return consenso_7, 0.95, None
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
