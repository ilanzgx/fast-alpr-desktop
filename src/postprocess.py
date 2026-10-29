"""Pós-processamento de OCR: validação de formatos de placa e correção posicional."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Expressões regulares dos formatos brasileiros
REGEX_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
REGEX_ANTIGO = re.compile(r"^[A-Z]{3}[0-9]{4}$")

# Mapas de confusão típicos do OCR
LETRA_PARA_DIGITO = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
    "T": "7",
}

DIGITO_PARA_LETRA = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
    "7": "T",
}


@dataclass(frozen=True)
class ResultadoPosProcessamento:
    """Resultado do pós-processamento e validação de formato."""

    texto_final: str
    formato: str  # "mercosul" | "antigo" | "invalido" | "livre"
    substituicoes: int
    valido: bool


def limpar_texto(texto: str) -> str:
    """Remove caracteres especiais, pontuações e espaços, convertendo para maiúsculas."""
    if not texto:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(texto).upper())


def validar_formato(texto: str) -> str:
    """Identifica se o texto atende rigorosamente a um dos formatos brasileiros."""
    if REGEX_MERCOSUL.match(texto):
        return "mercosul"
    if REGEX_ANTIGO.match(texto):
        return "antigo"
    return "invalido"


def corrigir_por_posicao(texto: str, modo_livre: bool = False) -> ResultadoPosProcessamento:
    """Aplica regras de pós-processamento e correção contextual posicional.
    
    Para placas de 7 caracteres no modo brasileiro, avalia as hipóteses
    Mercosul (LLL D L DD) e Antigo (LLL D D DD), selecionando a de menor custo de substituição.
    """
    limpo = limpar_texto(texto)
    if not limpo:
        return ResultadoPosProcessamento(
            texto_final="", formato="invalido", substituicoes=0, valido=False
        )

    # Modo Livre: preserva a leitura bruta limpa
    if modo_livre:
        formato_encontrado = validar_formato(limpo)
        return ResultadoPosProcessamento(
            texto_final=limpo,
            formato="livre" if formato_encontrado == "invalido" else formato_encontrado,
            substituicoes=0,
            valido=True,
        )

    # Se já é válido exatamente, não altera
    formato_exato = validar_formato(limpo)
    if formato_exato != "invalido":
        return ResultadoPosProcessamento(
            texto_final=limpo, formato=formato_exato, substituicoes=0, valido=True
        )

    # Avaliação das duas hipóteses: Mercosul vs Antigo
    # Mercosul: [L, L, L, D, L, D, D]
    # Antigo:   [L, L, L, D, D, D, D]
    padrao_mercosul = ["L", "L", "L", "D", "L", "D", "D"]
    padrao_antigo = ["L", "L", "L", "D", "D", "D", "D"]

    def avaliar_hipotese(candidato: str, padrao: list[str]) -> tuple[str, int, bool]:
        chars = list(candidato)
        trocas = 0
        possivel = True

        for i, esperado in enumerate(padrao):
            c = chars[i]
            if esperado == "L":
                if c.isalpha():
                    continue
                elif c in DIGITO_PARA_LETRA:
                    chars[i] = DIGITO_PARA_LETRA[c]
                    trocas += 1
                else:
                    possivel = False
                    trocas += 10
            else:  # "D"
                if c.isdigit():
                    continue
                elif c in LETRA_PARA_DIGITO:
                    chars[i] = LETRA_PARA_DIGITO[c]
                    trocas += 1
                else:
                    possivel = False
                    trocas += 10

        return "".join(chars), trocas, possivel

    # Se tamanho exato 7: avalia as duas hipóteses diretamente
    if len(limpo) == 7:
        cand_merc, trocas_merc, possivel_merc = avaliar_hipotese(limpo, padrao_mercosul)
        cand_antigo, trocas_antigo, possivel_antigo = avaliar_hipotese(limpo, padrao_antigo)

        # Critério de seleção: menor número de substituições
        if possivel_merc and (not possivel_antigo or trocas_merc < trocas_antigo):
            return ResultadoPosProcessamento(
                texto_final=cand_merc, formato="mercosul", substituicoes=trocas_merc, valido=True
            )
        elif possivel_antigo and (not possivel_merc or trocas_antigo < trocas_merc):
            return ResultadoPosProcessamento(
                texto_final=cand_antigo, formato="antigo", substituicoes=trocas_antigo, valido=True
            )
        elif possivel_merc and possivel_antigo and trocas_merc == trocas_antigo:
            return ResultadoPosProcessamento(
                texto_final=cand_merc, formato="mercosul", substituicoes=trocas_merc, valido=True
            )

    # Se o texto for mais longo que 7 (ruído de bordas/molduras, ex: "FPOX46215" ou "POX4G215")
    # Busca por janela deslizante de 7 caracteres
    if len(limpo) > 7:
        # 1. Verifica se há alguma substring de 7 caracteres que já seja exatamente válida
        for i in range(len(limpo) - 7 + 1):
            sub = limpo[i : i + 7]
            fmt = validar_formato(sub)
            if fmt != "invalido":
                return ResultadoPosProcessamento(
                    texto_final=sub, formato=fmt, substituicoes=0, valido=True
                )

        # 2. Avalia correção contextual em cada janela de 7 caracteres
        candidatos_janela = []
        for i in range(len(limpo) - 7 + 1):
            sub = limpo[i : i + 7]
            cm, tm, pm = avaliar_hipotese(sub, padrao_mercosul)
            ca, ta, pa = avaliar_hipotese(sub, padrao_antigo)
            if pm and tm <= 2:
                candidatos_janela.append((tm, 0, cm, "mercosul"))
            if pa and ta <= 2:
                candidatos_janela.append((ta, 1, ca, "antigo"))

        if candidatos_janela:
            # Ordena por menor número de trocas e desempata por Mercosul
            candidatos_janela.sort(key=lambda item: (item[0], item[1]))
            melhor = candidatos_janela[0]
            return ResultadoPosProcessamento(
                texto_final=melhor[2],
                formato=melhor[3],
                substituicoes=melhor[0],
                valido=True,
            )

    # Não foi possível recuperar com o mapa de confusão
    return ResultadoPosProcessamento(
        texto_final=limpo, formato="invalido", substituicoes=0, valido=False
    )

