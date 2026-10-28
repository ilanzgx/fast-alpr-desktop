"""Funções utilitárias de I/O de imagens e anotações visuais."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import Caixa


def ler_imagem(caminho: str | Path) -> np.ndarray:
    """Lê uma imagem do disco tratando caminhos com acentos e caracteres especiais no Windows.
    
    Retorna a imagem em formato BGR (padrão OpenCV).
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de imagem não encontrado: {caminho}")

    # np.fromfile + cv2.imdecode evita falhas de caminho do cv2.imread no Windows
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"Não foi possível decodificar a imagem: {caminho}")

    return imagem


def salvar_imagem(caminho: str | Path, imagem: np.ndarray) -> None:
    """Salva uma imagem no disco com suporte a caminhos com acentos no Windows."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ext = caminho.suffix.lower() or ".png"
    sucesso, buffer = cv2.imencode(ext, imagem)
    if not sucesso:
        raise ValueError(f"Falha ao codificar a imagem para {ext}")
    buffer.tofile(str(caminho))


def desenhar_caixa(
    imagem: np.ndarray,
    caixa: Caixa,
    cor: tuple[int, int, int] = (0, 200, 0),
    espessura: int = 3,
    rotulo: str | None = None,
) -> np.ndarray:
    """Desenha a caixa delimitadora e rótulo opcional na imagem anotada."""
    anotada = imagem.copy()
    cv2.rectangle(
        anotada,
        (caixa.x1, caixa.y1),
        (caixa.x2, caixa.y2),
        cor,
        espessura,
    )

    if rotulo:
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        escala = 0.6
        esp_texto = 2
        (w_texto, h_texto), baseline = cv2.getTextSize(rotulo, fonte, escala, esp_texto)

        # Fundo do texto para legibilidade
        y_topo = max(0, caixa.y1 - h_texto - baseline - 4)
        cv2.rectangle(
            anotada,
            (caixa.x1, y_topo),
            (caixa.x1 + w_texto + 8, y_topo + h_texto + baseline + 4),
            cor,
            -1,
        )
        cv2.putText(
            anotada,
            rotulo,
            (caixa.x1 + 4, y_topo + h_texto + 2),
            fonte,
            escala,
            (255, 255, 255),
            esp_texto,
            cv2.LINE_AA,
        )

    return anotada
