"""Worker assíncrono para execução do pipeline fora da thread de interface."""

from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.config import ParametrosPipeline


class PipelineWorker(QThread):
    """Executa o pipeline ALPR em segundo plano sem congelar a UI."""

    resultado_pronto = Signal(object)
    erro_ocorrido = Signal(str)

    def __init__(
        self,
        imagem: np.ndarray,
        parametros: ParametrosPipeline,
        funcao_pipeline,
        parent=None,
    ) -> None:
        super().__init__(parent)
        # Cria cópia para isolamento de memória entre threads
        self._imagem = imagem.copy()
        self._parametros = parametros
        self._funcao_pipeline = funcao_pipeline

    def run(self) -> None:
        try:
            resultado = self._funcao_pipeline(self._imagem, self._parametros)
            self.resultado_pronto.emit(resultado)
        except Exception as e:
            msg = f"Falha durante o processamento: {e}\n{traceback.format_exc()}"
            self.erro_ocorrido.emit(msg)
