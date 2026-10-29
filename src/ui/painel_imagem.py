"""Widget de exibição de imagens com redimensionamento proporcional e título."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from src.ui.conversoes import numpy_para_pixmap


class PainelImagem(QFrame):
    """Widget que exibe uma imagem com título superior e redimensionamento proporcional."""

    def __init__(self, titulo: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap_original: QPixmap | None = None

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("painel_imagem_card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._lbl_titulo = QLabel(titulo)
        self._lbl_titulo.setObjectName("painel_imagem_titulo")
        layout.addWidget(self._lbl_titulo)

        self._lbl_imagem = QLabel("(Vazio)")
        self._lbl_imagem.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_imagem.setObjectName("painel_imagem_viewport_vazio")
        self._lbl_imagem.setMinimumSize(80, 60)
        layout.addWidget(self._lbl_imagem, stretch=1)

    def definir_imagem(self, img: np.ndarray | None) -> None:
        """Atualiza a imagem a partir de um array NumPy (BGR ou monocromático)."""
        if img is None or img.size == 0:
            self.limpar()
            return

        if self._lbl_imagem.objectName() != "painel_imagem_viewport_ativo":
            self._lbl_imagem.setObjectName("painel_imagem_viewport_ativo")
            self._lbl_imagem.style().unpolish(self._lbl_imagem)
            self._lbl_imagem.style().polish(self._lbl_imagem)

        pix = numpy_para_pixmap(img)
        self._pixmap_original = pix
        self._atualizar_escala()

    def limpar(self) -> None:
        """Limpa o painel e redefine para o estado vazio."""
        self._pixmap_original = None
        self._lbl_imagem.setPixmap(QPixmap())
        self._lbl_imagem.setText("(Vazio)")
        if self._lbl_imagem.objectName() != "painel_imagem_viewport_vazio":
            self._lbl_imagem.setObjectName("painel_imagem_viewport_vazio")
            self._lbl_imagem.style().unpolish(self._lbl_imagem)
            self._lbl_imagem.style().polish(self._lbl_imagem)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._atualizar_escala()

    def _atualizar_escala(self) -> None:
        if self._pixmap_original is None or self._pixmap_original.isNull():
            return

        tamanho_disponivel = self._lbl_imagem.size()
        if tamanho_disponivel.width() <= 10 or tamanho_disponivel.height() <= 10:
            return

        pix_escalado = self._pixmap_original.scaled(
            tamanho_disponivel,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._lbl_imagem.setText("")
        self._lbl_imagem.setPixmap(pix_escalado)
