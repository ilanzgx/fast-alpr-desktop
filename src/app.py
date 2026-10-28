"""Ponto de entrada da aplicação desktop ALPR."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.pipeline import executar_pipeline
from src.ui.janela_principal import JanelaPrincipal


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Reconhecimento de Placas Veiculares")
    app.setOrganizationName("Projeto ALPR Didático")

    janela = JanelaPrincipal(funcao_pipeline=executar_pipeline)
    janela.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
