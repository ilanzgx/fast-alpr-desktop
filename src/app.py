"""Ponto de entrada da aplicação desktop ALPR."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication

from src.pipeline import executar_pipeline
from src.ui.janela_principal import JanelaPrincipal


def main() -> None:
    # Garante renderização consistente e dimensionamento correto no Windows
    if sys.platform == "win32":
        os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("alpr.placas.app")
        except Exception:
            pass

    app = QApplication(sys.argv)

    # Força o estilo Fusion para evitar que o tema escuro nativo do Windows 11
    # sobrescreva paletas e cause blocos pretos nas áreas de rolagem e controles
    app.setStyle("Fusion")

    # Alinha o esquema de cores da aplicação ao tema claro
    if hasattr(QGuiApplication, "styleHints"):
        try:
            QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Light)
        except Exception:
            pass

    app.setApplicationName("Reconhecimento de Placas Veiculares")
    app.setOrganizationName("Projeto ALPR Didático")

    # Ícone da aplicação e barra de tarefas
    caminho_icone = Path(__file__).resolve().parent / "ui" / "assets" / "icone.png"
    if not caminho_icone.exists() and getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        caminho_icone = Path(sys._MEIPASS) / "src" / "ui" / "assets" / "icone.png"
    if caminho_icone.exists():
        app.setWindowIcon(QIcon(str(caminho_icone)))

    janela = JanelaPrincipal(funcao_pipeline=executar_pipeline)
    if caminho_icone.exists():
        janela.setWindowIcon(QIcon(str(caminho_icone)))
    janela.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
