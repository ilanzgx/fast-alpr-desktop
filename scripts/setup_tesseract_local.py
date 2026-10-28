"""Script utilitário para auxílio na configuração do Tesseract OCR local (tools/tesseract/)."""

from __future__ import annotations

from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools" / "tesseract"
EXE_LOCAL = TOOLS_DIR / "tesseract.exe"


def verificar_ou_orientar() -> None:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if EXE_LOCAL.exists():
        print("=" * 60)
        print("TESSERACT LOCAL ENCONTRADO!")
        print("=" * 60)
        print(f"Executável: {EXE_LOCAL}")
        print("O sistema detectará esta instalação automaticamente.")
        return

    print("=" * 60)
    print("CONFIGURAÇÃO DO TESSERACT OCR (WINDOWS)")
    print("=" * 60)
    print(f"Destino local esperado: {TOOLS_DIR}")
    print()
    print("Você pode disponibilizar o Tesseract de 3 formas:")
    print()
    print("1. [Portátil Local no Projeto - Recomendado]:")
    print("   Baixe o instalador do UB Mannheim:")
    print("   https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.4.0.20240606.exe")
    print("   Abra o executável com 7-Zip ou WinRAR e extraia os arquivos")
    print("   (tesseract.exe, dlls e a pasta tessdata) diretamente em:")
    print(f"   {TOOLS_DIR}")
    print()
    print("2. [Instalação pelo Windows Package Manager (winget)]:")
    print("   Execute no terminal:")
    print("   winget install UB-Mannheim.TesseractOCR")
    print()
    print("3. [Variável de Ambiente]:")
    print("   Defina TESSERACT_CMD apontando para o seu executável tesseract.exe")
    print("=" * 60)


if __name__ == "__main__":
    verificar_ou_orientar()
