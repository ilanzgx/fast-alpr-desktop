"""Script utilitário para download do modelo pré-treinado YOLOv8 em ONNX."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# URL do modelo YOLOv8 treinado em detecção de placas (keremberke / ml-debi)
URL_MODELO = "https://huggingface.co/ml-debi/yolov8-license-plate-detection/resolve/main/best.onnx"
DESTINO = Path(__file__).resolve().parent.parent / "models" / "placas.onnx"


def progresso(blocos: int, tamanho_bloco: int, total: int) -> None:
    baixado = blocos * tamanho_bloco
    if total > 0:
        pct = (baixado / total) * 100
        mb_baixado = baixado / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        sys.stdout.write(f"\rBaixando modelo YOLO: {pct:.1f}% ({mb_baixado:.1f}MB / {mb_total:.1f}MB)")
        sys.stdout.flush()


def baixar_modelo() -> None:
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    if DESTINO.exists():
        print(f"O modelo já existe em: {DESTINO}")
        return

    print("Iniciando download do modelo pré-treinado YOLO...")
    print(f"Origem: {URL_MODELO}")
    print(f"Destino: {DESTINO}")
    try:
        urllib.request.urlretrieve(URL_MODELO, str(DESTINO), reporthook=progresso)
        print("\nDownload concluído com sucesso!")
    except Exception as e:
        print(f"\nFalha ao baixar o modelo: {e}")
        if DESTINO.exists():
            DESTINO.unlink()
        raise


if __name__ == "__main__":
    baixar_modelo()
