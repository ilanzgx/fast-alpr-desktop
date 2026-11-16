"""Script utilitário para download dos modelos pré-treinados ONNX (YOLO e FastPlateOCR)."""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

# URL do modelo YOLOv8 treinado em detecção de placas (keremberke / ml-debi)
URL_MODELO_YOLO = (
    "https://huggingface.co/ml-debi/yolov8-license-plate-detection/resolve/main/best.onnx"
)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DESTINO_YOLO = MODELS_DIR / "placas.onnx"
DESTINO_OCR_DIR = MODELS_DIR / "ocr"


def progresso(blocos: int, tamanho_bloco: int, total: int) -> None:
    baixado = blocos * tamanho_bloco
    if total > 0:
        pct = (baixado / total) * 100
        mb_baixado = baixado / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        sys.stdout.write(f"\rBaixando modelo YOLO: {pct:.1f}% ({mb_baixado:.1f}MB / {mb_total:.1f}MB)")
        sys.stdout.flush()


def baixar_modelo_yolo() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if DESTINO_YOLO.exists():
        print(f"O modelo YOLO já existe em: {DESTINO_YOLO}")
        return

    print("Iniciando download do modelo pré-treinado YOLO...")
    print(f"Origem: {URL_MODELO_YOLO}")
    print(f"Destino: {DESTINO_YOLO}")
    try:
        urllib.request.urlretrieve(URL_MODELO_YOLO, str(DESTINO_YOLO), reporthook=progresso)
        print("\nDownload do modelo YOLO concluído com sucesso!")
    except Exception as e:
        print(f"\nFalha ao baixar o modelo YOLO: {e}")
        if DESTINO_YOLO.exists():
            DESTINO_YOLO.unlink()
        raise


def baixar_modelo_ocr() -> None:
    DESTINO_OCR_DIR.mkdir(parents=True, exist_ok=True)
    dest_onnx = DESTINO_OCR_DIR / "cct_s_v2_global.onnx"
    dest_yaml = DESTINO_OCR_DIR / "cct_s_v2_global_plate_config.yaml"

    if dest_onnx.exists() and dest_yaml.exists():
        print(f"O modelo FastPlateOCR já existe em: {DESTINO_OCR_DIR}")
        return

    print("Baixando modelo FastPlateOCR (CCT-S v2)...")
    try:
        from fast_plate_ocr.inference.hub import download_model

        caminho_onnx, caminho_yaml = download_model("cct-s-v2-global-model")
        shutil.copy2(caminho_onnx, dest_onnx)
        shutil.copy2(caminho_yaml, dest_yaml)
        print("Modelo FastPlateOCR copiado para models/ocr com sucesso!")
    except Exception as e:
        print(f"Falha ao obter modelo FastPlateOCR: {e}")
        raise


def baixar_todos() -> None:
    baixar_modelo_yolo()
    baixar_modelo_ocr()


if __name__ == "__main__":
    baixar_todos()
