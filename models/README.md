# Modelos de Detecção de Placas

Este diretório armazena os modelos treinados em formato ONNX para inferência offline.

## Arquivos Esperados

- `placas.onnx`: Modelo YOLOv8 exportado para ONNX utilizado na detecção de placas veiculares pelo app desktop.

## Obtenção do Modelo

1. **Modelo de Referência (Temporário para testes):**
   Execute o script utilitário:
   ```bash
   uv run python scripts/baixar_modelo_referencia.py
   ```
   Ele fará o download do modelo `best.onnx` pré-treinado do Hugging Face (`ml-debi/yolov8-license-plate-detection`).

2. **Modelo Próprio (Treinado no Google Colab):**
   Execute o notebook [`notebooks/02_treino_yolo.ipynb`](../notebooks/02_treino_yolo.ipynb) no Colab com GPU, baixe o arquivo `best.onnx` gerado e renomeie-o para `placas.onnx` dentro desta pasta.

*Nota: Arquivos `.onnx` e `.pt` são ignorados pelo Git.*
