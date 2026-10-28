# Reconhecimento de Placas Veiculares (ALPR)

Aplicação desktop e offline em Python para localização de placas veiculares em fotografias, pré-processamento de imagem e reconhecimento óptico de caracteres (OCR), com interface gráfica interativa desenvolvida em PySide6.

## Principais Funcionalidades

- **Detecção Dupla:** Escolha entre método clássico (OpenCV Canny + contornos quadriláteros) e redes neurais profundas (YOLOv8 em ONNX).
- **Processamento de Imagem:** Pipeline visual com tons de cinza, filtro gaussiano, equalização CLAHE, binarização (Otsu e adaptativa) e correção de perspectiva/rotação.
- **Segmentação e OCR:** Segmentação dos caracteres por componentes conexos e leitura via Tesseract OCR em múltiplos modos.
- **Pós-processamento Inteligente:** Validação de padrões brasileiros (Mercosul `AAA0A00` e Antigo `AAA-0000`), com correção posicional por matriz de confusão e modo livre para placas internacionais.
- **Interface Reativa:** PySide6 com execução em thread separada (`QThread`), controles debounced e visualização de todas as etapas intermediárias.

## Requisitos do Sistema

- **Python:** 3.12 (gerenciado via [uv](https://docs.astral.sh/uv/))
- **Tesseract OCR:** Instalado no sistema operacional ou em `tools/tesseract/`
- **Modelo ONNX:** Armazenado em `models/placas.onnx`

## Instalação e Execução

1. Instalar as dependências com `uv`:
   ```bash
   uv sync
   ```

2. Executar a aplicação desktop:
   ```bash
   uv run python -m src.app
   ```

3. Executar os testes automatizados:
   ```bash
   uv run pytest
   ```

Consulte a documentação completa no diretório [`docs/`](docs/).
