# Fast ALPR Desktop — Reconhecimento de Placas Veiculares

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://docs.astral.sh/uv/)
[![Qt 6](https://img.shields.io/badge/GUI-PySide6-41CD52.svg)](https://wiki.qt.io/Qt_for_Python)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-005CED.svg)](https://onnxruntime.ai/)
[![CI](https://github.com/ilanzgx/fast-alpr-desktop/actions/workflows/ci.yml/badge.svg)](https://github.com/ilanzgx/fast-alpr-desktop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Aplicação desktop profissional e didática para **Reconhecimento Automático de Placas Veiculares (ALPR — *Automatic License Plate Recognition*)**, desenvolvida em Python 3.12 com interface gráfica moderna em **PySide6 (Qt 6)** e inferência de redes neurais profundas via **ONNX Runtime (CPU)**.

A ferramenta é **100% offline** (não requer conexão com a internet ou APIs externas) e apresenta visualmente cada etapa intermediária do processamento de visão computacional, tornando o aprendizado do pipeline de ALPR transparente e interativo.

---

## Principais Funcionalidades

- **Detecção Dupla de Placas (E1):**
  - *Deep Learning:* Detector neural **YOLOv8** exportado em formato ONNX de alta performance com Letterbox e NMS (*Non-Maximum Suppression*).
  - *Método Clássico:* Visão computacional geométrica tradicional via detector de bordas Canny, dilatação morfológica e filtragem de contornos retangulares.
- **Tratamento Angular Avançado (E3):**
  - Retificação angular baseada na Transformada Probabilística de Hough com **ponderação quadrática por comprimento ($comprimento^2$)** e limiares estruturais, eliminando rotações falsas causadas por traços inclinados de caracteres (`A`, `K`, `M`, `R`, etc.).
  - Expansão dinâmica de canvas com zoom-out para evitar cortes de caracteres nas extremidades da placa.
- **Reconhecimento Neural com FastPlateOCR (E9):**
  - Motor baseado em **Compact Convolutional Transformers (CCT-S v2)** executado diretamente em ONNX.
  - Robustez extrema contra ruídos de parafuso, molduras, reflexos na lataria e variações de iluminação.
- **Pós-processamento e Correção Posicional (E10):**
  - Validação estrita dos formatos brasileiros: **Mercosul (`AAA0A00`)** e **Padrão Antigo/Cinza (`AAA-0000`)**.
  - Matriz de confusão contextual posicional para corrigir ambiguidades entre letras e números (ex: `O` e `0`, `S` e `5`, `G` e `6`, `B` e `8`).
  - Suporte ao **Modo Livre** para placas internacionais ou formatos não-padronizados.
- **Interface Gráfica Reativa (PySide6):**
  - Execução assíncrona em thread separada (`QThread`), garantindo interface 100% fluida sem travamentos.
  - Abertura maximizada automática no monitor, tema escuro profissional (Fusion/QSS) e controles responsivos.
- **Distribuição Multiplataforma Pronta para Uso:**
  - Pipeline de Release automatizada que compila o aplicativo em executáveis autônomos para Windows (`ALPR-Placas-windows-x64-*.exe`) e Linux (`ALPR-Placas-linux-x64-*.AppImage`), sem necessidade de instalação do Python pelo usuário final.

---

## Resultados do Benchmark Quantitativo

Avaliação formal realizada sobre o dataset de teste (`data/teste`) com **120 fotografias veiculares reais**:

| Etapa Avaliada | Métrica | Desempenho |
|---|---|:---:|
| **FastPlateOCR Isolado** (`recortes/`) | **Taxa de Acerto Exato** | **95,8%** (115/120) |
| | Acurácia por Caractere | **99,3%** |
| | CER (*Character Error Rate*) | **0,7%** |
| | Formato Válido Brasileiro | **100,0%** |
| | Tempo Médio de OCR | ~102 ms |
| **Detecção YOLO ONNX** (`imagens/`) | Taxa de Detecção ($\text{IoU} \ge 0.50$) | **95,0%** (114/120) |
| | $\text{IoU}$ Médio de Detecção | **0,692** |
| | Tempo Médio de Inferência | ~230 ms |
| **Pipeline Ponta a Ponta** | **Acerto Exato (Detecção + OCR)** | **~91,0%** (110/120) |
| | Aproveitamento pós-detecção | **94,0%** (109/116) |

---

## Como Executar

### Opção 1: Executáveis Autônomos (Windows e Linux)

Basta baixar o binário correspondente ao seu sistema operacional na aba de **[Releases do GitHub](../../releases)**:

- **Windows (x64):** `ALPR-Placas-windows-x64-v1.0.0.exe` ou `.zip`
- **Linux (x86_64):** `ALPR-Placas-linux-x64-v1.0.0.AppImage` ou `.tar.gz`

Não é necessário instalar Python, bibliotecas ou compiladores.

---

### Opção 2: Execução a partir do Código-Fonte

#### Pré-requisitos
- Python `3.12`
- Gerenciador de pacotes [uv](https://docs.astral.sh/uv/)

#### Passos

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/ilanzgx/fast-alpr-desktop.git
   cd fast-alpr-desktop
   ```

2. **Instalar dependências e sincronizar ambiente:**
   ```bash
   uv sync
   ```

3. **Baixar os modelos neurais de referência:**
   ```bash
   uv run python scripts/baixar_modelo_referencia.py
   ```

4. **Iniciar o aplicativo:**
   ```bash
   uv run python -m src.app
   ```

---

## Testes, Qualidade e Benchmark

```bash
# Executar a suíte de testes unitários (pytest)
uv run pytest

# Executar o linter de código estático (Ruff)
uv run ruff check src tests scripts

# Executar a avaliação completa de benchmark
uv run python scripts/avaliar.py
```

---

## Estrutura do Projeto

```
algoritmoeprocessamentodeimagem/
├── .github/workflows/         # Pipelines CI/CD (CI e Release)
├── models/                    # Modelos neurais ONNX (YOLO e FastPlateOCR)
├── scripts/
│   ├── baixar_modelo_referencia.py  # Download e cache dos modelos ONNX
│   └── avaliar.py             # Script de benchmark quantitativo
├── src/
│   ├── app.py                 # Ponto de entrada da aplicação desktop
│   ├── config.py              # Dataclasses e parâmetros do pipeline
│   ├── pipeline.py            # Orquestrador das etapas E1 a E10
│   ├── preprocessing.py       # Visão computacional (Hough, CLAHE, rotação)
│   ├── segmentation.py        # Segmentação por componentes conexos
│   ├── ocr.py                 # Integração com FastPlateOCR (ONNX)
│   ├── postprocess.py         # Regex Mercosul/Antigo e correção posicional
│   ├── detection/             # Detecção por YOLO ONNX e Contornos Canny
│   └── ui/                    # Interface gráfica PySide6 e QSS
└── tests/                     # Testes automatizados com Pytest
```

---

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE` para mais detalhes.
