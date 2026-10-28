# 08. Setup e repositório

## 1. Pré-requisitos

| Item | Versão / observação |
|------|---------------------|
| Python | 3.11 ou 3.12 (versões muito novas podem demorar a ter *wheels* de `onnxruntime`) |
| uv | <https://docs.astral.sh/uv/> |
| Tesseract OCR | Programa instalado no sistema (não vem pelo `uv`) |
| Git | Para o fluxo em equipe |

**Instalação do Tesseract**

| Sistema | Comando / origem |
|---------|------------------|
| Windows | Instalador da UB Mannheim; adicionar ao PATH ou definir `TESSERACT_CMD` |
| Linux (Debian/Ubuntu) | `sudo apt install tesseract-ocr` |
| macOS | `brew install tesseract` |

Conferir com `tesseract --version`.

## 2. Instalação do projeto

```bash
git clone <url-do-repositorio>
cd alpr-placas
uv sync
```

Colocar o modelo em `models/placas.onnx` (link no `models/README.md`) e executar:

```bash
uv run python -m src.app
```

> Se o `app.py` estiver na raiz de `src/` e os imports forem relativos ao projeto, executar sempre **a partir da raiz do repositório**.

## 3. Dependências

```toml
[project]
name = "alpr-placas"
requires-python = ">=3.11,<3.13"
dependencies = [
    "PySide6",
    "opencv-python-headless",
    "numpy",
    "onnxruntime",
    "pytesseract",
]

[project.optional-dependencies]
easyocr = ["easyocr"]          # puxa PyTorch; só para comparação de OCR

[dependency-groups]
dev = ["pytest", "ruff"]
```

Comandos úteis:

```bash
uv add PySide6 opencv-python-headless numpy onnxruntime pytesseract
uv add --dev pytest ruff
uv add --optional easyocr easyocr
uv sync --extra easyocr          # só quem for rodar a comparação de OCR
```

**Migração a partir do ambiente atual** (se já instalou `opencv-python`, `gradio` e `ultralytics`):

```bash
uv remove gradio ultralytics opencv-python
uv add PySide6 opencv-python-headless onnxruntime
```

O `ultralytics` e o PyTorch ficam **só no Colab**.

## 4. Estrutura de pastas

```
alpr-placas/
├── pyproject.toml
├── uv.lock                      # versionar
├── .python-version              # versionar
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── app.py                   # ponto de entrada (QApplication)
│   ├── config.py                # ParametrosPipeline e caminhos
│   ├── pipeline.py              # orquestração
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── contours.py
│   │   └── yolo_onnx.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── ocr.py
│   ├── postprocess.py
│   ├── utils.py
│   └── ui/
│       ├── janela_principal.py
│       ├── painel_imagem.py
│       ├── painel_controles.py
│       ├── worker.py
│       └── conversoes.py
├── notebooks/
│   ├── 01_baseline_classico.ipynb
│   ├── 02_treino_yolo.ipynb     # roda no Colab
│   └── 03_avaliacao.ipynb
├── scripts/
│   └── avaliar.py               # avaliação em lote
├── models/
│   └── README.md                # origem, hash, link; pesos fora do git
├── data/
│   ├── README.md                # origem, licença, condições de captura
│   ├── ajuste/
│   ├── teste/                   # reservado
│   └── demo/                    # amostras autorizadas e anonimizadas
├── tests/
│   ├── test_postprocess.py
│   ├── test_nms.py
│   └── test_preprocessing.py
├── results/
│   ├── metricas.csv
│   └── figuras/
└── docs/                        # esta documentação
```

## 5. `.gitignore`

```
__pycache__/
.venv/
.ipynb_checkpoints/
runs/
*.pt
*.onnx
data/teste/
.env
*.log
```

Pesos (`*.pt`, `*.onnx`) e o conjunto de teste ficam fora do git. Documente em `models/README.md` onde baixá-los.

## 6. Variáveis de ambiente

| Variável | Uso | Padrão |
|----------|-----|--------|
| `MODELO_ONNX` | Caminho do modelo ONNX | `models/placas.onnx` |
| `TESSERACT_CMD` | Caminho do executável do Tesseract | (PATH do sistema) |

## 7. Testes

```bash
uv run pytest                    # todos os testes
uv run ruff check src tests      # análise estática
```

| Arquivo | Cobre |
|---------|-------|
| `test_postprocess.py` | Limpeza, formatos, correção por posição, empates |
| `test_nms.py` | NMS próprio igual ao `cv2.dnn.NMSBoxes` |
| `test_preprocessing.py` | Formato e tipo das saídas (CLAHE, limiarização) e polaridade |

## 8. Fluxo de trabalho com Git

- `main` estável; uma branch por funcionalidade (`feat/deteccao-yolo`, `feat/interface`, `feat/ocr`).
- Pull requests curtos, revisados por outra pessoa do grupo.
- Commits pequenos e com mensagens claras (`feat:`, `fix:`, `docs:`, `test:`).
- Nunca commitar chaves de API, dados pessoais, pesos grandes ou o conjunto de teste.

## 9. Problemas comuns

| Sintoma | Causa provável | Solução |
|---------|----------------|---------|
| `Could not load the Qt platform plugin "xcb"` | Conflito entre `opencv-python` e PySide6 | Usar `opencv-python-headless`; no Linux, instalar `libxcb-cursor0` |
| `TesseractNotFoundError` | Tesseract fora do PATH | Instalar e/ou definir `TESSERACT_CMD` |
| Saída do ONNX com forma inesperada | Versão do YOLO diferente | Conferir `sess.get_outputs()[0].shape` e o layout (ver `05-modelo-onnx.md`) |
| Caixas deslocadas | Letterbox não revertido | Subtrair o *padding* e dividir pela escala |
| Janela trava ao processar | Pipeline na thread da UI | Usar o worker (`QThread`/`QRunnable`) |
| `imread` retorna `None` com acento no caminho | Limitação do `cv2.imread` no Windows | Usar `np.fromfile` + `cv2.imdecode` |
| `uv sync` baixa PyTorch | Extra `easyocr` ativado sem querer | Rodar `uv sync` sem `--extra` |

## 10. Esqueleto do `README.md` do repositório

1. Descrição em duas linhas e *prints* da interface.
2. Pré-requisitos (Python, uv, Tesseract).
3. Instalação (`uv sync`) e obtenção do modelo.
4. Execução (`uv run python -m src.app`).
5. Estrutura de pastas e link para `docs/`.
6. Origem dos dados e aviso sobre privacidade.
7. Autoria e contribuição de cada integrante.
