# AGENTS.md — Contexto do Sistema ALPR Didático

> Guia de referência operacional, arquitetural e técnico para agentes autônomos e desenvolvedores que atuam no repositório **alpr-placas**.

---

## 1. Visão Geral do Projeto

O **alpr-placas** é uma aplicação desktop offline e didática para **Reconhecimento Automático de Placas Veiculares (ALPR — *Automatic License Plate Recognition*)**, desenvolvida em Python 3.12 com interface gráfica em PySide6 (Qt).

### Objetivos Centrais
- **Localização de Placas:** Suporte a dois métodos intercambiáveis:
  1. *Deep Learning:* Rede neural YOLOv8 exportada em formato **ONNX** executada com `onnxruntime` (CPU).
  2. *Processamento Clássico:* Visão computacional tradicional via detector de bordas Canny e filtragem geométrica de contornos quadriláteros.
- **Processamento de Imagem Transparente e Didático:** Exibição visual de cada etapa intermediária (bordas, recorte do ROI, equalização CLAHE, binarização por limiarização e segmentação de caracteres).
- **Tratamento Robusto de Inclinações e Fotos Diagonais:** Retificação angular baseada na Transformada Probabilística de Hough e expansão de tela sem corte de caracteres.
- **OCR e Pós-Processamento Contextual:** Leitura por Tesseract OCR com segmentação individual e aplicação de matriz de confusão posicional para padrões brasileiros (Mercosul `AAA0A00` e Cinza/Antigo `AAA-0000`).
- **Operação 100% Offline:** Nenhuma dependência de APIs em nuvem para inferência ou OCR.

---

## 2. Stack Tecnológica e Ferramentas

| Componente | Tecnologia | Versão / Detalhes |
|---|---|---|
| **Linguagem** | Python | `3.12` gerenciado pelo [uv](https://docs.astral.sh/uv/) |
| **Interface Gráfica** | PySide6 | `>= 6.6.0` (Qt 6 com QSS customizado) |
| **Processamento de Imagem** | OpenCV (`opencv-python-headless`) | `>= 4.8.0` |
| **Computação Vetorial** | NumPy | `>= 1.26.0, < 2.0.0` |
| **Inferência de Rede Neural** | ONNX Runtime | `>= 1.16.0` (execução CPU / DirectML) |
| **Motor de OCR** | Tesseract OCR (`pytesseract`) | `>= 0.3.10` (binário instalado no sistema ou em `tools/tesseract/`) |
| **Testes e Qualidade** | Pytest e Ruff | `pytest >= 8.0.0`, `ruff >= 0.3.0` |
| **Treinamento** | Ultralytics YOLOv8 / PyTorch | Executado no Google Colab (notebooks em `notebooks/`) |

---

## 3. Estrutura de Diretórios

```
algoritmoeprocessamentodeimagem/
├── AGENTS.md                  # Este documento (contexto mestre para agentes)
├── README.md                  # Apresentação do projeto e instruções de uso
├── pyproject.toml             # Metadados do projeto, dependências uv, ruff e pytest
├── uv.lock                    # Lockfile de dependências exatas
├── .python-version            # 3.12
│
├── docs/                      # Documentação técnica em Markdown
│   ├── 01-visao-geral.md      # Visão geral e contextualização didática
│   ├── 02-arquitetura.md      # Regras de arquitetura e isolamento de camadas
│   ├── 03-pipeline.md         # Detalhamento das etapas E1 a E10
│   ├── 04-interface.md        # Especificação da UI em PySide6
│   ├── 05-modelo-onnx.md      # Exportação, entrada e decodificação do YOLO
│   ├── 06-ocr-pos-processamento.md # Regras de validação e matrizes de substituição
│   ├── 07-dados-avaliacao.md  # Métricas e datasets (HuggingFace e IFRN)
│   ├── 08-setup-repositorio.md# Configuração do ambiente local
│   ├── 09-roadmap.md          # Próximos passos e backlog de melhorias
│   └── 10-riscos-etica.md     # Considerações éticas e LGPD
│
├── models/                    # Modelos de Machine Learning (ignorados no git se volumosos)
│   ├── placas.onnx            # Modelo de referência padrão carregado pela aplicação
│   └── README.md
│
├── notebooks/                 # Notebooks Jupyter para treino e validação no Google Colab
│   ├── 02_treino_yolo.ipynb           # Treino base YOLOv8n com Parquet HuggingFace
│   ├── 02b_treino_yolo_brasil.ipynb   # Fine-tuning incremental com dataset brasileiro
│   └── 03_benchmark_comparativo.ipynb # Avaliação formal A/B, teste de estresse angular e gráficos
│
├── scripts/                   # Utilitários de automação e setup inicial
│   ├── baixar_modelo_referencia.py    # Download do modelo ONNX de placas pré-treinado
│   └── setup_tesseract_local.py       # Auxiliar para instalação/verificação do Tesseract
│
├── src/                       # Código-fonte principal da aplicação
│   ├── __init__.py
│   ├── app.py                 # Ponto de entrada (`python -m src.app`)
│   ├── config.py              # Dataclasses de configuração (`ParametrosPipeline`, `ResultadoPipeline`, `Caixa`)
│   ├── pipeline.py            # Orquestrador central que encadeia todas as etapas
│   ├── preprocessing.py       # Funções de visão computacional pura (cinza, blur, CLAHE, Hough deskew)
│   ├── segmentation.py        # Segmentação de caracteres por componentes conexos
│   ├── ocr.py                 # Integração com Tesseract, recorte de miolo e faixas limpas
│   ├── postprocess.py         # Regex de placas brasileiras e correção contextual
│   ├── utils.py               # Funções utilitárias (leitura de imagem, desenho de caixas)
│   │
│   ├── detection/             # Módulos de detecção de placas (E1)
│   │   ├── __init__.py
│   │   ├── contours.py        # Detecção geométrica clássica (Canny + contornos)
│   │   └── yolo_onnx.py       # Inferência ONNX, letterbox, decodificação e NMS
│   │
│   └── ui/                    # Camada de Apresentação (PySide6)
│       ├── __init__.py
│       ├── janela_principal.py# Janela principal, layouts e coordenação de eventos
│       ├── painel_controles.py# Painel lateral com sliders, comboboxes e botões
│       ├── painel_imagem.py   # Viewports para exibição das imagens das etapas
│       ├── worker.py          # Execução assíncrona em thread separada (`QThread`)
│       ├── estilo.py          # Folha de estilos QSS (tema escuro profissional)
│       └── conversoes.py      # Conversão otimizada NumPy/OpenCV BGR/Gray -> QPixmap
│
└── tests/                     # Testes automatizados (Pytest)
    ├── test_nms.py            # Testes unitários do algoritmo NMS
    ├── test_preprocessing.py  # Testes de binarização, inversão, CLAHE e recortes
    └── test_postprocess.py    # Testes das regras de placas Mercosul, antigas e matriz de confusão
```

---

## 4. Arquitetura e Regras de Ouro para Agentes

```
┌────────────────────────────────────────────┐
│  UI (PySide6)                              │  janela, widgets, worker QThread, QPixmap
│  src/ui/                                   │
├────────────────────────────────────────────┤
│  Pipeline Orquestrador                     │  executar_pipeline(...) -> ResultadoPipeline
│  src/pipeline.py                           │
├────────────────────────────────────────────┤
│  Módulos de Processamento (Sem Qt)         │  detection/ · preprocessing · segmentation
│                                            │  ocr · postprocess · utils · config
├────────────────────────────────────────────┤
│  Infraestrutura                            │  OpenCV · NumPy · ONNX Runtime · Tesseract
└────────────────────────────────────────────┘
```

### Regras Mandatórias de Código:
1. **Regra de Dependência Unidirecional (Zero Qt no Core):**
   - Os módulos `src/detection/`, `src/preprocessing.py`, `src/segmentation.py`, `src/ocr.py`, `src/postprocess.py`, `src/config.py` e `src/pipeline.py` **NUNCA devem importar PySide6 ou Qt**.
   - Toda lógica nesses arquivos deve receber e retornar apenas tipos nativos de Python, `numpy.ndarray` ou dataclasses de `src/config.py`. Isso garante que o pipeline possa ser testado via CLI ou notebooks sem inicializar o loop de eventos gráfico.
2. **Execução Não-Bloqueante na UI:**
   - O processamento de imagens deve ser invocado exclusivamente através do `WorkerPipeline` (`QThread` em `src/ui/worker.py`), emitindo o sinal `processamento_concluido(ResultadoPipeline)`. Nunca invoque `executar_pipeline` diretamente na thread principal da interface.
3. **Imutabilidade e Tipagem Estrita:**
   - O contrato de dados usa dataclasses imutáveis (`Caixa(frozen=True)`, `ResultadoPosProcessamento(frozen=True)`). Todas as funções devem ter *type hints* rigorosos e docstrings explicativas em português.
4. **Respeito ao Linteamento e Formatação:**
   - Linhas com limite de até 100 caracteres.
   - Sempre valide alterações com `uv run pytest` e `uv run ruff check src tests`.

---

## 5. Fluxo de Execução do Pipeline (Etapas E1 a E10)

```mermaid
flowchart TD
    E0[Imagem Original BGR] --> E1[E1: Detecção da Placa\nYOLO ONNX ou Clássico Canny]
    E1 -->|Caixa Bounding Box| E2[E2: Recorte do ROI\n+ Margem Configurável]
    E2 --> E3[E3: Retificação Geométrica\nTransformada de Hough + Expansão Canvas]
    E3 --> E4[E4: Conversão para Tons de Cinza\n+ Redimensionamento Padronizado]
    E4 --> E5[E5: Suavização Gaussiana\nFiltro 2D anti-ruído]
    E5 --> E6[E6: Equalização de Contraste\nCLAHE Adaptativo]
    E6 --> E7[E7: Binarização e Inversão\nOtsu ou Adaptativa]
    E7 --> E8[E8: Segmentação de Caracteres\nComponentes Conexos 8-conectados]
    E7 --> E9[E9: OCR Tesseract\nLeitura Direta / Faixa Segmentada / Miolo]
    E8 --> E9
    E9 --> E10[E10: Pós-Processamento e Validação\nRegex Mercosul/Antigo + Matriz de Confusão]
    E10 --> Out[ResultadoPipeline Estruturado]
```

### Detalhes das Etapas Críticas:

- **E1 (Detecção):**
  - *YOLO:* `DetectorYolo` faz letterbox mantendo proporção (padrão $640 \times 640$), normalização para $[0, 1]$, inferência ONNX e decodificação com Non-Maximum Suppression (NMS).
  - *Clássico:* Gradiente morfológico / Canny, dilatação retangular e seleção de contornos quadriláteros com aspecto entre $2.0$ e $5.5$.
- **E3 (Correção Angular e Retificação):**
  - Implementado em `corrigir_inclinacao`. Usa `cv2.HoughLinesP` ponderado pelo comprimento das bordas após CLAHE, cobrindo de $-45^\circ$ a $+45^\circ$.
  - Recalcula a largura e altura do canvas ($W_{novo} = |H \sin \theta| + |W \cos \theta|$) para **não podar as extremidades da placa**.
  - Recorta o miolo claro para eliminar o excesso de para-choque/grade em rotações $\ge 2^\circ$.
- **E8 (Segmentação de Caracteres):**
  - `cv2.connectedComponentsWithStats` com conectividade 8. Filtra por faixa de altura proporcional (`altura_min_pct` a `altura_max_pct`), largura máxima e área mínima, ordenando da esquerda para a direita.
- **E9 (OCR):**
  - Avalia com prioridade absoluta imagens com padrão alfanumérico limpo (inclusive faixas concatenadas de caracteres isolados e miolo sem parafusos/molduras).
- **E10 (Pós-processamento Posicional):**
  - Valida com regex se o texto limpo atende a:
    - **Mercosul:** `^[A-Z]{3}[0-9][A-Z][0-9]{2}$`
    - **Antigo:** `^[A-Z]{3}[0-9]{4}$`
  - Se houver pequenas confusões típicas de OCR (ex: `O` por `0`, `S` por `5`, `G` por `6`, `B` por `8`, `E` por `8`, `A` por `4`), o algoritmo aplica substituições contextuais baseadas na posição do caractere (letras nas posições 0-2 e dígitos nas posições 3, 5 e 6).
  - Suporta busca por janela deslizante de 7 caracteres para extrair a placa de textos contendo ruídos de borda ou moldura.

---

## 6. Datasets e Treinamento de Modelos

A pasta `notebooks/` contém os fluxos de trabalho documentados para execução no Google Colab:

1. **`02_treino_yolo.ipynb`:**
   - Download do dataset público via Parquet do Hugging Face (`keremberke/license-plate-object-detection`).
   - Opção de **duplicação offline** com 4 variações fotométricas realistas (noite/gama, motion blur, alto contraste e ruído ISO).
   - Treinamento do YOLOv8n com *Data Augmentation* dinâmico na GPU (`degrees=25.0`, `shear=10.0`, `perspective=0.001`, `hsv_v=0.4`, `mosaic=1.0`).
   - Exportação direta para ONNX (`placas.onnx`).
2. **`02b_treino_yolo_brasil.ipynb`:**
   - Fine-tuning incremental especializado em placas brasileiras (dataset do IFRN).
   - Aplica taxa de aprendizado suave (`lr0=0.001`) sobre os pesos do treino anterior.
3. **`03_benchmark_comparativo.ipynb`:**
   - Compara cientificamente dois modelos (`modelo_a.pt` vs `modelo_b.pt` ou `.onnx`).
   - Calcula métricas formais COCO: Precisão, Revocação, mAP50, mAP50-95 e tempo de inferência.
   - **Teste de Estresse Angular:** Curva de confiança média avaliada em $-25^\circ, -15^\circ, 0^\circ, +15^\circ, +25^\circ$.
   - Visualização comparativa lado a lado (*Side-by-Side*).

---

## 7. Guia de Comandos para Agentes

### Instalação e Execução
```bash
# Sincronizar ambiente virtual uv com dependências travadas
uv sync

# Baixar modelo de referência padrão ONNX (se necessário)
uv run python scripts/baixar_modelo_referencia.py

# Iniciar aplicação desktop PySide6
uv run python -m src.app
```

### Qualidade de Código e Testes
```bash
# Executar suíte de testes automatizados
uv run pytest

# Executar linter e verificador de código (Ruff)
uv run ruff check src tests

# Aplicar correções automáticas seguras do Ruff
uv run ruff check --fix src tests
```

### Boas Práticas ao Executar Comandos no Windows/PowerShell
- No ambiente Windows, comandos com aspas duplas aninhadas ou strings com quebras de linha inline no PowerShell podem falhar com erros de parsing (`ParameterBindingException` ou `unterminated string literal`).
- **Recomendação para agentes:** Sempre que precisar rodar scripts de teste ou inspeção ad-hoc, crie um arquivo temporário em `scratch/` ou use ferramentas de manipulação de arquivo dedicadas (`write_file`, `edit_file`) e invoque `uv run python caminho/do_script.py`. Limpe os arquivos temporários ao final.

---

## 8. Convenções de Código e Estilo

- **Linguagem dos comentários e documentação:** Português do Brasil.
- **Tipagem:** Uso obrigatório de `from __future__ import annotations` e anotações de tipo em todas as assinaturas de funções e métodos.
- **Tratamento de exceções no Pipeline:** O pipeline nunca deve lançar uma exceção não capturada que derrube a UI; em vez disso, preencha o campo `erro: str | None` de `ResultadoPipeline`.
- **Caminhos de Arquivos:** Sempre utilizar `pathlib.Path` para manipulação de caminhos multiplataforma. Evite concatenação de strings para caminhos.
- **Design da UI:** Manter a consistência visual em `src/ui/estilo.py`. A interface adota um tema escuro profissional (`#0B0E14`, `#151B28`, `#1E2638`), tipografia clara e controles com feedback de estado.
