# 02. Arquitetura

## 1. Visão em camadas

```
┌────────────────────────────────────────────┐
│  UI (PySide6)                              │  janela, widgets, threads, conversão para QPixmap
│  src/ui/                                   │
├────────────────────────────────────────────┤
│  Pipeline                                  │  orquestra as etapas e devolve um ResultadoPipeline
│  src/pipeline.py                           │
├────────────────────────────────────────────┤
│  Módulos de processamento (sem Qt)         │
│  detection/ · preprocessing · segmentation │
│  ocr · postprocess · utils                 │
├────────────────────────────────────────────┤
│  Infraestrutura                            │  OpenCV · NumPy · onnxruntime · Tesseract
└────────────────────────────────────────────┘
```

**Regra de dependência:** a seta só aponta para baixo. A UI conhece o pipeline; o pipeline conhece os módulos; **os módulos não importam Qt** e não conhecem a UI. Isso permite rodar o pipeline em notebooks e testes sem abrir janela.

## 2. Módulos

| Módulo | Responsabilidade | Entrada → saída |
|--------|------------------|-----------------|
| `detection/contours.py` | Detecção clássica (Canny + contornos) | imagem BGR → candidatos a placa (quadrilátero/caixa) |
| `detection/yolo_onnx.py` | Inferência ONNX, letterbox, decodificação e NMS | imagem BGR → caixas com confiança |
| `preprocessing.py` | Cinza, suavização, CLAHE, limiarização, perspectiva | ROI BGR → ROI tratado |
| `segmentation.py` | Segmentação de caracteres por componentes conexos | ROI binarizado → lista de caracteres recortados |
| `ocr.py` | Tesseract (e EasyOCR opcional) | imagem do ROI → texto bruto e confiança |
| `postprocess.py` | Validação de formato e correção por posição | texto bruto → texto final e tipo de placa |
| `pipeline.py` | Encadeia tudo e mede tempos | imagem + parâmetros → `ResultadoPipeline` |
| `config.py` | Parâmetros padrão e caminhos | (constantes e `dataclass`) |
| `utils.py` | Leitura de imagem, desenho de caixas, conversões | (funções auxiliares) |
| `ui/` | Interface, workers e widgets de imagem | usuário ↔ pipeline |

## 3. Fluxo de dados

```
Arquivo → ler_imagem → [Detecção: clássica | YOLO] → caixa
                                                        │
                                                        ▼
                                              recortar ROI (+ margem)
                                                        │
                                                        ▼
                         cinza → suavização → CLAHE → limiarização
                                                        │
                                  ┌─────────────────────┴───────────┐
                                  ▼                                 ▼
                       segmentação de caracteres                   OCR
                                  │                                 │
                                  └────────────────┬────────────────┘
                                                   ▼
                                        pós-processamento
                                                   ▼
                                           ResultadoPipeline
```

## 4. Contratos de dados

Os módulos trocam estruturas simples e imutáveis. Exemplo de referência:

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class Caixa:
    x1: int
    y1: int
    x2: int
    y2: int
    confianca: float = 1.0          # 1.0 no método clássico

@dataclass
class ParametrosPipeline:
    metodo: str = "yolo"            # "yolo" | "classico"
    confianca_min: float = 0.25
    iou_nms: float = 0.45
    canny_min: int = 30
    canny_max: int = 200
    kernel_gauss: int = 5
    clahe_clip: float = 2.0
    clahe_grade: int = 8
    binarizacao: str = "otsu"       # "otsu" | "adaptativa"
    bloco_adaptativo: int = 31
    constante_adaptativa: int = 10
    ocr_engine: str = "tesseract"
    ocr_psm: int = 7
    margem_roi: float = 0.04

@dataclass
class ResultadoPipeline:
    original: np.ndarray
    caixa: Caixa | None
    anotada: np.ndarray | None      # original com a caixa desenhada
    bordas: np.ndarray | None       # só no método clássico
    roi: np.ndarray | None
    roi_cinza: np.ndarray | None
    roi_clahe: np.ndarray | None
    binarizada: np.ndarray | None
    caracteres: list[np.ndarray] = field(default_factory=list)
    texto_bruto: str = ""
    texto_final: str = ""
    formato: str = "invalido"       # "antigo" | "mercosul" | "invalido"
    confianca_ocr: float | None = None
    tempos_ms: dict[str, float] = field(default_factory=dict)
    erro: str | None = None         # mensagem amigável quando algo falha
```

**Convenções**
- Imagens são `np.ndarray` em **BGR** (padrão do OpenCV). A conversão para RGB acontece só na UI.
- Etapas que não rodaram ficam como `None`, e a UI mostra um painel vazio.
- Falhas esperadas (sem placa, OCR vazio) não são exceções: preenchem `erro` e o pipeline devolve o que conseguiu.

## 5. Concorrência

- O pipeline roda em um **worker** (`QThread` ou `QRunnable`), nunca na thread da interface.
- O worker recebe uma **cópia** da imagem e dos parâmetros e emite um sinal com o `ResultadoPipeline`.
- Alterações rápidas de slider passam por um **debounce** de cerca de 250 ms. Se um novo pedido chegar com um processamento em curso, o resultado antigo é descartado (vale o último).
- O modelo ONNX é carregado **uma vez** (na primeira execução ou no início) e reutilizado.

## 6. Configuração

- Valores padrão em `config.py` (`ParametrosPipeline`).
- Caminhos configuráveis por variável de ambiente: `MODELO_ONNX` (padrão `models/placas.onnx`) e `TESSERACT_CMD` (necessário no Windows se o Tesseract não estiver no PATH).
- Nenhuma chave, senha ou dado pessoal em código ou repositório.

## 7. Decisões de projeto

| Decisão | Alternativa considerada | Motivo |
|---------|-------------------------|--------|
| PySide6 (desktop) | Gradio no Colab; Tauri + ort | Interface offline sem navegador e mantém Python obrigatório |
| `onnxruntime` local | `ultralytics`/PyTorch no desktop | Instalação leve, sem PyTorch no projeto local |
| Treino só no Colab | Treino local | GPU gratuita e ambiente local simples |
| `opencv-python-headless` | `opencv-python` | Evita conflito de plugins Qt com o PySide6 |
| Letterbox e NMS próprios | Usar tudo pronto | São operações do grupo e reduzem dependência de bibliotecas |
| Dois métodos de detecção | Só YOLO | Permite comparação e análise crítica |
| `uv` com `uv.lock` | `pip` + `requirements.txt` | Ambiente reproduzível |
