# 06. OCR Neural e Pós-Processamento

> Especificação técnica do motor de reconhecimento de caracteres (FastPlateOCR), integração neural em ONNX e pipeline de validação/correção contextual para padrões brasileiros.

---

## 1. Evolução do Motor de OCR: Por que FastPlateOCR?

Inicialmente, o projeto contava com o **Tesseract OCR**. No entanto, durante as avaliações quantitativas e benchmarks empíricos com placas veiculares brasileiras reais, o Tesseract apresentou graves limitações estruturais:

* **Treinamento inadequado:** O Tesseract foi concebido para digitalização de documentos e livros escaneados (fundo branco, texto linear preto e fontes tipográficas planas).
* **Vulnerabilidade a elementos metálicos:** Parafusos de fixação, faixas azuis do Mercosul, brasões estaduais, reflexos na chapa e variações de relevo estrito causavam quedas severas de acurácia (taxa de acerto exato de apenas ~23% no Tesseract).
* **Dependência de binários externos:** Exigia a instalação e localização de executáveis do sistema operacional (`tesseract.exe`), dificultando a portabilidade.

### Transição para o FastPlateOCR (CCT Transformer ONNX)

O sistema migrou integralmente para o **FastPlateOCR**, baseado na arquitetura **Compact Convolutional Transformer (CCT-S v2)**:

| Característica | Tesseract OCR (Antigo) | FastPlateOCR (Atual) |
|---|---|---|
| **Paradigma** | OCR Tradicional / Heurístico | Deep Learning (Convolução + Mecanismo de Atenção) |
| **Formato de Execução** | Executável C++ externo (`pytesseract`) | **ONNX Runtime (CPU)** nativo em Python |
| **Operação Offline** | Exigia configuração de `tessdata` local | **100% Offline** (pesos ONNX embutidos) |
| **Acerto Exato (120 Amostras)** | ~23,3% | **95,8%** (115/120) |
| **Acurácia por Caractere** | 63,1% | **99,3%** |
| **CER Médio (*Error Rate*)** | 93,7% | **0,7%** |
| **Tempo Médio de Inferência** | ~290 ms | **~100 ms** |

---

## 2. Integração e Inferência do FastPlateOCR

O módulo [`src/ocr.py`](../src/ocr.py) gerencia o carregamento e execução do modelo CCT-S v2:

```python
from fast_plate_ocr import LicensePlateRecognizer
from src.config import ROOT_DIR

# Carregamento local prioritário dos pesos ONNX embutidos
local_onnx = ROOT_DIR / "models" / "ocr" / "cct_s_v2_global.onnx"
local_yaml = ROOT_DIR / "models" / "ocr" / "cct_s_v2_global_plate_config.yaml"

recognizer = LicensePlateRecognizer(
    onnx_model_path=str(local_onnx),
    plate_config_path=str(local_yaml),
    device="cpu",
)
```

### Pré-processamento e Entrada do Modelo
1. **Canalização BGR:** O FastPlateOCR opera sobre tensores tridimensionais $[H, W, 3]$. Recortes em tons de cinza ou binarizados são automaticamente convertidos para 3 canais via `cv2.cvtColor`.
2. **Invariância de Escala e Orientação:** A combinação da retificação angular prévia (Hough Lines com ponderação quadrática) com a capacidade intrínseca do transformer garante leituras precisas mesmo com ligeiras inclinações residuais.
3. **Confiança Ponderada:** O modelo extrai as probabilidades individuais de cada caractere (`char_probs`), permitindo calcular uma pontuação de confiança real e transparente para a interface gráfica.

---

## 3. Pós-Processamento e Validação Contextual

Mesmo com um modelo neural de altíssima precisão, pequenos ruídos de imagem (ex: reflexo solar intenso) podem induzir ambiguidades visuais entre caracteres morfologicamente similares. O módulo [`src/postprocess.py`](../src/postprocess.py) aplica regras contextuais fundamentadas na legislação de trânsito brasileira.

### Padrões de Placas Veiculares Brasileiras

| Formato | Padrão | Expressão Regular | Exemplo |
|---|---|---|---|
| **Mercosul** | `AAA0A00` | `^[A-Z]{3}[0-9][A-Z][0-9]{2}$` | `BRA2E19` |
| **Antigo (Cinza)** | `AAA-0000` | `^[A-Z]{3}[0-9]{4}$` | `ABC1234` |

---

## 4. Correção por Posição (Matriz de Confusão)

A estrutura fixa das placas brasileiras define rigidamente quais tipos de caractere são permitidos em cada posição:

```
Posição:   0   1   2   3   4   5   6
Mercosul: [L] [L] [L] [D] [L] [D] [D]   (L = Letra, D = Dígito)
Antigo:   [L] [L] [L] [D] [D] [D] [D]
```

* **Posições 0, 1 e 2:** Exclusivamente **Letras** (`A-Z`).
* **Posição 3:** Exclusivamente **Dígito** (`0-9`).
* **Posição 4:** Ponto de bifurcação — **Dígito** (Padrão Antigo) ou **Letra** (Mercosul).
* **Posições 5 e 6:** Exclusivamente **Dígitos** (`0-9`).

### Matriz de Mapeamento Bidirecional

| Letra $\rightarrow$ Dígito | Dígito $\rightarrow$ Letra |
|:---:|:---:|
| `O` $\rightarrow$ `0`, `Q` $\rightarrow$ `0`, `D` $\rightarrow$ `0` | `0` $\rightarrow$ `O` |
| `I` $\rightarrow$ `1`, `L` $\rightarrow$ `1`, `T` $\rightarrow$ `1` | `1` $\rightarrow$ `I` |
| `Z` $\rightarrow$ `2` | `2` $\rightarrow$ `Z` |
| `A` $\rightarrow$ `4` | `4` $\rightarrow$ `A` |
| `S` $\rightarrow$ `5` | `5` $\rightarrow$ `S` |
| `G` $\rightarrow$ `6` | `6` $\rightarrow$ `G` |
| `T` $\rightarrow$ `7`, `Y` $\rightarrow$ `7` | `7` $\rightarrow$ `T` |
| `B` $\rightarrow$ `8`, `E` $\rightarrow$ `8` | `8` $\rightarrow$ `B` |

---

## 5. Algoritmo de Decisão e Desempate

Quando uma cadeia de caracteres é recebida:

1. **Limpeza e Normalização:** Converte para maiúsculas, remove hífens, espaços e caracteres especiais fora de `[A-Z0-9]`.
2. **Extração por Janela Deslizante (7 Caracteres):** Se a leitura contiver caracteres espúrios decorrentes da moldura externa (ex: `11BRA2E19`), o algoritmo avalia subjanelas de comprimento 7 para identificar a placa válida.
3. **Avaliação das Duas Hipóteses:**
   - **Hipótese Mercosul:** Força letras nas posições 0, 1, 2, 4 e dígitos nas posições 3, 5, 6.
   - **Hipótese Antigo:** Força letras nas posições 0, 1, 2 e dígitos nas posições 3, 4, 5, 6.
4. **Cálculo da Distância de Substituição:** Conta quantas alterações da matriz de confusão foram necessárias para cada hipótese.
5. **Critério de Seleção:**
   - A hipótese que exigir **menor número de substituições** vence.
   - **Desempate:** Em caso de empate na contagem de alterações, prioriza-se o formato **Mercosul**, por ser o padrão de emplacamento padrão e predominante no Brasil.
6. **Formato Inválido:** Se nenhuma hipótese atender às restrições legais com as substituições permitidas, o formato é classificado como `invalido`.

---

## 6. Modo Livre (Placas Internacionais)

Para análise de veículos com placas de outros países (como Argentina, Uruguai, Paraguai ou União Europeia):

* O usuário pode ativar a opção **"Modo Livre (placas internacionais)"** no painel de controles da interface.
* Nesse modo, a validação posicional estrita brasileira é ignorada: o texto é apenas higienizado (remoção de caracteres não-alfanuméricos) e entregue diretamente ao usuário com o formato marcado como `livre`.

---

## 7. Contrato de Dados (`ResultadoPosProcessamento`)

Definido no módulo [`src/postprocess.py`](../src/postprocess.py):

```python
@dataclass(frozen=True)
class ResultadoPosProcessamento:
    texto_original: str       # Leitura bruta saída do modelo FastPlateOCR
    texto_final: str          # Texto corrigido e validado
    formato: str              # "mercosul" | "antigo" | "invalido" | "livre"
    confianca: float | None   # Confiança estimada média dos caracteres
    substituicoes: int        # Quantidade de caracteres corrigidos posicionalmente
    valido: bool              # True se atende a um formato válido
```
