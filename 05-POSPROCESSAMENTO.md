# 05-POSPROCESSAMENTO: OCR Neural, Validação e Correção Posicional de Placas

As etapas E9 e E10 convertem a informação visual em texto alfanumérico e aplicam regras de validação e correção para o padrão veicular brasileiro. O reconhecimento de caracteres está implementado em [src/ocr.py](src/ocr.py) e o pós-processamento contextual em [src/postprocess.py](src/postprocess.py).

---

## 1. Motor de Reconhecimento Neural: FastPlateOCR (E9)

A aplicação utiliza como motor de OCR a rede neural **FastPlateOCR**, baseada na arquitetura **Compact Convolutional Transformer (CCT-S v2)**, executada localmente via `onnxruntime` em CPU com cache de sessão singleton (`obter_recognizer_fastplate`):

- **Arquitetura CCT Transformer**: Combina blocos convolucionais para extração de mapas de características com camadas de mecanismo de auto-atenção (*self-attention*), modelando dependências espaciais e sequenciais entre os caracteres da placa.
- **Entrada Direta**: Opera diretamente sobre o ROI da placa recortado e retificado (ou escala de cinza convertida para 3 canais BGR), sem depender da binarização intermediária ou de segmentações manuais sujeitas a ruídos de lataria.
- **Robustez Industrial**: Alta imunidade a reflexos metálicos, variações de relevo estrito, parafusos de fixação, faixas azuis do Mercosul e sombras parciais de para-choques.
- **Desempenho no Benchmark (120 Amostras Reais)**:
  - **95,8% de Acerto Exato** (115 de 120 placas perfeitamente decodificadas).
  - **99,3% de Acurácia de Caractere**.
  - **0,7% de Taxa de Erro de Caractere (CER)**.
  - Tempo médio de inferência de apenas **~100 ms** na CPU.
- **Confiança Ponderada**: Extrai as probabilidades individuais de cada caractere (`char_probs`) e computa a confiança média para apresentação transparente na interface gráfica.
- **Operação 100% Offline**: Pesos ONNX (`models/ocr/cct_s_v2_global.onnx`) e configurações YAML são embutidos diretamente no aplicativo e no executável PyInstaller.

---

## 2. Validação e Correção Posicional (E10)

A função `corrigir_por_posicao` em [src/postprocess.py](src/postprocess.py) recebe o texto bruto do OCR e aplica a normalização sintática de acordo com a legislação brasileira de trânsito.

### Normalização Inicial
- `limpar_texto`: Remove pontuações, hífens, espaços e caracteres fora do conjunto alfanumérico ASCII básico (`[A-Z0-9]`), convertendo tudo para letras maiúsculas.

### Modo Livre (`modo_livre=True`)
- Mantém o texto limpo sem impor a gramática das placas brasileiras, permitindo a leitura de placas internacionais (Mercosul de outros países, União Europeia, etc.) ou frotas especiais. O formato é classificado como `"livre"`.

### Validação de Padrões Brasileiros (`modo_livre=False`)
O texto é comparado com as expressões regulares oficiais:
- **Padrão Mercosul**: Três letras, um dígito, uma letra e dois dígitos (`^[A-Z]{3}[0-9][A-Z][0-9]{2}$`).
- **Padrão Cinza/Antigo**: Três letras e quatro dígitos (`^[A-Z]{3}[0-9]{4}$`).

Se o texto de 7 caracteres for estritamente válido em um dos formatos, é retornado de imediato com 0 substituições.

---

## 3. Matriz de Confusão Contextual

Quando a leitura de 7 caracteres contém ambiguidades visuais decorrentes de iluminação ou reflexo, o algoritmo avalia a distribuição posicional legal obrigatória:

```
Posição:   0   1   2   3   4   5   6
Mercosul: [L] [L] [L] [D] [L] [D] [D]   (L = Letra, D = Dígito)
Antigo:   [L] [L] [L] [D] [D] [D] [D]
```

- **Posições 0, 1 e 2**: Devem ser estritamente letras (`A-Z`).
- **Posição 3**: Deve ser estritamente dígito (`0-9`).
- **Posição 4**: Letra no Mercosul ou dígito no padrão Antigo.
- **Posições 5 e 6**: Devem ser estritamente dígitos (`0-9`).

### Mapeamentos de Substituição Bidirecionais

| Letra $\rightarrow$ Dígito | Dígito $\rightarrow$ Letra |
|:---:|:---:|
| `O` $\rightarrow$ `0`, `D` $\rightarrow$ `0`, `Q` $\rightarrow$ `0` | `0` $\rightarrow$ `O` |
| `I` $\rightarrow$ `1`, `L` $\rightarrow$ `1`, `T` $\rightarrow$ `1` | `1` $\rightarrow$ `I` |
| `Z` $\rightarrow$ `2` | `2` $\rightarrow$ `Z` |
| `E` $\rightarrow$ `3` | `3` $\rightarrow$ `E` |
| `A` $\rightarrow$ `4` | `4` $\rightarrow$ `A` |
| `S` $\rightarrow$ `5` | `5` $\rightarrow$ `S` |
| `G` $\rightarrow$ `6`, `C` $\rightarrow$ `6` | `6` $\rightarrow$ `G` |
| `T` $\rightarrow$ `7`, `Y` $\rightarrow$ `7` | `7` $\rightarrow$ `T` |
| `B` $\rightarrow$ `8`, `E` $\rightarrow$ `8` | `8` $\rightarrow$ `B` |

### Critério de Decisão e Desempate
O algoritmo avalia as hipóteses **Mercosul** e **Antigo** e contabiliza o número de substituições necessárias:
1. A hipótese com **menor quantidade de substituições** vence.
2. Em caso de empate no número de trocas, o formato **Mercosul** é selecionado por ser o padrão moderno e predominante em circulação.

---

## 4. Janela Deslizante para Ruídos de Borda

Quando a detecção captura resíduos da moldura externa, parafusos ou lataria que resultam em uma leitura com mais de 7 caracteres (ex: `11BRA2E19`), uma **janela deslizante de 7 posições** percorre a cadeia completa. Cada substring de 7 caracteres é testada contra a validação posicional e a que exigir menor custo de substituição é selecionada, descartando os caracteres espúrios das bordas.

---

## 5. Saída Estruturada do Processo

Ao final do pós-processamento, os seguintes dados integram o `ResultadoPipeline`:
- `texto_bruto`: Leitura direta devolvida pelo modelo neural FastPlateOCR.
- `texto_final`: Sequência alfanumérica normalizada e corrigida.
- `formato`: Classificação categórica (`"mercosul"`, `"antigo"`, `"livre"` ou `"invalido"`).
- `confianca_ocr`: Grau de certeza estatística do reconhecimento.
- `substituicoes`: Número de caracteres alterados pela matriz posicional.
