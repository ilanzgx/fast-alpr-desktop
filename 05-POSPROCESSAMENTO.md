# 05-POSPROCESSAMENTO: OCR, Validação e Correção Posicional de Placas

As etapas E9 e E10 convertem a informação visual em texto alfanumérico e aplicam regras de validação e correção para o padrão veicular brasileiro. O reconhecimento de caracteres está implementado em [src/ocr.py](src/ocr.py) e o pós-processamento contextual em [src/postprocess.py](src/postprocess.py).

---

## 1. Motores de Reconhecimento Óptico (E9)

O pipeline suporta dois motores de OCR intercambiáveis, selecionáveis via `ParametrosPipeline.ocr_engine` ou pelo painel lateral da interface gráfica:

### Motor Neural: FastPlateOCR (Padrão)
- **Arquitetura**: Modelo de rede neural especializado (`cct-s-v2-global-model`), executado localmente via `onnxruntime` em CPU com cache de sessão singleton (`obter_recognizer_fastplate`).
- **Entrada Direta**: Opera diretamente sobre o ROI retificado e colorido (ou em escala de cinza convertida para 3 canais), sem depender da segmentação de caracteres ou de binarizações intermediárias.
- **Desempenho**: Execução em cerca de 15 a 25 milissegundos por placa na CPU.
- **Confiança**: Calcula a probabilidade média a partir do vetor de probabilidades dos caracteres (`char_probs`).

### Motor Clássico: Tesseract OCR
- **Configuração Especializada**: Executa com whitelist restrita a caracteres alfanuméricos maiúsculos (`A-Z0-9`) e dicionários linguísticos desativados (`load_system_dawg=0` e correlatos) em `CONFIG_BASE` para eliminar viés de vocabulário comum.
- **Deslanting dos Glifos**:
  - `_estimar_inclinacao_caracteres`: Estima o cisalhamento (*skew*) mediano dos caracteres segmentados via momentos centrais ($\mu_{11} / \mu_{02}$), com intervalo contido entre -0.35 e +0.35.
  - `_deslantar_caractere`: Aplica transformação afim horizontal (`cv2.warpAffine`) com acolchoamento dinâmico para endireitar glifos tombados sem decepar topo ou base.
  - `_montar_faixa_deslantada`: Recompõe a fita horizontal com fator de escala de 0.7, aproximando a altura dos caracteres da zona ideal do modelo LSTM do Tesseract (~35 px).
- **Leitura Isolada e Consenso**:
  - Quando 7 caracteres são segmentados no padrão brasileiro, executa `_ler_caracteres_isolados` em cada glifo deslantado (PSM 10 com fallback para PSM 8), impedindo que o modelo LSTM alucine caracteres por contexto de sequência.
  - No modo robusto (`leitura_robusta=True`), avalia múltiplos recortes de miolo (`recortar_miolo_placa`), variações de imagem (cinza, CLAHE, binarizada) e modos de PSM (6, 7, 8, 11, 13), combinando os candidatos mais votados por posição via `combinar_por_votacao`.

---

## 2. Validação e Correção Posicional (E10)

A função `corrigir_por_posicao` em [src/postprocess.py](src/postprocess.py) recebe o texto bruto do OCR e aplica a normalização sintática de acordo com a legislação brasileira.

### Normalização Inicial
- `limpar_texto`: Remove pontuações, hífens, espaços e caracteres fora do conjunto alfanumérico ASCII básico, convertendo tudo para letras maiúsculas.

### Modo Livre (`modo_livre=True`)
- Mantém o texto limpo sem impor a gramática brasileira, permitindo a leitura de placas internacionais, frotas especiais ou formatos alfanuméricos arbitrários. O formato é classificado como `"livre"`.

### Validação de Padrões Brasileiros (`modo_livre=False`)
O texto é comparado com as expressões regulares oficiais:
- **Padrão Mercosul**: Três letras, um dígito, uma letra e dois dígitos (`^[A-Z]{3}[0-9][A-Z][0-9]{2}$`).
- **Padrão Cinza/Antigo**: Três letras e quatro dígitos (`^[A-Z]{3}[0-9]{4}$`).

Se o texto de 7 caracteres for estritamente válido em um dos formatos, é retornado de imediato sem substituições.

### Matriz de Confusão Contextual
Quando a leitura de 7 caracteres contém erros típicos de OCR, o algoritmo avalia a distribuição posicional obrigatória:
- **Posições 0, 1 e 2**: Devem ser estritamente letras.
- **Posição 3**: Deve ser estritamente dígito.
- **Posição 4**: Letra no Mercosul ou dígito no padrão Antigo.
- **Posições 5 e 6**: Devem ser estritamente dígitos.

O sistema consulta tabelas de correspondência visual bidirecionais:
- Letra para Dígito: `O/D/Q -> 0`, `I/L/T -> 1`, `Z -> 2`, `E -> 3`, `A -> 4`, `S -> 5`, `G/C -> 6`, `B -> 8`, etc.
- Dígito para Letra: `0 -> O`, `1 -> I`, `2 -> Z`, `3 -> E`, `4 -> A`, `5 -> S`, `6 -> G`, `8 -> B`, etc.

A hipótese que exigir a menor quantidade de substituições válidas é selecionada, registrando a contagem de substituições realizadas.

### Janela Deslizante para Textos com Ruído de Borda
Quando o texto extraído contém mais de 7 caracteres devido a bordas da moldura, parafusos ou sombras nas extremidades (ex: `FABC1D23`), uma janela deslizante de 7 posições percorre a cadeia completa. Cada substring de 7 caracteres é testada contra a validação posicional, selecionando a janela com menor custo de substituição (até o limite de tolerância de 2 substituições).

---

## 3. Saída Estruturada do Processo

Ao final do pós-processamento, os seguintes dados integram o `ResultadoPipeline`:
- `texto_bruto`: Leitura direta devolvida pelo motor de OCR.
- `texto_final`: Sequência alfanumérica normalizada e corrigida.
- `formato`: Classificação categórica (`"mercosul"`, `"antigo"`, `"livre"` ou `"invalido"`).
- `confianca_ocr`: Grau de certeza estatística do reconhecimento.
- `substituicoes`: Número de caracteres alterados pela matriz posicional.
