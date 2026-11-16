# 03-PREPROCESSAMENTO: Pré-Processamento e Binarização Adaptativa

As etapas E4 a E7 convertem o ROI recortado e retificado da placa em uma máscara binária de alto contraste, preparando a imagem para a segmentação de caracteres e para os motores de OCR. As rotinas estão concentradas em [src/preprocessing.py](src/preprocessing.py).

---

## 1. Conversão para Tons de Cinza (E4)

A função `converter_cinza` reduz os canais BGR da imagem recortada para um único canal de luminância através de `cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)`. Esta etapa elimina a variação de cores de lataria e fundo, concentrando o processamento exclusivamente na intensidade luminosa.

---

## 2. Suavização Gaussiana (E5)

A função `suavizar_gauss` aplica um filtro linear Gaussiano bidimensional com desvio padrão calculado automaticamente a partir do tamanho do kernel (`kernel_gauss`, padrão $5 \times 5$).

- **Objetivo**: Atenuar artefatos de compressão JPEG, ruído sensor de alta frequência e micro-reflexos metálicos.
- **Validação**: Garante que o tamanho do kernel seja estritamente ímpar; caso um valor par seja informado nos parâmetros ou interface, ele é incrementado em uma unidade.

---

## 3. Realce de Contraste Adaptativo - CLAHE (E6)

A função `aplicar_clahe` emprega o algoritmo CLAHE (*Contrast Limited Adaptive Histogram Equalization*):

- **Grade Local**: Divide o ROI em uma grade de $8 \times 8$ blocos homogêneos (`clahe_grade=8`).
- **Equalização com Corte**: Calcula o histograma local de cada bloco e redistribui intensidades respeitando o limiar de contraste (`clahe_clip=2.0`).
- **Interpolação Bilinear**: Combina as bordas dos blocos por interpolação para evitar descontinuidades visíveis.
- **Resultado Prático**: Revela caracteres sombreados por para-choques sem superexpor regiões iluminadas por luz solar direta ou faróis.

---

## 4. Binarização e Normalização de Polaridade (E7)

A função `binarizar` converte a imagem em níveis de cinza realçada para uma máscara booleana estrita (valores 0 e 255). O sistema disponibiliza dois métodos:

### Métodos de Limiarização
1. **Otsu (`modo="otsu"`)**:
   - Calcula automaticamente um limiar global ótimo que minimiza a variância intraclasse das duas distribuições de intensidade (fundo e caracteres).
   - Indicado para imagens com iluminação homogênea em toda a extensão da placa.
2. **Adaptativa Gaussiana (`modo="adaptativa"`)**:
   - Calcula um limiar específico para cada pixel baseado na média ponderada Gaussiana de uma vizinhança local de tamanho configurável (`bloco_adaptativo`, padrão $31 \times 31$), subtraída de uma constante (`constante_adaptativa`, padrão 10).
   - Indicada para placas com sombras parciais ou variações de iluminação transversal.

### Inversão de Polaridade Consciente do Miolo
A segmentação e a análise de componentes conexos exigem caracteres claros (255) sobre fundo escuro (0). Como placas automotivas possuem caracteres escuros sobre fundo claro (branco no padrão Mercosul ou cinza no padrão anterior), a máscara precisa de inversão fotométrica.

Para determinar a polaridade correta sem ser enganada pelo entorno escuro da lataria ou grade trazido pelo zoom-out:
1. `binarizar` analisa a densidade de pixels claros exclusivamente na **janela central da placa** (intervalo vertical de 25% a 85% e horizontal de 15% a 85%), ignorando as margens externas.
2. Se a maioria dos pixels dentro dessa janela central for clara, o algoritmo aplica `cv2.bitwise_not`, invertendo a imagem para produzir caracteres brancos sobre fundo preto, preservando a legibilidade dos glifos para a segmentação.
