# 04-SEGMENTACAO: Segmentação de Caracteres por Componentes Conexos

A etapa E8 segmenta e isola cada caractere da placa a partir da máscara binária produzida na etapa anterior. Os glifos resultantes são normalizados individualmente para exibição na interface gráfica e utilizados como entrada auxiliar para a leitura do OCR. A lógica está implementada em [src/segmentation.py](src/segmentation.py).

---

## 1. Extração de Componentes Conexos

A função `segmentar_caracteres` processa a imagem binarizada através de `cv2.connectedComponentsWithStats` com conectividade 8. O algoritmo identifica todas as regiões conexas de pixels brancos (255) e extrai suas propriedades geométricas:
- Coordenadas da caixa delimitadora (`x, y, w, h`).
- Área total do componente em pixels.
- Mapa de rótulos (*label matrix*) com a identificação de cada ilha de pixels.

---

## 2. Critérios de Filtragem Geométrica

Para distinguir caracteres verdadeiros de parafusos, rebites, bordas e ruídos de fundo, cada componente conexo é submetido a filtros geométricos proporcionais às dimensões do ROI:

1. **Altura Proporcional**:
   - Altura mínima de 29% da altura do ROI (`altura_min_pct=0.29`), compatível com placas com margem contextual de zoom-out.
   - Altura máxima de 95% da altura do ROI (`altura_max_pct=0.95`), descartando colunas ou molduras verticais contínuas.
2. **Largura Proporcional**:
   - Largura mínima de 3 pixels, descartando artefatos pontuais e linhas finas.
   - Largura máxima de 25% da largura total do ROI (`largura_max_pct=0.25`), evitando blocos de caracteres fundidos ou barras horizontais.
3. **Área Mínima**:
   - Área mínima proporcional à altura mínima e largura do ROI ($0.03 \times h_{min} \times W_{ROI}$), eliminando elementos menores como o texto "BR", padrões do QR Code e cabeças de parafusos.

---

## 3. Descarte de Fragmentos de Cantos e Bordas

Para evitar que resíduos da lataria, grade ou moldura externa presentes nas extremidades do ROI sejam interpretados como caracteres:
- O algoritmo verifica se o componente conexo toca simultaneamente uma borda lateral (`x <= 1` ou `x + w >= W - 1`) e uma borda vertical (`y <= 1` ou `y + h >= H - 1`).
- Componentes ancorados nos cantos extremos são descartados de imediato, enquanto caracteres reais próximos às margens permanecem preservados.

---

## 4. Isolamento Estrito por Mapa de Rótulos

Para assegurar que caracteres vizinhos inclinados ou com serifas próximas não sofram sobreposição:
- O recorte de cada caractere é extraído diretamente da matriz de rótulos do OpenCV:
  ```python
  mascara_caractere = np.where(labels[y : y + h, x : x + w] == i, 255, 0).astype(np.uint8)
  ```
- Apenas os pixels indexados pelo rótulo `i` do componente atual são mantidos como brancos; qualquer pixel de glifos vizinhos que invada a caixa delimitadora é convertido em fundo preto (0).

---

## 5. Padronização e Montagem de Faixa

Os componentes aprovados passam pelas seguintes etapas de finalização:

1. **Ordenação**: São ordenados da esquerda para a direita com base na coordenada horizontal inicial (`x`).
2. **Normalização Individual (`normalizar_caractere`)**:
   - Cada glifo é redimensionado proporcionalmente com interpolação cúbica para caber em um quadro padronizado de $40 \times 60$ pixels, respeitando uma margem interna de 5 pixels.
   - A imagem passa por limiarização fixa em 127 para manter bordas bem definidas.
3. **Montagem de Faixa (`montar_faixa_caracteres`)**:
   - Cria uma faixa horizontal unificada com espaçamento uniforme entre caracteres, apresentada em texto escuro sobre fundo claro para alimentação de motores ópticos clássicos.
