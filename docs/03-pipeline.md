# 03. Pipeline de processamento

Cada etapa recebe uma imagem e devolve outra, e todas as saídas intermediárias ficam disponíveis para a interface. Os valores padrão de CLAHE e do filtro gaussiano vêm dos testes já feitos no notebook.

## Visão geral

```
Imagem → [E1 Detecção] → [E2 ROI] → [E3 Perspectiva] → [E4 Cinza]
       → [E5 Suavização] → [E6 CLAHE] → [E7 Limiarização]
       → [E8 Segmentação] → [E9 OCR] → [E10 Pós-processamento] → Texto
```

## E1. Detecção da placa

Dois métodos, escolhidos na interface.

### Método clássico (`detection/contours.py`)

1. Converter para cinza.
2. **Filtro bilateral** (`d=11, sigmaColor=17, sigmaSpace=17`): suaviza mantendo bordas.
3. **Canny** (`min=30, max=200`).
4. `findContours` (`RETR_TREE`, `CHAIN_APPROX_SIMPLE`); ordenar por área e manter os 10 maiores.
5. `approxPolyDP` com `epsilon = 0.018 × perímetro`; manter contornos com **4 vértices**.
6. Filtrar por **proporção** (largura/altura entre 1,5 e 6) e por área mínima (por exemplo, 0,3% da imagem).
7. Escolher o melhor candidato (maior área entre os válidos) e devolver o quadrilátero.

Saídas intermediárias: imagem em cinza, mapa de bordas e contornos candidatos.

### Método YOLO (`detection/yolo_onnx.py`)

Letterbox → inferência ONNX → decodificação → NMS → melhor caixa. Detalhes em `05-modelo-onnx.md`.

## E2. Recorte do ROI

- Recorta a caixa com **margem** (padrão 4%, limitada às bordas da imagem).
- Se a caixa for muito pequena (por exemplo, altura menor que 15 px), registra aviso: a leitura tende a falhar.

## E3. Correção de perspectiva e inclinação

- **Método clássico:** com os 4 vértices ordenados (sup-esq, sup-dir, inf-dir, inf-esq), `getPerspectiveTransform` + `warpPerspective` para um retângulo de **400×130 px** (proporção da placa brasileira, 40×13 cm).
- **Método YOLO:** só há caixa, sem vértices. Estimar a inclinação pelo menor retângulo rotacionado do maior contorno após a binarização (`minAreaRect`) e girar o ROI para alinhar. Se o ângulo for pequeno (< 2°), não corrigir.
- **Ampliação:** redimensionar o ROI para altura de cerca de 120 px com `INTER_CUBIC`, pois o OCR rende melhor com caracteres maiores.

## E4. Tons de cinza

`cvtColor(BGR2GRAY)`. Representa a luminosidade e reduz o problema a um canal.

## E5. Suavização

- Filtro gaussiano, **kernel 5×5** por padrão (ímpar, ajustável de 1 a 15).
- Reduz ruído antes da limiarização. Kernel grande demais borra os caracteres.

## E6. CLAHE

- `createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` por padrão.
- Realça o contraste **localmente**, ajudando em placas com sombra ou reflexo.
- `clipLimit` alto amplifica ruído. Ajustável de 1,0 a 8,0.

## E7. Limiarização

| Modo | Quando usar | Parâmetros |
|------|-------------|------------|
| **Otsu** (padrão) | Iluminação uniforme no ROI | nenhum |
| **Adaptativa gaussiana** | Sombras ou gradiente de luz | bloco (ímpar, padrão 31) e constante C (padrão 10) |

**Polaridade:** os componentes conexos esperam caracteres **brancos sobre fundo preto**. Após binarizar, se a maior parte do ROI for branca (placa clara com texto escuro), inverter a imagem automaticamente.

Opcional: abertura/fechamento morfológico com elemento 2×2 para remover ruído pequeno.

## E8. Segmentação de caracteres (`segmentation.py`)

1. `connectedComponentsWithStats` na imagem binarizada.
2. Filtrar componentes por:
   - altura entre **35% e 95%** da altura do ROI;
   - largura menor que 25% da largura do ROI;
   - área mínima (descarta pontos e ruídos).
3. Ordenar da esquerda para a direita (pela coordenada x).
4. Recortar cada caractere e normalizar para um tamanho fixo.
5. Se o número de caracteres for incoerente (menos de 5 ou mais de 8), registrar aviso.

A segmentação serve para visualização e validação. O OCR pode rodar sobre a linha inteira ou sobre os caracteres.

## E9. OCR

Ver `06-ocr-pos-processamento.md`.

## E10. Pós-processamento

Ver `06-ocr-pos-processamento.md`.

## Tabela de parâmetros expostos na interface

| Parâmetro | Etapa | Padrão | Faixa |
|-----------|-------|--------|-------|
| Método de detecção | E1 | YOLO | clássico / YOLO |
| Confiança mínima | E1 (YOLO) | 0,25 | 0,05 a 0,95 |
| IoU do NMS | E1 (YOLO) | 0,45 | 0,1 a 0,9 |
| Canny mínimo / máximo | E1 (clássico) | 30 / 200 | 0 a 255 |
| Margem do ROI | E2 | 4% | 0% a 15% |
| Kernel gaussiano | E5 | 5 | 1 a 15 (ímpar) |
| `clipLimit` do CLAHE | E6 | 2,0 | 1,0 a 8,0 |
| Grade do CLAHE | E6 | 8×8 | 2 a 16 |
| Modo de limiarização | E7 | Otsu | Otsu / adaptativa |
| Bloco e constante adaptativa | E7 | 31 / 10 | bloco ímpar de 3 a 61 |
| Motor e PSM do OCR | E9 | Tesseract / 7 | PSM 6, 7, 8, 13 |

## Imagens intermediárias exibidas

| Etapa | Imagem |
|-------|--------|
| E1 | Original com caixa; bordas e contornos (clássico) |
| E2/E3 | ROI recortado e alinhado |
| E4 a E6 | Cinza, suavizada e com CLAHE |
| E7 | Binarizada |
| E8 | Caracteres segmentados |
