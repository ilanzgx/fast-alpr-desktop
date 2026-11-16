# 01-DETECCAO: Detecção e Localização de Placas

A etapa de detecção (E1) localiza a placa veicular na imagem de entrada e delimita suas coordenadas por meio de uma caixa delimitadora (*bounding box*). O sistema disponibiliza dois métodos configuráveis via `ParametrosPipeline.metodo` ou na interface gráfica: o detector baseado em rede neural (YOLOv8 em ONNX) e o método geométrico clássico.

---

## 1. Método Neural: YOLOv8 em ONNX

O método padrão utiliza um modelo YOLOv8 exportado para ONNX (`models/placas.onnx`), executado localmente na CPU com o `onnxruntime` através da classe `DetectorYolo` em [src/detection/yolo_onnx.py](src/detection/yolo_onnx.py). A sessão de inferência é mantida em cache (`_DETECTOR_YOLO_CACHE` em [src/pipeline.py](src/pipeline.py)) para evitar o custo de recarregar o arquivo do modelo a cada chamada.

### Pré-processamento da Entrada
A rede neural opera com tensores quadrados de dimensão fixa $640 \times 640$:
1. **Letterbox (`letterbox`)**: Redimensiona a imagem proporcionalmente até que a maior dimensão atinja 640 pixels, preenchendo as bordas restantes com cinza neutro (valor 114) para preservar a razão de aspecto original.
2. **Normalização (`preparar`)**: Converte a imagem de BGR para RGB, normaliza os valores dos pixels para o intervalo $[0.0, 1.0]$ e reordena os eixos para o formato de entrada `(1, 3, 640, 640)`.

### Inferência e Decodificação
A saída da inferência suporta tensores no formato `[1, 5, 8400]` (YOLOv8 com classe única) e `[1, N, 6]` (padrão YOLOv5/v7):
1. **Filtro de Confiança**: Descarta previsões com probabilidade inferior ao limiar mínimo (`confianca_min`, padrão 0.25).
2. **Deslocamento de Coordenadas**: Remove a escala e os preenchimentos do letterbox, reprojetando as caixas nas dimensões reais da imagem original.
3. **Supressão de Não-Máximos (NMS)**: Algoritmo implementado em NumPy (`nms`) que ordena as detecções por confiança e suprime sobreposições com Intersection over Union (IoU) superior a 0.45 (`iou_nms`).
4. **Seleção**: Retorna a caixa de maior confiança como objeto imutável `Caixa(x1, y1, x2, y2, confianca)`.

---

## 2. Método Clássico: Canny e Filtragem Geométrica de Contornos

O método clássico, implementado em `detectar_placa_classica` ([src/detection/contours.py](src/detection/contours.py)), localiza a placa exclusivamente por processamento de imagem tradicional, sem redes neurais:

1. **Escala de Cinza**: Converte a imagem original para canal único de luminância.
2. **Filtro Bilateral**: Aplica suavização com preservação de bordas (`d=11, sigmaColor=17, sigmaSpace=17`) para atenuar ruídos da pintura e textura do veículo.
3. **Detector de Bordas Canny**: Computa o mapa de gradientes utilizando limiares ajustáveis (`canny_min` e `canny_max`, padrões 30 e 200).
4. **Aproximação Poligonal**:
   - Extrai contornos com `cv2.findContours` e ordena os 15 maiores por área.
   - Aplica `cv2.approxPolyDP` com tolerância de 1.8% do perímetro do contorno.
5. **Filtragem Geométrica**:
   - Exige exatamente 4 vértices (quadrilátero fechado e convexo).
   - Área mínima de 0.3% da área total da imagem.
   - Razão de aspecto (largura/altura) restrita ao intervalo entre 1.5 e 6.0.

---

## 3. Saídas da Etapa

Ao final da etapa E1, o pipeline produz:
- O objeto `Caixa` contendo as coordenadas da placa (`x1, y1, x2, y2`) e sua confiança.
- A imagem original anotada com o retângulo delimitador e percentual de confiança.
- Um mapa visual de bordas Canny para exibição no painel de etapas intermediárias da interface.
