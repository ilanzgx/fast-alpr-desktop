# 05. Modelo YOLO e ONNX

**Princípio:** treinar no Colab (com GPU), exportar para ONNX e rodar no desktop apenas com `onnxruntime`. O projeto local não depende de PyTorch nem de `ultralytics`.

```
Colab:   dataset → fine-tuning YOLO → best.pt → best.onnx
Desktop: best.onnx → onnxruntime → letterbox → inferência → decodificação → NMS → caixa
```

## 1. Ponto de partida

O notebook atual já usa um modelo ONNX de detecção de placas com bons resultados de localização. Há dois caminhos, que podem ser combinados:

| Caminho | Quando |
|---------|--------|
| **Usar o ONNX existente** | Garantir o funcionamento rápido do desktop |
| **Fazer fine-tuning no Colab** | Melhorar a detecção em placas brasileiras e ter métricas próprias para o relatório |

## 2. Treino no Colab

**Dataset:** *License Plate Object Detection* (Hugging Face, `keremberke/license-plate-object-detection`), de origem Roboflow, com cerca de 8,8 mil imagens (6.176 treino, 1.765 validação, 882 teste). É uma base **internacional**, então o ideal é complementá-la com imagens brasileiras. Confirmar a licença no *dataset card*.

**Passos**

1. Ativar GPU: *Ambiente de execução → Alterar tipo de ambiente de execução*.
2. Instalar: `pip install ultralytics`.
3. Obter o dataset no formato YOLO (imagens e `.txt` com `classe cx cy w h` normalizados) e criar o `placas.yaml`:

```yaml
path: /content/dataset
train: images/train
val: images/val
test: images/test
names:
  0: placa
```

4. Treinar a partir de um modelo pré-treinado (*fine-tuning*):

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="placas.yaml", epochs=50, imgsz=640, patience=15)
```

5. Avaliar no conjunto de teste:

```python
model = YOLO("runs/detect/train/weights/best.pt")
metricas = model.val(data="placas.yaml", split="test")
```

6. Registrar precisão, recall, mAP@0,5 e mAP@0,5:0,95 em `results/metricas_yolo.csv`.

**Cuidados**
- Não deixar imagens do teste vazarem para o treino.
- Se o dataset anotado for grande, treinar também uma variante pequena (`yolov8n`) para manter a inferência rápida em CPU.
- Guardar no relatório: versão do modelo, hiperparâmetros, número de épocas e curvas de treino.

## 3. Exportação para ONNX

```python
model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
```

Resultado: `best.onnx`, que vira `models/placas.onnx` no repositório local (fora do git, com link no `models/README.md`).

## 4. Verificação da exportação (obrigatória)

Antes de levar o arquivo para o desktop, conferir no Colab:

```python
import onnxruntime as ort
sess = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
print(sess.get_inputs()[0].name, sess.get_inputs()[0].shape)    # ex.: images [1, 3, 640, 640]
print(sess.get_outputs()[0].name, sess.get_outputs()[0].shape)  # ex.: output0 [1, 5, 8400]
```

Comparar as caixas do `.onnx` com as do `.pt` em algumas imagens (diferença de poucos pixels é normal).

**Formato da saída**

| Versão | Forma típica (1 classe, 640) | Conteúdo |
|--------|------------------------------|----------|
| YOLOv8 | `[1, 5, 8400]` | `cx, cy, w, h, score` por âncora (sem *objectness*), **transposto** |
| YOLOv5 | `[1, 25200, 6]` | `cx, cy, w, h, objectness, score` |

Regra prática: se `shape[1] < shape[2]`, a saída está no formato transposto do YOLOv8. O código deve **detectar o layout** em vez de fixá-lo.

## 5. Inferência local (`detection/yolo_onnx.py`)

### 5.1 Pré-processamento (letterbox)

```python
def letterbox(img, tamanho=640, cor=114):
    h, w = img.shape[:2]
    r = min(tamanho / h, tamanho / w)
    nw, nh = round(w * r), round(h * r)
    redim = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    topo, esq = (tamanho - nh) // 2, (tamanho - nw) // 2
    tela = np.full((tamanho, tamanho, 3), cor, np.uint8)
    tela[topo:topo + nh, esq:esq + nw] = redim
    return tela, r, esq, topo

def preparar(img):
    tela, r, esq, topo = letterbox(img)
    x = cv2.cvtColor(tela, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]          # HWC -> NCHW, com batch
    return x, r, esq, topo
```

### 5.2 Decodificação (YOLOv8, 1 classe)

```python
def decodificar(saida, r, esq, topo, conf_min):
    p = saida[0].T                                 # (8400, 5)
    p = p[p[:, 4] >= conf_min]
    cx, cy, w, h, s = p.T
    x1 = (cx - w / 2 - esq) / r
    y1 = (cy - h / 2 - topo) / r
    x2 = (cx + w / 2 - esq) / r
    y2 = (cy + h / 2 - topo) / r
    return np.stack([x1, y1, x2, y2], 1), s
```

### 5.3 NMS próprio (NumPy)

```python
def nms(caixas, scores, iou_max=0.45):
    ordem = scores.argsort()[::-1]
    manter = []
    while ordem.size:
        i = ordem[0]
        manter.append(i)
        if ordem.size == 1:
            break
        resto = ordem[1:]
        xx1 = np.maximum(caixas[i, 0], caixas[resto, 0])
        yy1 = np.maximum(caixas[i, 1], caixas[resto, 1])
        xx2 = np.minimum(caixas[i, 2], caixas[resto, 2])
        yy2 = np.minimum(caixas[i, 3], caixas[resto, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (caixas[i, 2] - caixas[i, 0]) * (caixas[i, 3] - caixas[i, 1])
        area_r = (caixas[resto, 2] - caixas[resto, 0]) * (caixas[resto, 3] - caixas[resto, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        ordem = resto[iou <= iou_max]
    return manter
```

Validar o NMS próprio comparando com `cv2.dnn.NMSBoxes` em algumas imagens (teste automatizado).

### 5.4 Pós-ajustes

- Limitar as coordenadas às dimensões da imagem original (`np.clip`).
- Se houver várias placas, devolver a de maior confiança (ou todas, se a interface evoluir para isso).
- Carregar a sessão **uma vez** e reutilizar:

```python
class DetectorYolo:
    def __init__(self, caminho: str):
        self.sess = ort.InferenceSession(caminho, providers=["CPUExecutionProvider"])
        self.entrada = self.sess.get_inputs()[0].name

    def detectar(self, img, conf_min=0.25, iou_nms=0.45) -> list["Caixa"]:
        x, r, esq, topo = preparar(img)
        saida = self.sess.run(None, {self.entrada: x})[0]
        caixas, scores = decodificar(saida, r, esq, topo, conf_min)
        idx = nms(caixas, scores, iou_nms)
        ...
```

## 6. Versionamento do modelo

| Item | Onde registrar |
|------|----------------|
| Origem (pré-treinado ou fine-tuning), data, dataset e hiperparâmetros | `models/README.md` |
| Hash (SHA-256) do arquivo `.onnx` | `models/README.md` |
| Métricas no conjunto de teste | `results/metricas_yolo.csv` |
| Link para download (Drive, Hugging Face ou Releases) | `models/README.md` |

## 7. Metas de qualidade (a confirmar com medições)

| Métrica | Meta inicial |
|---------|--------------|
| Recall da detecção (IoU ≥ 0,5) no teste | ≥ 90% |
| Tempo de inferência em CPU (640×640) | < 300 ms |

## 8. Segundo Modelo ONNX: FastPlateOCR (CCT Transformer)

Além do YOLO para detecção (E1), o sistema executa um segundo modelo neural em formato ONNX para o reconhecimento óptico de caracteres (E9):

| Propriedade | Detecção da Placa (E1) | OCR de Caracteres (E9) |
|---|---|---|
| **Modelo** | YOLOv8 License Plate | FastPlateOCR (`cct-s-v2-global-model`) |
| **Arquitetura** | CNN Convolucional com Feature Pyramid | Compact Convolutional Transformer (CCT) |
| **Arquivo Local** | `models/placas.onnx` (~6,2 MB) | `models/ocr/cct_s_v2_global.onnx` (~5,3 MB) |
| **Configuração** | Embutida no grafo ONNX | `models/ocr/cct_s_v2_global_plate_config.yaml` |
| **Inferência** | `onnxruntime` (CPU) | `onnxruntime` (CPU) via `fast_plate_ocr` |
| **Tempo de Execução** | ~230 ms por veículo completo | ~100 ms por placa recortada |

