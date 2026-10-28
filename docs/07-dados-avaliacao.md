# 07. Dados e avaliação

## 1. Fontes de dados

| Conjunto | Origem | Uso | Observação |
|----------|--------|-----|------------|
| Dataset público | *License Plate Object Detection* (Hugging Face), cerca de 8,8 mil imagens | Treino e validação do detector | Internacional; confirmar licença no *dataset card* |
| Fotos próprias | Veículos com autorização do proprietário | **Teste reservado** | Placas brasileiras, de preferência 30 a 50 imagens |
| Complemento brasileiro (opcional) | Dataset público brasileiro com licença compatível | Treino e ajuste | Registrar origem e licença |

Toda fonte é registrada em `data/README.md` com nome, link, licença, data de acesso e condições de captura.

## 2. Separação dos conjuntos

| Conjunto | Para quê | Regra |
|----------|----------|-------|
| **Treino** | Ajustar os pesos do YOLO | Só o Colab usa |
| **Validação** | Escolher hiperparâmetros e parar o treino | Não entra na métrica final |
| **Ajuste** (`data/ajuste/`) | Calibrar parâmetros do pipeline (CLAHE, limiar, PSM...) | Pode ser consultado à vontade |
| **Teste** (`data/teste/`) | Medir o desempenho final | **Não é usado para ajustar nada** |

**Regras**
- Separar as pastas antes de olhar qualquer resultado.
- Se uma imagem do teste for usada para corrigir algo, ela passa para o ajuste e é substituída.
- Registrar a divisão em arquivo (lista de nomes por conjunto) para que seja reproduzível.

## 3. Gabarito

O dataset traz **apenas as caixas** das placas, sem o texto. Para avaliar o OCR é preciso transcrever manualmente o texto do conjunto de teste:

```csv
arquivo,texto,tipo,condicao
foto_001.jpg,ABC1D23,mercosul,dia
foto_002.jpg,XYZ1234,antigo,sombra
```

Campos úteis em `condicao`: dia, noite, sombra, reflexo, ângulo, distante, suja. Eles permitem agrupar as falhas depois. As caixas de referência para a detecção ficam em formato YOLO (`.txt`).

## 4. Métricas

### Detecção
| Métrica | Definição |
|---------|-----------|
| **Taxa de detecção** | Imagens em que a caixa prevista tem IoU ≥ 0,5 com a de referência ÷ total |
| **Precisão / Recall** | Padrão para detecção de objetos |
| **mAP@0,5** | Média da precisão média (apenas para o YOLO) |
| **Falsos positivos** | Caixas previstas sem placa correspondente |

### Leitura
| Métrica | Definição |
|---------|-----------|
| **Acerto exato** | Texto final idêntico ao gabarito ÷ total de placas |
| **Acerto por caractere** | 1 − CER (distância de Levenshtein ÷ comprimento do gabarito) |
| **Formato válido** | Proporção de saídas aceitas por uma das regras |
| **Ganho do pós-processamento** | Acerto exato com e sem as correções |

### Sistema
| Métrica | Definição |
|---------|-----------|
| **Taxa ponta a ponta** | Placas detectadas **e** lidas corretamente ÷ total |
| **Tempo médio** | Milissegundos por imagem e por etapa, em CPU |

## 5. Experimentos planejados

| ID | Pergunta | Variáveis | Métrica principal |
|----|----------|-----------|-------------------|
| EXP-01 | Clássico ou YOLO detecta melhor? | Método de detecção | Taxa de detecção, falsos positivos, tempo |
| EXP-02 | O pré-processamento ajuda o OCR? | Com e sem: suavização, CLAHE, limiarização (ablação) | Acerto exato, CER |
| EXP-03 | Qual PSM funciona melhor? | PSM 6, 7, 8, 13 | Acerto exato |
| EXP-04 | Tesseract ou EasyOCR? | Motor de OCR | Acerto exato, CER, tempo |
| EXP-05 | A leitura robusta compensa? | Uma tentativa × multi-tentativa | Acerto exato × tempo |
| EXP-06 | Quanto o pós-processamento ajuda? | Com e sem correção por posição | Ganho em acerto exato |
| EXP-07 | Fine-tuning melhora o detector? | Modelo existente × ajustado | mAP, recall |

**Protocolo:** escolher os parâmetros no conjunto de **ajuste**, congelá-los, e rodar **uma vez** no conjunto de **teste**. Reportar os números do teste.

## 6. Modelo de tabela de resultados

| Configuração | Detecção | Acerto exato | Acerto por caractere | Tempo (ms) |
|--------------|----------|--------------|----------------------|------------|
| Clássico + Tesseract (PSM 7) | _a medir_ | _a medir_ | _a medir_ | _a medir_ |
| YOLO + Tesseract (PSM 7) | _a medir_ | _a medir_ | _a medir_ | _a medir_ |
| YOLO + Tesseract (leitura robusta) | _a medir_ | _a medir_ | _a medir_ | _a medir_ |
| YOLO + EasyOCR | _a medir_ | _a medir_ | _a medir_ | _a medir_ |

Os resultados vão para `results/metricas.csv`, e os scripts de avaliação ficam em `notebooks/03_avaliacao.ipynb` ou `scripts/avaliar.py`.

## 7. Catálogo de falhas

Cada falha relevante é registrada com a causa provável:

| Imagem | Esperado | Obtido | Etapa que falhou | Causa provável | Ação |
|--------|----------|--------|------------------|----------------|------|
| `foto_014.jpg` | ABC1D23 | ABC1O23 | OCR | Confusão D/O em baixa resolução | Ampliação maior do ROI |
| _exemplo_ | | | Detecção | Reflexo cobrindo a placa | Reduzir confiança mínima |

**Causas recorrentes a observar:** reflexo, ângulo forte, baixa luz, placa suja, placa pequena (distante), fundo poluído e fonte atípica.

## 8. Resultados já observados (qualitativos)

- Detecção: o modelo ONNX localizou a placa nos exemplos testados.
- OCR: `AFR2011` foi lida como `AER2011` e `52Y6490` não foi lida no ajuste inicial. O recorte ajustado e o PSM 13 corrigiram esses casos.
- **Limite:** são três exemplos e não indicam precisão geral. A avaliação sistemática acima é o próximo passo.
