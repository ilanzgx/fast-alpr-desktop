# 01. Visão geral

## 1. Resumo

O sistema recebe a fotografia de um veículo, localiza a placa, recorta e trata a região, segmenta e reconhece os caracteres e apresenta o texto final. Cada etapa intermediária é exibida na interface, e os principais parâmetros podem ser ajustados pelo usuário. Todo o uso é **offline**: o treino do detector acontece uma única vez no Google Colab e o desktop apenas executa o modelo exportado.

## 2. Objetivos

**Geral:** construir um protótipo didático de reconhecimento de placas, baseado em um pipeline de processamento de imagem, com interface gráfica desktop.

**Específicos**

1. Localizar placas por dois métodos selecionáveis (processamento clássico e YOLO em ONNX) e compará-los.
2. Implementar operações próprias de processamento: filtragem, bordas, contornos, correção de perspectiva, CLAHE, limiarização, segmentação de caracteres, letterbox e NMS.
3. Reconhecer os caracteres com OCR e corrigi-los com regras de formato de placa.
4. Avaliar o sistema com imagens reservadas, registrando acertos, falhas e causas.

## 3. Escopo

**Inclui**

- Entrada por imagem estática (JPG/PNG/BMP).
- Detecção clássica e detecção por YOLO (ONNX).
- Pré-processamento, segmentação de caracteres e OCR.
- Validação de formato brasileiro (antigo e Mercosul).
- Interface PySide6 com parâmetros ajustáveis e etapas intermediárias.
- Avaliação quantitativa e catálogo de falhas.

**Não inclui**

- Vídeo, webcam ou rastreamento de veículos.
- Treino dentro do aplicativo desktop.
- Empacotamento em instalador (opcional, ao final).
- Uso em produção ou qualquer fim de fiscalização.

## 4. Requisitos funcionais

| ID | Requisito |
|----|-----------|
| RF01 | Carregar uma imagem por seletor de arquivos. |
| RF02 | Exibir a imagem original. |
| RF03 | Escolher o método de detecção (clássico ou YOLO). |
| RF04 | Detectar a placa e desenhar a caixa na imagem original. |
| RF05 | Recortar a região da placa (ROI) e corrigir inclinação/perspectiva quando possível. |
| RF06 | Aplicar tons de cinza, suavização, CLAHE e limiarização com parâmetros ajustáveis. |
| RF07 | Segmentar os caracteres e exibi-los. |
| RF08 | Reconhecer o texto com OCR (Tesseract; EasyOCR opcional). |
| RF09 | Validar o formato e corrigir caracteres por posição. |
| RF10 | Exibir todas as etapas intermediárias e o texto final, indicando se o formato é válido. |
| RF11 | Reprocessar ao alterar um parâmetro, sem travar a interface. |
| RF12 | Exportar o resultado (imagens das etapas e texto) para uma pasta. |
| RF13 | Informar quando nenhuma placa for encontrada ou o texto não puder ser lido. |

## 5. Requisitos não funcionais

| ID | Requisito |
|----|-----------|
| RNF01 | **Offline:** nenhuma chamada de rede em tempo de execução. |
| RNF02 | **Reprodutível:** `uv sync` e `uv run` bastam, com `uv.lock` versionado. |
| RNF03 | **Desempenho:** menos de 2 s por imagem em CPU comum (meta, a medir). |
| RNF04 | **Responsivo:** processamento fora da thread da interface. |
| RNF05 | **Separação:** a lógica de processamento não depende do Qt. |
| RNF06 | **Testável:** módulos de processamento e regras de formato com testes automatizados. |
| RNF07 | **Privacidade:** nenhuma imagem sai da máquina; placas reais não vão para material público. |

## 6. Critérios de aceite

- Carregar uma imagem, processar e visualizar original, etapas e texto pela interface.
- Pelo menos uma etapa intermediária visível (o projeto exibe várias).
- Alterar um parâmetro atualiza o resultado.
- Ambos os métodos de detecção funcionam e podem ser comparados nas mesmas imagens.
- Avaliação feita em imagens que não foram usadas para ajustar parâmetros.
- Instruções de execução permitem que outra pessoa rode o projeto do zero.

## 7. Rastreabilidade com os requisitos da disciplina

| Exigência | Onde é atendida |
|-----------|-----------------|
| Python | Todo o projeto |
| Interface com carregar, executar e visualizar | `04-interface.md` (RF01, RF04, RF10) |
| Ajuste de parâmetros | `03-pipeline.md` e `04-interface.md` (RF06, RF11) |
| Entrada, resultado final e etapa intermediária | RF02, RF10 |
| Operações próprias, modelos prontos como apoio | `03-pipeline.md`, `05-modelo-onnx.md` |
| Testes em imagens reservadas | `07-dados-avaliacao.md` |
| Análise crítica e discussão de erros | `07-dados-avaliacao.md`, `10-riscos-etica.md` |
| Instruções de execução | `08-setup-repositorio.md` |

## 8. Glossário

| Termo | Significado |
|-------|-------------|
| **ROI** | Região de interesse: o recorte da placa. |
| **CLAHE** | Equalização adaptativa de histograma com limite de contraste. |
| **Limiarização** | Conversão para preto e branco a partir de um limiar (Otsu ou adaptativa). |
| **Letterbox** | Redimensionar mantendo a proporção, completando com bordas até o tamanho de entrada do modelo. |
| **NMS** | Supressão de não-máximos: remove caixas sobrepostas do mesmo objeto. |
| **IoU** | Interseção sobre união entre duas caixas. |
| **ONNX** | Formato aberto de modelo, executável pelo `onnxruntime` sem PyTorch. |
| **PSM** | Modo de segmentação de página do Tesseract (7 = linha única, 13 = linha crua). |
| **CER** | Taxa de erro de caracteres (distância de Levenshtein normalizada). |
