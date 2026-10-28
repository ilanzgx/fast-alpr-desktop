# 09. Roadmap

O projeto avança por etapas, cada uma terminando em algo demonstrável. Os itens marcados já existem, em forma de notebook, na versão em Colab/Gradio e serão migrados para o desktop.

## Visão geral

| Fase | Foco | Resultado demonstrável |
|------|------|------------------------|
| 0 | Base (feita no notebook) | Interface Gradio, escala de cinza, gaussiano, CLAHE, detecção ONNX e OCR experimental |
| 1 | Migração para PySide6 | Janela desktop com carregar, processar e visualizar |
| 2 | Pipeline modular | Código em `src/` independente da UI, com testes |
| 3 | Detecção e segmentação | Os dois métodos de detecção, ROI, perspectiva, limiarização, caracteres |
| 4 | OCR e pós-processamento | Leitura multi-tentativa e regras de formato |
| 5 | Avaliação | Experimentos EXP-01 a EXP-07 em imagens reservadas |
| 6 | Versão final | Documentação, relatório e apresentação |

## Fase 1: Migração para PySide6

- [ ] Criar o projeto com `uv`, ajustar dependências (`PySide6`, `opencv-python-headless`, `onnxruntime`).
- [ ] Janela principal com **carregar imagem** e dois painéis (original e resultado).
- [ ] `PainelImagem` com redimensionamento proporcional.
- [ ] Funções de conversão `numpy → QPixmap`.

**Pronto quando:** carregar uma foto e vê-la nos dois painéis.

## Fase 2: Pipeline modular

- [ ] Mover as operações do notebook (cinza, gaussiano, CLAHE) para `preprocessing.py`.
- [ ] Criar `ParametrosPipeline` e `ResultadoPipeline`.
- [ ] Criar `pipeline.py` retornando todas as etapas intermediárias.
- [ ] Worker em thread separada e botão **Processar** conectado.
- [ ] Controles de parâmetros (kernel, `clipLimit`, grade) com *debounce*.
- [ ] Testes básicos de pré-processamento.

**Pronto quando:** alterar o `clipLimit` atualiza a imagem sem travar a janela.

## Fase 3: Detecção e segmentação

- [ ] Detecção clássica (`contours.py`) com bordas, contornos e filtro de proporção.
- [ ] Carregar o modelo ONNX com `onnxruntime`; letterbox, decodificação e **NMS próprio**.
- [ ] Teste do NMS contra `cv2.dnn.NMSBoxes`.
- [ ] Seletor de método na interface.
- [ ] Recorte do ROI com margem; correção de perspectiva (clássico) e de inclinação (YOLO).
- [ ] Limiarização (Otsu e adaptativa) e inversão automática de polaridade.
- [ ] Segmentação de caracteres por componentes conexos.
- [ ] Painéis das etapas intermediárias.

**Pronto quando:** os dois métodos desenham a caixa e a interface mostra ROI, binarizada e caracteres.

## Fase 4: OCR e pós-processamento

- [ ] `ocr.py` com Tesseract (PSM e whitelist configuráveis).
- [ ] `postprocess.py` com regras de formato e correção por posição.
- [ ] Testes de `postprocess` (casos e empates).
- [ ] Leitura multi-tentativa ("Leitura robusta").
- [ ] Modo livre para placas não brasileiras.
- [ ] Exibir texto bruto, texto final e selo de formato.
- [ ] Exportar resultados.

**Pronto quando:** carregar uma foto brasileira e obter a placa final com formato validado.

## Fase 5: Avaliação

- [ ] Separar `ajuste` e `teste` e registrar as listas.
- [ ] Transcrever o gabarito do conjunto de teste.
- [ ] Script de avaliação em lote (`scripts/avaliar.py`).
- [ ] Rodar os experimentos EXP-01 a EXP-07.
- [ ] (Opcional) Fine-tuning no Colab e exportação do novo `.onnx`.
- [ ] Catalogar as falhas com causa provável.

**Pronto quando:** `results/metricas.csv` e o catálogo de falhas estão preenchidos com dados do conjunto de teste.

## Fase 6: Versão final

- [ ] README e instruções de execução testadas em uma máquina limpa.
- [ ] Amostras de demonstração autorizadas e anonimizadas em `data/demo/`.
- [ ] Relatório com métodos, resultados, limitações e contribuição de cada integrante.
- [ ] Roteiro da apresentação e ensaio com todos.
- [ ] (Opcional) Empacotamento com PyInstaller, incluindo o Tesseract.

## Backlog por módulo

| Módulo | Itens |
|--------|-------|
| Interface | Painéis, controles, worker, exportação, estados e mensagens, clique para ampliar |
| Detecção | Contornos, ONNX, NMS, seletor de método, fine-tuning |
| Pré-processamento e segmentação | Perspectiva, CLAHE, limiarização, polaridade, componentes conexos |
| OCR, dados e testes | Tesseract, multi-tentativa, regras de formato, gabarito, métricas, relatório |

## Definição de "pronto" para cada item

1. Funciona pela interface (quando aplicável).
2. Tem teste automatizado quando é lógica pura (formato, NMS, filtros).
3. Parâmetros documentados em `03-pipeline.md`.
4. Código revisado por outra pessoa e mesclado na `main`.
5. Sem dados sensíveis ou arquivos pesados versionados.

## Pontos de atenção do cronograma

- A migração para PySide6 muda a ferramenta usada nas entregas anteriores (Colab/Gradio). Vale comunicar a mudança e o motivo (execução offline e desktop) logo no início.
- O gabarito de texto do teste leva tempo. Começar cedo.
- A avaliação só é válida se o conjunto de teste ficar intocado até a Fase 5.
