# 04. Interface PySide6

## 1. Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Reconhecimento de Placas                                                     │
├───────────────────────────┬──────────────────────────────────────────────────┤
│ PAINEL DE CONTROLES       │  IMAGEM ORIGINAL (com caixa da placa)            │
│                           │  ┌────────────────────────────────────────────┐  │
│ ▸ Entrada                 │  │                                            │  │
│   [Carregar imagem]       │  │                                            │  │
│                           │  └────────────────────────────────────────────┘  │
│ ▸ Detecção                │                                                  │
│   Método [Clássico|YOLO]  │  ETAPAS INTERMEDIÁRIAS                           │
│   Confiança  ───●──       │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│   IoU (NMS)  ──●───       │  │ Bordas │ │  ROI   │ │ CLAHE  │ │Binária │     │
│   Canny min/max           │  └────────┘ └────────┘ └────────┘ └────────┘     │
│                           │                                                  │
│ ▸ Pré-processamento       │  CARACTERES SEGMENTADOS                          │
│   Kernel     ──●───       │  [A][B][C][1][D][2][3]                           │
│   CLAHE clip ──●───       │                                                  │
│   Grade      ──●───       │  RESULTADO                                       │
│   Limiar [Otsu|Adaptat.]  │  ┌────────────────────────────┐                  │
│   Bloco / C               │  │  ABC1D23   ✔ Mercosul      │                  │
│                           │  └────────────────────────────┘                  │
│ ▸ OCR                     │  Bruto: ABC1O23                                  │
│   Motor [Tesseract]       │                                                  │
│   PSM   [7]               │                                                  │
│                           │                                                  │
│ [Processar] [Exportar]    │                                                  │
├───────────────────────────┴──────────────────────────────────────────────────┤
│ Status: pronto · detecção 42 ms · pré-proc 8 ms · OCR 95 ms                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 2. Componentes

| Área | Widget Qt | Observações |
|------|-----------|-------------|
| Janela | `QMainWindow` | Painel de controles à esquerda, resultados à direita (`QSplitter`) |
| Grupos de controles | `QGroupBox` | Entrada, Detecção, Pré-processamento, OCR |
| Carregar imagem | `QPushButton` + `QFileDialog` | Filtro: png, jpg, jpeg, bmp |
| Método de detecção | `QComboBox` | Ao trocar, mostra/oculta os controles do método |
| Parâmetros numéricos | `QSlider` + `QSpinBox` ou `QDoubleSpinBox` | Slider e caixa numérica sincronizados, com o valor sempre visível |
| Modo de limiarização | `QComboBox` | Bloco/C só habilitam em "Adaptativa" |
| Processar | `QPushButton` | Desabilitado até haver imagem; vira "Processando…" durante o trabalho |
| Imagem original e etapas | `PainelImagem` (herda de `QWidget`) | Título + `QLabel` com `QPixmap` redimensionado mantendo proporção |
| Caracteres | `QHBoxLayout` de miniaturas | Um `QLabel` por caractere |
| Resultado | `QLineEdit` somente leitura | Fonte grande e monoespaçada; selo de formato ao lado |
| Texto bruto | `QLabel` | Texto cru do OCR, antes das correções |
| Exportar | `QPushButton` + `QFileDialog` (pasta) | Salva etapas e `resultado.txt` |
| Status | `QStatusBar` | Mensagens e tempos por etapa |

## 3. Fluxo de interação

1. O usuário clica em **Carregar imagem**: a original aparece e o botão **Processar** é habilitado.
2. Ao clicar em **Processar** (ou alterar um parâmetro), a UI monta `ParametrosPipeline` e dispara o worker.
3. O worker executa o pipeline e emite `resultado_pronto(ResultadoPipeline)`.
4. A UI atualiza todos os painéis: etapas ausentes (`None`) ficam vazias.
5. O selo de formato mostra **Antigo**, **Mercosul** ou **Inválido**, e `erro` (se houver) aparece no status.

**Reprocessamento ao vivo:** mudanças em sliders disparam o processamento após ~250 ms de inatividade (debounce). Trocar o método de detecção reprocessa imediatamente.

## 4. Estados

| Estado | Condição | Comportamento |
|--------|----------|---------------|
| Vazio | Nenhuma imagem | Painéis com "(vazio)"; Processar e Exportar desabilitados |
| Pronto | Imagem carregada | Processar habilitado |
| Processando | Worker ativo | Botão "Processando…"; cursor de espera; sliders continuam livres |
| Concluído | Resultado recebido | Painéis atualizados; Exportar habilitado |
| Sem placa | `caixa is None` | Mensagem "Nenhuma placa encontrada"; sugere ajustar a confiança ou trocar o método |
| Texto inválido | `formato == "invalido"` | Mostra o texto bruto e destaca em amarelo |
| Erro | Exceção inesperada | Diálogo curto e log no terminal; a janela continua utilizável |

## 5. Sinais e slots (resumo)

| Origem | Sinal | Destino |
|--------|-------|---------|
| Botão Processar | `clicked` | `JanelaPrincipal.executar()` |
| Sliders e combos | `valueChanged` / `currentIndexChanged` | `QTimer` de debounce → `executar()` |
| `PipelineWorker` | `resultado_pronto(object)` | `JanelaPrincipal.atualizar_resultado()` |
| `PipelineWorker` | `falhou(str)` | `JanelaPrincipal.mostrar_erro()` |

## 6. Conversão de imagem para exibição

```python
def numpy_para_pixmap(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:                                   # cinza ou binária
        img = np.ascontiguousarray(img)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:                                               # BGR -> RGB
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())               # copy(): o buffer NumPy pode ser liberado
```

## 7. Boas práticas

- `PainelImagem` reescala o pixmap no `resizeEvent`, guardando o original para não perder qualidade.
- Abrir arquivos com `np.fromfile` + `cv2.imdecode` evita problemas com acentos no caminho (Windows).
- Nenhum processamento pesado na thread da UI. Os sliders nunca devem travar.
- Textos e mensagens em português; erros em linguagem simples, sem *stack trace* para o usuário.
- (Opcional) Clique duplo em um painel abre a imagem em tamanho maior.

## 8. Arquivos previstos

```
src/ui/
├── janela_principal.py   # layout e conexões
├── painel_imagem.py      # widget de exibição
├── painel_controles.py   # grupos de parâmetros
├── worker.py             # QThread/QRunnable do pipeline
└── conversoes.py         # numpy_para_pixmap
src/app.py                # QApplication e ponto de entrada
```
