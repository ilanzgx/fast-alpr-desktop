"""Painel lateral de controles com formulários e sincronização slider-spinbox."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.config import ParametrosPipeline


class PainelControles(QWidget):
    """Painel de controles agrupados por etapa com parâmetros e botões de ação."""

    parametros_alterados = Signal()
    carregar_clicado = Signal(str)
    processar_clicado = Signal()
    exportar_clicado = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._caminho_imagem_atual: str | None = None

        layout_principal = QVBoxLayout(self)
        layout_principal.setContentsMargins(0, 0, 0, 0)

        # ScrollArea para permitir rolar em telas menores
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        conteudo = QWidget()
        conteudo.setObjectName("painel_controles_conteudo")
        self._layout_conteudo = QVBoxLayout(conteudo)
        self._layout_conteudo.setSpacing(10)
        self._layout_conteudo.setContentsMargins(8, 8, 8, 8)

        self._criar_grupo_entrada()
        self._criar_grupo_deteccao()
        self._criar_grupo_preprocessamento()
        self._criar_grupo_ocr()
        self._criar_botoes_acao()

        self._layout_conteudo.addStretch(1)
        scroll.setWidget(conteudo)
        layout_principal.addWidget(scroll)

    def _criar_grupo_entrada(self) -> None:
        grupo = QGroupBox("1. Entrada da Imagem")
        layout = QVBoxLayout(grupo)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self._btn_carregar = QPushButton("Carregar Imagem...")
        self._btn_carregar.setMinimumHeight(36)
        self._btn_carregar.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_carregar.clicked.connect(self._abrir_seletor_arquivo)
        layout.addWidget(self._btn_carregar)

        self._lbl_arquivo = QLabel("Nenhum arquivo carregado")
        self._lbl_arquivo.setStyleSheet("color: #64748B; font-size: 11px;")
        self._lbl_arquivo.setWordWrap(True)
        layout.addWidget(self._lbl_arquivo)

        self._layout_conteudo.addWidget(grupo)

    def _criar_grupo_deteccao(self) -> None:
        grupo = QGroupBox("2. Detecção de Placa")
        layout = QVBoxLayout(grupo)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        # Método
        layout.addWidget(QLabel("Método:"))
        self._combo_metodo = QComboBox()
        self._combo_metodo.addItems(["YOLO (Rede Neural ONNX)", "Clássico (Canny + Contornos)"])
        self._combo_metodo.currentIndexChanged.connect(self._on_metodo_alterado)
        layout.addWidget(self._combo_metodo)

        # Controles YOLO
        self._widget_yolo = QWidget()
        layout_yolo = QVBoxLayout(self._widget_yolo)
        layout_yolo.setContentsMargins(0, 0, 0, 0)
        layout_yolo.setSpacing(6)

        self._slider_conf, self._spin_conf = self._criar_slider_decimal(
            "Confiança mínima:", 0.05, 0.95, 0.25, 0.05, layout_yolo
        )
        self._slider_nms, self._spin_nms = self._criar_slider_decimal(
            "IoU (NMS):", 0.10, 0.90, 0.45, 0.05, layout_yolo
        )
        layout.addWidget(self._widget_yolo)

        # Controles Clássico
        self._widget_classico = QWidget()
        layout_classico = QVBoxLayout(self._widget_classico)
        layout_classico.setContentsMargins(0, 0, 0, 0)
        layout_classico.setSpacing(6)

        self._slider_canny_min, self._spin_canny_min = self._criar_slider_inteiro(
            "Canny Mínimo:", 0, 255, 30, layout_classico
        )
        self._slider_canny_max, self._spin_canny_max = self._criar_slider_inteiro(
            "Canny Máximo:", 0, 255, 200, layout_classico
        )
        layout.addWidget(self._widget_classico)
        self._widget_classico.setVisible(False)

        self._layout_conteudo.addWidget(grupo)

    def _criar_grupo_preprocessamento(self) -> None:
        grupo = QGroupBox("3. Pré-processamento")
        layout = QVBoxLayout(grupo)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        # Kernel Gaussiano (ímpares de 1 a 15)
        self._slider_kernel, self._spin_kernel = self._criar_slider_inteiro(
            "Kernel Gaussiano:", 1, 15, 5, layout, passo=2
        )

        # CLAHE clip
        self._slider_clahe, self._spin_clahe = self._criar_slider_decimal(
            "CLAHE Clip Limit:", 1.0, 8.0, 2.0, 0.5, layout
        )

        # Limiarização
        layout.addWidget(QLabel("Limiarização:"))
        self._combo_limiar = QComboBox()
        self._combo_limiar.addItems(["Otsu", "Adaptativa Gaussiana"])
        self._combo_limiar.currentIndexChanged.connect(self._on_limiar_alterado)
        layout.addWidget(self._combo_limiar)

        # Parâmetros adaptativos
        self._widget_adaptativo = QWidget()
        layout_adaptativo = QVBoxLayout(self._widget_adaptativo)
        layout_adaptativo.setContentsMargins(0, 0, 0, 0)
        layout_adaptativo.setSpacing(6)

        self._slider_bloco, self._spin_bloco = self._criar_slider_inteiro(
            "Bloco Adaptativo:", 3, 61, 31, layout_adaptativo, passo=2
        )
        self._slider_c, self._spin_c = self._criar_slider_inteiro(
            "Constante C:", 1, 30, 10, layout_adaptativo
        )
        layout.addWidget(self._widget_adaptativo)
        self._widget_adaptativo.setVisible(False)

        self._layout_conteudo.addWidget(grupo)

    def _criar_grupo_ocr(self) -> None:
        grupo = QGroupBox("4. OCR e Pós-processamento")
        layout = QVBoxLayout(grupo)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)

        lbl_motor = QLabel("Motor: <b>FastPlateOCR</b> (Rede Neural ONNX)")
        lbl_motor.setStyleSheet("color: #4A5568; font-size: 11px;")
        layout.addWidget(lbl_motor)

        # Opções
        self._chk_livre = QCheckBox("Modo Livre (placas internacionais)")
        self._chk_livre.stateChanged.connect(lambda: self.parametros_alterados.emit())
        layout.addWidget(self._chk_livre)

        self._layout_conteudo.addWidget(grupo)

    def _criar_botoes_acao(self) -> None:
        layout = QVBoxLayout()
        layout.setSpacing(8)

        self._btn_processar = QPushButton("Processar Imagem")
        self._btn_processar.setObjectName("btn_primario")
        self._btn_processar.setMinimumHeight(42)
        self._btn_processar.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_processar.setEnabled(False)
        self._btn_processar.clicked.connect(self.processar_clicado.emit)
        layout.addWidget(self._btn_processar)

        self._btn_exportar = QPushButton("Exportar Resultados...")
        self._btn_exportar.setMinimumHeight(36)
        self._btn_exportar.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_exportar.setEnabled(False)
        self._btn_exportar.clicked.connect(self.exportar_clicado.emit)
        layout.addWidget(self._btn_exportar)

        self._layout_conteudo.addLayout(layout)

    def _criar_slider_inteiro(
        self, rotulo: str, vmin: int, vmax: int, vpadrao: int, layout_pai: QVBoxLayout, passo: int = 1
    ) -> tuple[QSlider, QSpinBox]:
        layout_rotulo = QHBoxLayout()
        layout_rotulo.setContentsMargins(0, 0, 0, 0)
        layout_rotulo.setSpacing(6)
        layout_rotulo.addWidget(QLabel(rotulo))

        spin = QSpinBox()
        spin.setRange(vmin, vmax)
        spin.setSingleStep(passo)
        spin.setValue(vpadrao)
        spin.setFixedWidth(54)
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_rotulo.addWidget(spin)
        layout_pai.addLayout(layout_rotulo)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(vmin, vmax)
        slider.setSingleStep(passo)
        slider.setValue(vpadrao)
        layout_pai.addWidget(slider)

        # Sincronização
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(lambda: self.parametros_alterados.emit())

        return slider, spin

    def _criar_slider_decimal(
        self, rotulo: str, vmin: float, vmax: float, vpadrao: float, passo: float, layout_pai: QVBoxLayout
    ) -> tuple[QSlider, QDoubleSpinBox]:
        layout_rotulo = QHBoxLayout()
        layout_rotulo.setContentsMargins(0, 0, 0, 0)
        layout_rotulo.setSpacing(6)
        layout_rotulo.addWidget(QLabel(rotulo))

        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setSingleStep(passo)
        spin.setValue(vpadrao)
        spin.setFixedWidth(54)
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_rotulo.addWidget(spin)
        layout_pai.addLayout(layout_rotulo)

        # Slider trabalha em inteiros (escala x100)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(vmin * 100), int(vmax * 100))
        slider.setSingleStep(int(passo * 100))
        slider.setValue(int(vpadrao * 100))
        layout_pai.addWidget(slider)

        def on_slider(val: int) -> None:
            spin.setValue(val / 100.0)

        def on_spin(val: float) -> None:
            slider.setValue(int(val * 100))

        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)
        slider.valueChanged.connect(lambda: self.parametros_alterados.emit())

        return slider, spin

    def _on_metodo_alterado(self, index: int) -> None:
        eh_yolo = (index == 0)
        self._widget_yolo.setVisible(eh_yolo)
        self._widget_classico.setVisible(not eh_yolo)
        self.parametros_alterados.emit()

    def _on_limiar_alterado(self, index: int) -> None:
        eh_adaptativa = (index == 1)
        self._widget_adaptativo.setVisible(eh_adaptativa)
        self.parametros_alterados.emit()

    def _abrir_seletor_arquivo(self) -> None:
        caminho, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Imagem do Veículo",
            str(ParametrosPipeline().metodo),
            "Imagens (*.jpg *.jpeg *.png *.bmp)",
        )
        if caminho:
            self.definir_arquivo_carregado(caminho)
            self.carregar_clicado.emit(caminho)

    def definir_arquivo_carregado(self, caminho: str) -> None:
        self._caminho_imagem_atual = caminho
        nome = caminho.replace("\\", "/").split("/")[-1]
        self._lbl_arquivo.setText(f"Arquivo: {nome}")
        self._btn_processar.setEnabled(True)

    def habilitar_exportacao(self, habilitar: bool = True) -> None:
        self._btn_exportar.setEnabled(habilitar)

    def definir_estado_processando(self, processando: bool) -> None:
        if processando:
            self._btn_processar.setEnabled(False)
            self._btn_processar.setText("Processando...")
        else:
            self._btn_processar.setEnabled(self._caminho_imagem_atual is not None)
            self._btn_processar.setText("Processar Imagem")

    def obter_parametros(self) -> ParametrosPipeline:
        """Coleta os valores atuais dos controles da interface."""
        metodo = "yolo" if self._combo_metodo.currentIndex() == 0 else "classico"
        binarizacao = "otsu" if self._combo_limiar.currentIndex() == 0 else "adaptativa"

        # Garante que o kernel seja ímpar
        k = self._spin_kernel.value()
        if k % 2 == 0:
            k += 1

        b = self._spin_bloco.value()
        if b % 2 == 0:
            b += 1

        return ParametrosPipeline(
            metodo=metodo,
            confianca_min=self._spin_conf.value(),
            iou_nms=self._spin_nms.value(),
            canny_min=self._spin_canny_min.value(),
            canny_max=self._spin_canny_max.value(),
            kernel_gauss=k,
            clahe_clip=self._spin_clahe.value(),
            clahe_grade=8,
            binarizacao=binarizacao,
            bloco_adaptativo=b,
            constante_adaptativa=self._spin_c.value(),
            ocr_engine="fast_plate_ocr",
            modo_livre=self._chk_livre.isChecked(),
        )
