"""Janela principal da aplicação desktop em PySide6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.config import ResultadoPipeline
from src.ui.conversoes import numpy_para_pixmap
from src.ui.estilo import TEMA_CLARO
from src.ui.painel_controles import PainelControles
from src.ui.painel_imagem import PainelImagem
from src.ui.worker import PipelineWorker
from src.utils import ler_imagem, salvar_imagem


class JanelaPrincipal(QMainWindow):
    """Janela principal do sistema de reconhecimento de placas."""

    def __init__(self, funcao_pipeline) -> None:
        super().__init__()
        self._funcao_pipeline = funcao_pipeline
        self._imagem_atual: np.ndarray | None = None
        self._ultimo_resultado: ResultadoPipeline | None = None
        self._worker: PipelineWorker | None = None

        # Timer para debounce de sliders (~250 ms)
        self._timer_debounce = QTimer(self)
        self._timer_debounce.setSingleShot(True)
        self._timer_debounce.setInterval(250)
        self._timer_debounce.timeout.connect(self._executar_pipeline)

        self.setWindowTitle("Reconhecimento de Placas de Veiculos")
        self.resize(1240, 820)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(TEMA_CLARO)

        self._configurar_interface()
        self.statusBar().showMessage("Pronto. Carregue uma fotografia para iniciar.")

    def _configurar_interface(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # 1. Painel Esquerdo: Controles
        self._controles = PainelControles()
        self._controles.carregar_clicado.connect(self._ao_carregar_imagem)
        self._controles.processar_clicado.connect(self._executar_pipeline)
        self._controles.parametros_alterados.connect(self._ao_alterar_parametros)
        self._controles.exportar_clicado.connect(self._ao_exportar)
        splitter.addWidget(self._controles)

        # 2. Painel Direito: Visualização e Resultados
        scroll_direito = QScrollArea()
        scroll_direito.setWidgetResizable(True)
        widget_direito = QWidget()
        layout_direito = QVBoxLayout(widget_direito)
        layout_direito.setContentsMargins(16, 16, 16, 16)
        layout_direito.setSpacing(14)

        # 2.1 Imagem Principal (Original / Anotada)
        self._painel_original = PainelImagem("Imagem Original / Placa Localizada")
        self._painel_original.setMinimumHeight(280)
        layout_direito.addWidget(self._painel_original, stretch=2)

        # 2.2 Grade de Etapas Intermediárias
        layout_grid = QGridLayout()
        layout_grid.setSpacing(10)

        self._painel_bordas = PainelImagem("1. Bordas / Detecção")
        self._painel_roi = PainelImagem("2. ROI Recortado")
        self._painel_clahe = PainelImagem("3. CLAHE / Contraste")
        self._painel_binaria = PainelImagem("4. Binarizada (Limiar)")

        for p in (self._painel_bordas, self._painel_roi, self._painel_clahe, self._painel_binaria):
            p.setMinimumHeight(140)

        layout_grid.addWidget(self._painel_bordas, 0, 0)
        layout_grid.addWidget(self._painel_roi, 0, 1)
        layout_grid.addWidget(self._painel_clahe, 1, 0)
        layout_grid.addWidget(self._painel_binaria, 1, 1)
        layout_direito.addLayout(layout_grid, stretch=2)

        # 2.3 Caracteres Segmentados
        quadro_caracteres = QFrame()
        quadro_caracteres.setObjectName("card_caracteres")
        layout_caract_bloco = QVBoxLayout(quadro_caracteres)
        layout_caract_bloco.setContentsMargins(12, 10, 12, 10)
        layout_caract_bloco.setSpacing(8)

        lbl_caract_tit = QLabel("CARACTERES SEGMENTADOS")
        lbl_caract_tit.setObjectName("card_secao_titulo")
        layout_caract_bloco.addWidget(lbl_caract_tit)

        self._layout_caracteres = QHBoxLayout()
        self._layout_caracteres.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._layout_caracteres.setSpacing(8)
        self._lbl_sem_caracteres = QLabel("(Nenhum caractere segmentado)")
        self._lbl_sem_caracteres.setStyleSheet("color: #94A3B8; font-style: italic;")
        self._layout_caracteres.addWidget(self._lbl_sem_caracteres)
        layout_caract_bloco.addLayout(self._layout_caracteres)
        layout_direito.addWidget(quadro_caracteres)

        # 2.4 Cartão de Resultado e OCR
        quadro_resultado = QFrame()
        quadro_resultado.setObjectName("card_resultado")
        layout_resultado = QVBoxLayout(quadro_resultado)
        layout_resultado.setContentsMargins(14, 12, 14, 12)
        layout_resultado.setSpacing(10)

        linha_placa = QHBoxLayout()
        self._txt_resultado = QLineEdit()
        self._txt_resultado.setObjectName("resultado_texto")
        self._txt_resultado.setReadOnly(True)
        self._txt_resultado.setPlaceholderText("AGUARDANDO LEITURA")
        linha_placa.addWidget(self._txt_resultado, stretch=2)

        self._badge_formato = QLabel("AGUARDANDO")
        self._badge_formato.setObjectName("badge_livre")
        self._badge_formato.setAlignment(Qt.AlignmentFlag.AlignCenter)
        linha_placa.addWidget(self._badge_formato)
        layout_resultado.addLayout(linha_placa)

        linha_detalhes = QHBoxLayout()
        self._lbl_bruto = QLabel("Texto bruto OCR: -")
        self._lbl_bruto.setObjectName("lbl_bruto_texto")
        linha_detalhes.addWidget(self._lbl_bruto)

        self._lbl_metricas_tempo = QLabel("")
        self._lbl_metricas_tempo.setObjectName("lbl_metricas_tempo")
        linha_detalhes.addWidget(self._lbl_metricas_tempo, alignment=Qt.AlignmentFlag.AlignRight)
        layout_resultado.addLayout(linha_detalhes)

        layout_direito.addWidget(quadro_resultado)

        scroll_direito.setWidget(widget_direito)
        splitter.addWidget(scroll_direito)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    def _ao_carregar_imagem(self, caminho: str) -> None:
        try:
            self._imagem_atual = ler_imagem(caminho)
            self._painel_original.definir_imagem(self._imagem_atual)
            self._limpar_etapas()
            self.statusBar().showMessage(f"Imagem carregada: {Path(caminho).name}")
            self._executar_pipeline()
        except Exception as e:
            QMessageBox.critical(self, "Erro ao Carregar", f"Falha ao carregar a imagem:\n{e}")

    def _ao_alterar_parametros(self) -> None:
        if self._imagem_atual is not None:
            self._timer_debounce.start()

    def _executar_pipeline(self) -> None:
        if self._imagem_atual is None:
            return

        # Cancela execução anterior se ainda estiver rodando
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self._controles.definir_estado_processando(True)
        self.statusBar().showMessage("Processando imagem...")

        parametros = self._controles.obter_parametros()
        self._worker = PipelineWorker(self._imagem_atual, parametros, self._funcao_pipeline, self)
        self._worker.resultado_pronto.connect(self._ao_receber_resultado)
        self._worker.erro_ocorrido.connect(self._ao_ocorrer_erro)
        self._worker.start()

    def _ao_receber_resultado(self, resultado: ResultadoPipeline) -> None:
        self._controles.definir_estado_processando(False)
        self._ultimo_resultado = resultado

        # 1. Imagem Principal
        if resultado.anotada is not None:
            self._painel_original.definir_imagem(resultado.anotada)
        else:
            self._painel_original.definir_imagem(resultado.original)

        # 2. Etapas Intermediárias
        self._painel_bordas.definir_imagem(resultado.bordas)
        self._painel_roi.definir_imagem(resultado.roi)
        self._painel_clahe.definir_imagem(resultado.roi_clahe)
        self._painel_binaria.definir_imagem(resultado.binarizada)

        # 3. Caracteres segmentados
        self._atualizar_caracteres_segmentados(resultado.caracteres)

        # 4. Resultado do Texto e Formato
        texto_exibicao = resultado.texto_final or resultado.texto_bruto or "SEM LEITURA"
        self._txt_resultado.setText(texto_exibicao)
        self._lbl_bruto.setText(f"Texto bruto OCR: {resultado.texto_bruto or '(vazio)'}")
        self._atualizar_badge(resultado.formato)

        # 5. Tempos e Status
        self._controles.habilitar_exportacao(True)
        self._atualizar_status_tempos(resultado)

    def _ao_ocorrer_erro(self, msg_erro: str) -> None:
        self._controles.definir_estado_processando(False)
        self.statusBar().showMessage("Erro durante o processamento.")
        QMessageBox.warning(self, "Aviso de Processamento", msg_erro)

    def _limpar_etapas(self) -> None:
        self._painel_bordas.limpar()
        self._painel_roi.limpar()
        self._painel_clahe.limpar()
        self._painel_binaria.limpar()
        self._atualizar_caracteres_segmentados([])
        self._txt_resultado.clear()
        self._lbl_bruto.setText("Texto bruto OCR: -")
        self._atualizar_badge("aguardando")

    def _atualizar_caracteres_segmentados(self, lista_caracteres: list[np.ndarray]) -> None:
        # Remove widgets anteriores
        while self._layout_caracteres.count():
            item = self._layout_caracteres.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not lista_caracteres:
            lbl = QLabel("(Nenhum caractere segmentado)")
            lbl.setStyleSheet("color: #94A3B8; font-style: italic;")
            self._layout_caracteres.addWidget(lbl)
            return

        for idx, char_img in enumerate(lista_caracteres):
            lbl_char = QLabel()
            pix = numpy_para_pixmap(char_img)
            lbl_char.setPixmap(
                pix.scaled(
                    34,
                    50,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            lbl_char.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_char.setObjectName("tile_caractere")
            self._layout_caracteres.addWidget(lbl_char)

    def _atualizar_badge(self, formato: str) -> None:
        formato = formato.lower()
        if formato == "mercosul":
            self._badge_formato.setText("✔ MERCOSUL")
            self._badge_formato.setObjectName("badge_mercosul")
        elif formato == "antigo":
            self._badge_formato.setText("✔ PADRÃO CINZA")
            self._badge_formato.setObjectName("badge_antigo")
        elif formato == "livre":
            self._badge_formato.setText("MODO LIVRE")
            self._badge_formato.setObjectName("badge_livre")
        elif formato == "invalido":
            self._badge_formato.setText("✘ INVÁLIDO")
            self._badge_formato.setObjectName("badge_invalido")
        else:
            self._badge_formato.setText("AGUARDANDO")
            self._badge_formato.setObjectName("badge_livre")

        # Força o recarregamento do estilo do badge
        self._badge_formato.style().unpolish(self._badge_formato)
        self._badge_formato.style().polish(self._badge_formato)

    def _atualizar_status_tempos(self, res: ResultadoPipeline) -> None:
        tempos = res.tempos_ms
        partes = []
        if "deteccao" in tempos:
            partes.append(f"Detecção: {tempos['deteccao']:.1f}ms")
        if "preproc" in tempos:
            partes.append(f"Pré-proc: {tempos['preproc']:.1f}ms")
        if "ocr" in tempos:
            partes.append(f"OCR: {tempos['ocr']:.1f}ms")
        if "total" in tempos:
            partes.append(f"Total: {tempos['total']:.1f}ms")

        msg_tempo = " · ".join(partes)
        self._lbl_metricas_tempo.setText(msg_tempo)

        if res.erro:
            self.statusBar().showMessage(f"Atenção: {res.erro} | {msg_tempo}")
        else:
            self.statusBar().showMessage(f"Processamento concluído. {msg_tempo}")

    def _ao_exportar(self) -> None:
        if self._ultimo_resultado is None:
            return

        pasta = QFileDialog.getExistingDirectory(self, "Selecione a Pasta para Exportar Resultados")
        if not pasta:
            return

        caminho_pasta = Path(pasta)
        res = self._ultimo_resultado

        try:
            if res.anotada is not None:
                salvar_imagem(caminho_pasta / "01_deteccao.png", res.anotada)
            if res.bordas is not None:
                salvar_imagem(caminho_pasta / "02_bordas.png", res.bordas)
            if res.roi is not None:
                salvar_imagem(caminho_pasta / "03_roi.png", res.roi)
            if res.roi_clahe is not None:
                salvar_imagem(caminho_pasta / "04_clahe.png", res.roi_clahe)
            if res.binarizada is not None:
                salvar_imagem(caminho_pasta / "05_binarizada.png", res.binarizada)

            for idx, c_img in enumerate(res.caracteres):
                salvar_imagem(caminho_pasta / f"caractere_{idx + 1:02d}.png", c_img)

            txt_path = caminho_pasta / "resultado.txt"
            conteudo = (
                f"Texto Final: {res.texto_final}\n"
                f"Texto Bruto: {res.texto_bruto}\n"
                f"Formato: {res.formato}\n"
                f"Substituições: {res.substituicoes}\n"
                f"Tempos (ms): {res.tempos_ms}\n"
            )
            txt_path.write_text(conteudo, encoding="utf-8")

            QMessageBox.information(
                self, "Exportação Concluída", f"Resultados salvos com sucesso em:\n{caminho_pasta}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Erro na Exportação", f"Falha ao exportar resultados:\n{e}")
