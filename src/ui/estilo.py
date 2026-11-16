"""Estilo visual moderno e refinado (Clean Academic Theme) em QSS para PySide6."""

TEMA_CLARO = """
/* =========================================================================
   1. RESET E JANELA PRINCIPAL
   ========================================================================= */
QMainWindow {
    background-color: #F8FAFC;
}

QWidget {
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "SF Pro Text", Helvetica, Arial, sans-serif;
    font-size: 13px;
    color: #0F172A;
}

QScrollArea, QScrollArea::viewport {
    border: none;
    background-color: #F8FAFC;
}

QWidget#painel_controles_conteudo, QWidget#painel_direito_conteudo {
    background-color: #F8FAFC;
}

/* =========================================================================
   2. SCROLLBARS MODERNAS E ULTRAFINAS
   ========================================================================= */
QScrollBar:vertical {
    background-color: transparent;
    width: 8px;
    margin: 4px 2px 4px 2px;
}

QScrollBar::handle:vertical {
    background-color: #CBD5E1;
    border-radius: 4px;
    min-height: 32px;
}

QScrollBar::handle:vertical:hover {
    background-color: #94A3B8;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
    border: none;
    height: 0px;
}

QScrollBar:horizontal {
    background-color: transparent;
    height: 8px;
    margin: 2px 4px 2px 4px;
}

QScrollBar::handle:horizontal {
    background-color: #CBD5E1;
    border-radius: 4px;
    min-width: 32px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #94A3B8;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: none;
    border: none;
    width: 0px;
}

/* =========================================================================
   3. GRUPOS DE CONTROLES (CARDS DO PAINEL LATERAL)
   ========================================================================= */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 2px 8px;
    color: #475569;
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* =========================================================================
   4. BOTÕES
   ========================================================================= */
QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 7px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 12px;
    color: #334155;
}

QPushButton:hover {
    background-color: #F8FAFC;
    border-color: #94A3B8;
    color: #0F172A;
}

QPushButton:pressed {
    background-color: #F1F5F9;
}

QPushButton:disabled {
    background-color: #F8FAFC;
    border-color: #E2E8F0;
    color: #94A3B8;
}

/* Botão Primário de Ação */
QPushButton#btn_primario {
    background-color: #2563EB;
    border: 1px solid #1D4ED8;
    border-radius: 7px;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 13px;
    padding: 10px 18px;
}

QPushButton#btn_primario:hover {
    background-color: #1D4ED8;
    border-color: #1E40AF;
}

QPushButton#btn_primario:pressed {
    background-color: #1E40AF;
}

QPushButton#btn_primario:disabled {
    background-color: #BFDBFE;
    border-color: #DBEAFE;
    color: #FFFFFF;
}

/* =========================================================================
   5. ENTRADAS DE DADOS: COMBOBOX E SPINBOX
   ========================================================================= */
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 22px;
    font-size: 12px;
    font-weight: 500;
    color: #1E293B;
}

QComboBox:hover {
    border-color: #94A3B8;
}

QComboBox:focus {
    border: 1.5px solid #2563EB;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: none;
}

QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #64748B;
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 4px;
    color: #1E293B;
    selection-background-color: #EFF6FF;
    selection-color: #1D4ED8;
    outline: none;
}

QSpinBox, QDoubleSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 3px 6px;
    min-height: 20px;
    font-size: 12px;
    font-weight: 600;
    color: #1E293B;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #94A3B8;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1.5px solid #2563EB;
}

/* =========================================================================
   6. CONTROLES DESLIZANTES (SLIDERS)
   ========================================================================= */
QSlider:horizontal {
    min-height: 24px;
    height: 24px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #E2E8F0;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #2563EB;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #2563EB;
    width: 16px;
    height: 16px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #EFF6FF;
    border-color: #1D4ED8;
}

QSlider::handle:horizontal:pressed {
    background: #2563EB;
    border-color: #1E40AF;
}

/* =========================================================================
   7. CAIXAS DE SELEÇÃO (CHECKBOXES)
   ========================================================================= */
QCheckBox {
    spacing: 8px;
    color: #334155;
    font-size: 12px;
    font-weight: 500;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1.5px solid #CBD5E1;
    border-radius: 4px;
    background-color: #FFFFFF;
}

QCheckBox::indicator:hover {
    border-color: #94A3B8;
}

QCheckBox::indicator:checked {
    background-color: #2563EB;
    border-color: #2563EB;
}

/* =========================================================================
   8. BARRA DE STATUS E SPLITTER
   ========================================================================= */
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E2E8F0;
    color: #64748B;
    font-size: 12px;
    padding: 6px 12px;
}

QSplitter {
    background-color: #F8FAFC;
}

QSplitter::handle {
    background-color: #E2E8F0;
    width: 2px;
}

QSplitter::handle:hover {
    background-color: #94A3B8;
}

/* =========================================================================
   9. PAINÉIS DE IMAGEM (VIEWPORTS)
   ========================================================================= */
QFrame#painel_imagem_card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

QLabel#painel_imagem_titulo {
    font-size: 11px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

QLabel#painel_imagem_viewport_vazio {
    background-color: #F8FAFC;
    border: 1.5px dashed #CBD5E1;
    border-radius: 7px;
    color: #94A3B8;
    font-size: 12px;
}

QLabel#painel_imagem_viewport_ativo {
    background-color: #0F172A;
    border: 1px solid #CBD5E1;
    border-radius: 7px;
}

/* =========================================================================
   10. CARTÕES DE CARACTERES E RESULTADO FINAL
   ========================================================================= */
QFrame#card_caracteres, QFrame#card_resultado {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

QLabel#card_secao_titulo {
    font-size: 11px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

QLabel#tile_caractere {
    background-color: #0F172A;
    border: 1.5px solid #1E293B;
    border-radius: 6px;
    padding: 3px 5px;
}

QLineEdit#resultado_texto {
    font-family: "Consolas", "Cascadia Code", "JetBrains Mono", "Courier New", monospace;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 6px;
    background-color: #F8FAFC;
    border: 2px solid #E2E8F0;
    border-radius: 8px;
    padding: 8px 16px;
    color: #0F172A;
    min-height: 40px;
}

QLineEdit#resultado_texto:focus {
    border-color: #2563EB;
    background-color: #FFFFFF;
}

/* =========================================================================
   11. BADGES DE FORMATO DE PLACA
   ========================================================================= */
QLabel#badge_mercosul {
    background-color: #ECFDF5;
    color: #047857;
    border: 1.5px solid #6EE7B7;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
    min-height: 28px;
}

QLabel#badge_antigo {
    background-color: #EFF6FF;
    color: #1D4ED8;
    border: 1.5px solid #93C5FD;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
    min-height: 28px;
}

QLabel#badge_invalido {
    background-color: #FFF1F2;
    color: #BE123C;
    border: 1.5px solid #FDA4AF;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
    min-height: 28px;
}

QLabel#badge_livre {
    background-color: #F8FAFC;
    color: #475569;
    border: 1.5px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
    min-height: 28px;
}

QLabel#lbl_bruto_texto {
    color: #64748B;
    font-size: 12px;
    font-weight: 500;
}

QLabel#lbl_metricas_tempo {
    color: #64748B;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    background-color: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 4px;
}
"""
