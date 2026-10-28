"""Estilo visual limpo acadêmico (Light Theme) em QSS para a aplicação PySide6."""

TEMA_CLARO = """
/* Reset e Janela Principal */
QMainWindow {
    background-color: #F8F9FA;
}

QWidget {
    font-family: "Segoe UI", "SF Pro Text", Helvetica, Arial, sans-serif;
    font-size: 13px;
    color: #212529;
}

/* Painel de Controles e Painéis Laterais */
QScrollArea {
    border: none;
    background-color: transparent;
}

/* Grupos de Controles (Cards) */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-top: 24px;
    padding: 12px 14px 14px 14px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 2px 8px;
    color: #1E293B;
    background-color: #F1F5F9;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Botões */
QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 500;
    color: #334155;
}

QPushButton:hover {
    background-color: #F8FAFC;
    border-color: #94A3B8;
}

QPushButton:pressed {
    background-color: #F1F5F9;
}

QPushButton:disabled {
    background-color: #F8F9FA;
    border-color: #E2E8F0;
    color: #94A3B8;
}

/* Botão de Ação Primária */
QPushButton#btn_primario {
    background-color: #2563EB;
    border: 1px solid #1D4ED8;
    color: #FFFFFF;
    font-weight: 600;
}

QPushButton#btn_primario:hover {
    background-color: #1D4ED8;
}

QPushButton#btn_primario:pressed {
    background-color: #1E40AF;
}

QPushButton#btn_primario:disabled {
    background-color: #93C5FD;
    border-color: #BFDBFE;
    color: #FFFFFF;
}

/* Controles de Formulário: Combobox, SpinBox */
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 5px 8px;
    min-height: 20px;
    color: #1E293B;
}

QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #3B82F6;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: none;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 4px;
    background: #E2E8F0;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #3B82F6;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #3B82F6;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #EFF6FF;
    border-color: #1D4ED8;
}

/* Checkboxes */
QCheckBox {
    spacing: 8px;
    color: #334155;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    background-color: #FFFFFF;
}

QCheckBox::indicator:checked {
    background-color: #2563EB;
    border-color: #1D4ED8;
}

/* Barra de Status */
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E2E8F0;
    color: #64748B;
    font-size: 12px;
    padding: 4px 8px;
}

/* Splitter */
QSplitter::handle {
    background-color: #E2E8F0;
    width: 2px;
}

/* Selo de formato e resultado */
QLineEdit#resultado_texto {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 4px;
    background-color: #F8FAFC;
    border: 2px solid #CBD5E1;
    border-radius: 8px;
    padding: 6px 14px;
    color: #0F172A;
}

QLabel#badge_mercosul {
    background-color: #DCFCE7;
    color: #15803D;
    border: 1px solid #86EFAC;
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: 700;
    font-size: 12px;
}

QLabel#badge_antigo {
    background-color: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #93C5FD;
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: 700;
    font-size: 12px;
}

QLabel#badge_invalido {
    background-color: #FEE2E2;
    color: #B91C1C;
    border: 1px solid #FCA5A5;
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: 700;
    font-size: 12px;
}

QLabel#badge_livre {
    background-color: #F1F5F9;
    color: #475569;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: 700;
    font-size: 12px;
}
"""
