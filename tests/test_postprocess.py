"""Testes automatizados do módulo de pós-processamento e regras de formato."""

from src.postprocess import (
    corrigir_por_posicao,
    limpar_texto,
    validar_formato,
)


def test_limpeza_de_texto():
    assert limpar_texto("abc-1234") == "ABC1234"
    assert limpar_texto("  A B C 1 D 2 3 . ") == "ABC1D23"
    assert limpar_texto("51F-075.68") == "51F07568"
    assert limpar_texto("") == ""


def test_validacao_formato():
    assert validar_formato("ABC1234") == "antigo"
    assert validar_formato("ABC1D23") == "mercosul"
    assert validar_formato("AB1234") == "invalido"
    assert validar_formato("123ABCD") == "invalido"


def test_exemplos_documentacao():
    # ABC1O23: na posição 4, O é letra válida em Mercosul (0 trocas vs 1 no antigo)
    res1 = corrigir_por_posicao("ABC1O23")
    assert res1.texto_final == "ABC1O23"
    assert res1.formato == "mercosul"
    assert res1.substituicoes == 0

    # A8C1D23: '8' na posição 1 corrigido para 'B'
    res2 = corrigir_por_posicao("A8C1D23")
    assert res2.texto_final == "ABC1D23"
    assert res2.formato == "mercosul"
    assert res2.substituicoes == 1

    # ABC12S4: 'S' na posição 5 corrigido para '5'
    res3 = corrigir_por_posicao("ABC12S4")
    assert res3.texto_final == "ABC1254"
    assert res3.formato == "antigo"
    assert res3.substituicoes == 1

    # AB1234: comprimento inválido (6 chars) não deve ser alterado
    res4 = corrigir_por_posicao("AB1234")
    assert res4.texto_final == "AB1234"
    assert res4.formato == "invalido"
    assert res4.valido is False


def test_modo_livre():
    # Placa internacional (ex: vietnamita ou americana)
    res = corrigir_por_posicao("51F07568", modo_livre=True)
    assert res.texto_final == "51F07568"
    assert res.formato == "livre"
    assert res.valido is True


def test_idempotencia():
    entrada = "A8C1D23"
    primeira = corrigir_por_posicao(entrada)
    segunda = corrigir_por_posicao(primeira.texto_final)
    assert primeira.texto_final == segunda.texto_final
    assert primeira.formato == segunda.formato


def test_extracao_janela_com_ruidos_de_borda():
    # Ruído de borda/QR code no início e final da placa
    res1 = corrigir_por_posicao("FPOX4G215")
    assert res1.texto_final == "POX4G21"
    assert res1.formato == "mercosul"
    assert res1.valido is True

    # Ruído com sufixo ou prefixo (ex: "BR" no início)
    res2 = corrigir_por_posicao("BRABC1D23")
    assert res2.texto_final == "ABC1D23"
    assert res2.formato == "mercosul"
    assert res2.valido is True

    # Placa antiga com ruído de moldura
    res3 = corrigir_por_posicao("1ABC1234F")
    assert res3.texto_final == "ABC1234"
    assert res3.formato == "antigo"
    assert res3.valido is True

