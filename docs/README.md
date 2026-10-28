# Documentação do Projeto: Reconhecimento de Placas Veiculares

Aplicação **desktop e offline** em Python que localiza placas em fotografias de veículos, prepara a região recortada e lê os caracteres, exibindo cada etapa do processamento.

**Stack:** PySide6 · OpenCV · onnxruntime · Tesseract · uv
**Treino do detector:** Google Colab (YOLO → exportação para ONNX)

## Índice

| # | Documento | Conteúdo |
|---|-----------|----------|
| 01 | [Visão geral](01-visao-geral.md) | Objetivos, escopo, requisitos e critérios de aceite |
| 02 | [Arquitetura](02-arquitetura.md) | Camadas, módulos, contratos de dados, concorrência e decisões |
| 03 | [Pipeline de processamento](03-pipeline.md) | Cada etapa, parâmetros e valores padrão |
| 04 | [Interface PySide6](04-interface.md) | Layout, widgets, estados e fluxo de interação |
| 05 | [Modelo YOLO / ONNX](05-modelo-onnx.md) | Treino no Colab, exportação e inferência local |
| 06 | [OCR e pós-processamento](06-ocr-pos-processamento.md) | Tesseract, regras de formato e correção de caracteres |
| 07 | [Dados e avaliação](07-dados-avaliacao.md) | Dataset, gabarito, métricas e experimentos |
| 08 | [Setup e repositório](08-setup-repositorio.md) | Instalação, estrutura de pastas e comandos |
| 09 | [Roadmap](09-roadmap.md) | Fases, checklists e backlog por módulo |
| 10 | [Riscos e cuidados](10-riscos-etica.md) | Riscos, limitações e tratamento dos dados |

## Ordem de leitura sugerida

1. **01** e **02** para entender o que será construído e como está organizado.
2. **03**, **05** e **06** para o núcleo técnico.
3. **04** para a interface.
4. **07** e **08** antes de começar os testes e a configuração do ambiente.
5. **09** e **10** para planejamento.
