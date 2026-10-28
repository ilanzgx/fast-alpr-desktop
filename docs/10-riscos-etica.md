# 10. Riscos, limitações e cuidados com os dados

## 1. Riscos técnicos

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Dataset internacional representa mal as placas brasileiras | Alta | Alto | Complementar com fotos próprias e dataset brasileiro; testar só em placas brasileiras |
| OCR instável conforme recorte e iluminação | Alta | Alto | Multi-tentativa, ajuste de parâmetros pela interface, correção por posição |
| Saída do ONNX diferente do esperado | Média | Médio | Detectar o layout automaticamente; testar no Colab antes de levar ao desktop |
| Método clássico falha em fotos difíceis | Alta | Baixo | Esperado; vira material de análise e comparação com o YOLO |
| Interface travar durante o processamento | Média | Médio | Worker em thread separada e *debounce* nos sliders |
| Tesseract ausente em outras máquinas | Média | Alto | Instruções no README e variável `TESSERACT_CMD` |
| Dependências pesadas (PyTorch) no ambiente local | Média | Médio | Treino só no Colab; `easyocr` como extra opcional |
| Conjunto de teste contaminado durante o ajuste | Média | Alto | Pastas separadas desde o início; lista de nomes registrada |
| Gabarito de texto incompleto ou com erros | Média | Alto | Dupla conferência das transcrições |
| Prazo curto para avaliar | Média | Alto | Script de avaliação em lote pronto cedo; começar o gabarito na Fase 3 |

## 2. Limitações conhecidas do protótipo

- Reconhece uma imagem por vez, sem vídeo ou rastreamento.
- Desempenho depende de resolução, ângulo, iluminação e estado da placa.
- Validação de formato pensada para placas brasileiras; outras exigem o modo livre.
- Placas de duas linhas (motos) e veículos de categorias especiais podem não ser tratados corretamente.
- Resultados refletem o conjunto de teste usado e **não** generalizam para qualquer situação.
- O sistema é um **protótipo didático**, sem uso para fiscalização, identificação de pessoas ou qualquer decisão automatizada.

## 3. Cuidados com os dados

| Tema | Regra |
|------|-------|
| Origem | Usar apenas imagens próprias ou de fontes que permitam uso acadêmico, com origem e licença registradas em `data/README.md` |
| Autorização | Fotos de veículos de terceiros só com autorização do proprietário |
| Placas reais | Borradas em figuras do relatório, apresentações e repositório público; preferir imagens demonstrativas |
| Pessoas | Evitar rostos nas fotos; se aparecerem, desfocar nas figuras públicas |
| Teste reservado | `data/teste/` não é versionado nem publicado |
| Segredos | Nenhuma chave de API, senha ou dado pessoal no repositório; usar `.env` ou *Secrets* do Colab |
| Offline | O aplicativo não envia imagens para nenhum servidor |
| Metadados | Remover EXIF (localização, data) das fotos próprias antes de compartilhar |

## 4. Licenças e atribuição

- Registrar em `data/README.md`: nome do dataset, autor, link, licença, data de acesso e alterações feitas.
- Citar bibliotecas e modelos usados (OpenCV, onnxruntime, Tesseract, YOLO/Ultralytics, PySide6). Verificar as licenças, especialmente a do **Ultralytics YOLO** (AGPL-3.0) e a do **PySide6** (LGPL), caso haja distribuição do aplicativo.
- No relatório, indicar claramente o que é código do grupo e o que é modelo ou biblioteca externa.

## 5. O que registrar no relatório sobre limitações

1. Condições de captura do conjunto de teste e quantidade de imagens.
2. Taxa de acerto **com o intervalo de incerteza** quando o conjunto for pequeno.
3. Exemplos de sucesso **e** de falha, com a causa provável.
4. Diferença entre o ajuste e o teste, e como foi evitado o vazamento.
5. Aviso de que o sistema não deve ser usado para fiscalização ou decisões reais.

## 6. Plano de contingência

| Se... | Então... |
|-------|----------|
| O fine-tuning não melhorar a detecção | Manter o modelo existente e relatar a tentativa com as métricas |
| O OCR ficar abaixo do esperado | Reportar as causas, mostrar o ganho do pré e pós-processamento e propor melhorias |
| O tempo apertar | Priorizar: pipeline completo funcionando > avaliação com poucas imagens bem feitas > extras (EasyOCR, multi-tentativa, empacotamento) |
| O Tesseract não instalar numa máquina de demonstração | Levar notebook ou imagens de resultados prontos como reserva |
