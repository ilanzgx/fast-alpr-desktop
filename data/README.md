# Diretório de Dados

Este diretório gerencia os conjuntos de imagens de veículos para ajuste, demonstração e avaliação do sistema de reconhecimento de placas.

## Estrutura

- `demo/`: Amostras autorizadas e anonimizadas utilizadas na demonstração interativa da aplicação.
- `ajuste/`: Imagens reservadas para calibração de parâmetros (CLAHE, limiares, filtros, PSM).
- `teste/`: Conjunto de teste oficial reservado para avaliação quantitativa cega (não versionado no Git).

## Cuidados com Privacidade e Ética

- Placas reais e rostos de pessoas devem ser anonimizados/desfocados em materiais públicos.
- Fotografias próprias devem ter autorização prévia e metadados EXIF removidos.
- Nenhuma imagem é transmitida para a rede: todo o processamento é estritamente local e offline.
