# 02-RECORTE: Recorte Contextual, Retificação Geométrica e Zoom-Out

As etapas E2 e E3 do pipeline extraem a região de interesse (ROI) da placa a partir da caixa delimitadora e aplicam correções geométricas para alinhar caracteres inclinados, eliminar distorções de perspectiva e preservar integralmente o texto nas bordas. A lógica está implementada em [src/pipeline.py](src/pipeline.py) e [src/preprocessing.py](src/preprocessing.py).

---

## 1. Recorte com Contexto Expandido

Para evitar o corte de caracteres extremos e permitir a retificação sem perda de pixels úteis:

1. **Recorte com Offset (`recortar_roi_com_offset`)**:
   - Extrai a região da placa com margem contextual de pelo menos 20% ao redor da caixa detectada (`margem_ctx = max(params.margem_roi, 0.20)`).
   - Registra o deslocamento `(ox, oy)` relativo à imagem original completa, permitindo mapear coordenadas locais de volta ao sistema de coordenadas global.
2. **Normalização de Escala (`ampliar_roi_com_escala`)**:
   - Padroniza a altura do bloco contextual para 160 pixels sempre que o ROI for pequeno, preservando a razão de escala (`escala_ctx`) para reprojeções exatas na imagem de resolução plena.

---

## 2. Detecção de Quadriláteros e Retificação em Perspectiva

Quando o veículo é fotografado em ângulo diagonal, a placa sofre escorço projetivo tridimensional. O sistema analisa e corrige essa distorção:

1. **Busca de Vértices (`encontrar_vertices_placa_roi`)**:
   - Executa detecção Canny com fechamento morfológico retangular em múltiplos limiares para identificar o contorno externo da placa.
   - Aplica critérios de rejeição para evitar falsos positivos:
     - Descarta contornos excessivamente achatados (`aspecto > 3.65`), que ignoram a faixa superior.
     - Descarta paralelogramos com ângulos internos fora da faixa de 78° a 102° em placas que não apresentam escorço de profundidade.
2. **Detecção Pós-Rotação (`detectar_vertices_pos_rotacao`)**:
   - Caso a placa esteja fortemente comprimida em perspectiva (`largura / altura < 2.2`), desinclina previamente o recorte pelo ângulo dominante e busca novamente os quatro vértices externos.
3. **Verificação de Distorção (`verificar_distorcao_geometrica`)**:
   - Avalia as inclinações dos segmentos laterais e horizontais. Se a assimetria ou cisalhamento angular for superior a 2.5°, a correção por perspectiva é acionada.
4. **Transformação Homográfica com Zoom-Out (`corrigir_perspectiva`)**:
   - Aplica `cv2.getPerspectiveTransform` e `cv2.warpPerspective` buscando os pixels diretamente na **imagem original completa**.
   - Adiciona uma margem de segurança de 9% (`margem_saida=0.09`) nos quatro cantos da projeção de destino, garantindo que o zoom-out capture o entorno real da lataria sem gerar faixas pretas artificiais.
   - Quando o escorço deforma a largura da placa, o algoritmo restaura a proporção padrão das placas brasileiras (razão nominal de 3.08).

---

## 3. Retificação Angular 2D (Deskew com Zoom-Out)

Quando a placa apresenta apenas inclinação planar (sem distorção de perspectiva), o pipeline utiliza a função `corrigir_inclinacao`:

1. **Estimativa Angular Robusta (`estimar_angulo_inclinacao`)**:
   - Realça o contraste com CLAHE e calcula gradientes com Canny.
   - Aplica a Transformada Probabilística de Hough (`cv2.HoughLinesP`) restrita ao intervalo de -35° a +35°.
   - Exige linhas estruturais longas (comprimento mínimo de pelo menos 26% da largura da placa).
   - Aplica **ponderação quadrática por comprimento ($comprimento^2$)**: garante que bordas longas da chapa da placa dominem decisivamente sobre traços curtos de letras inclinadas (como `A`, `K`, `M`, `R`, `4`, `7`).
   - Requer que o pico dominante concentre pelo menos 25% do peso total para acionar rotação.
   - Filtra placas quadradas ou de motos (`aspecto < 1.6`), impedindo rotações espúrias em placas de duas linhas.
2. **Expansão de Canvas**:
   - A matriz de rotação é recalculada para novas dimensões ($W_{novo} = |H \sin \theta| + |W \cos \theta|$ e $H_{novo} = |H \cos \theta| + |W \sin \theta|$), impedindo o corte geométrico dos vértices da placa.
   - O preenchimento externo utiliza `cv2.BORDER_REPLICATE` para evitar bordas pretas que possam gerar blocos brancos na posterior binarização invertida.
3. **Proteção da Faixa Superior**:
   - Reprojeta as coordenadas da caixa interna para a imagem alinhada e impõe piso de 95% da altura para pequenos ângulos, limitando o aspecto máximo a 3.45 para não ceifar a faixa superior.
4. **Recorte com Zoom-Out Final (`_recortar_com_zoom_out`)**:
   - Adiciona margem de segurança de aproximadamente 8% na largura e 11% na altura (mínimo de 10 pixels no topo e na base), entregando uma folga visual clara ao redor de todos os caracteres.
   - Para desvios angulares insignificantes (inferiores a 0.8°), mantém o recorte regular respeitando `margem_roi`.

---

## 4. Padronização Final do ROI

Ao término do alinhamento geométrico, a imagem recortada passa pela função `ampliar_roi`, que normaliza a altura para 130 pixels via interpolação cúbica (`cv2.INTER_CUBIC`), entregando dimensões padronizadas para as etapas seguintes de pré-processamento e OCR.
