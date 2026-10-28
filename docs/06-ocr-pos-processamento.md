# 06. OCR e pós-processamento

## 1. Motores de OCR

| Motor | Papel | Observações |
|-------|-------|-------------|
| **Tesseract** (`pytesseract`) | Principal | Leve, offline, exige o programa instalado no sistema |
| **EasyOCR** | Alternativa opcional para comparação | Puxa PyTorch; instalar só como dependência opcional |

Nos testes iniciais, o Tesseract acertou onde o EasyOCR divergiu em uma das placas, mas três exemplos não permitem concluir qual é melhor. A comparação sistemática está em `07-dados-avaliacao.md`.

## 2. Configuração do Tesseract

```python
import pytesseract

CONFIG = "--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
texto = pytesseract.image_to_string(imagem, config=CONFIG.format(psm=7))
```

| PSM | Significado | Uso |
|-----|-------------|-----|
| 7 | Uma única linha de texto | **Padrão** para placas |
| 13 | Linha crua, sem heurísticas de página | Funcionou quando o 7 falhou em um dos testes |
| 8 | Uma única palavra | Alternativa |
| 6 | Bloco uniforme de texto | Placas de duas linhas (motos) |

- **Whitelist** restringe a saída a letras maiúsculas e dígitos.
- Para obter confiança por caractere, usar `image_to_data(..., output_type=Output.DICT)`.
- No Windows, definir `TESSERACT_CMD` (ver `08-setup-repositorio.md`).

## 3. Estratégia de leitura (multi-tentativa)

Como o desempenho do OCR varia com recorte e tratamento, o módulo tenta **variantes** e escolhe a melhor:

```
Variantes de entrada:   ROI cinza · ROI com CLAHE · ROI binarizada (Otsu) · ROI binarizada (adaptativa)
Variantes de PSM:       7 · 13
                              ↓
                Para cada combinação: ler → pós-processar
                              ↓
Escolha:  1º formato válido  ·  2º maior confiança média  ·  3º maior nº de caracteres plausíveis
```

- O modo de uma tentativa só (rápido) continua disponível: segue o PSM escolhido na interface.
- O modo multi-tentativa fica como opção ("Leitura robusta") e deve ser comparado ao modo simples na avaliação.
- O tempo total sobe com o número de combinações, então o relatório deve registrar o custo.

## 4. Formatos de placa brasileiros

| Tipo | Padrão | Expressão regular | Exemplo |
|------|--------|-------------------|---------|
| Antigo | `AAA-0000` | `^[A-Z]{3}[0-9]{4}$` | `ABC1234` |
| Mercosul | `AAA0A00` | `^[A-Z]{3}[0-9][A-Z][0-9]{2}$` | `ABC1D23` |

O hífen é removido antes da validação.

## 5. Correção por posição

Cada posição só admite letras ou dígitos, então confusões típicas do OCR podem ser corrigidas pelo contexto.

| Posição (0 a 6) | Esperado |
|-----------------|----------|
| 0, 1, 2 | Letra |
| 3 | Dígito |
| 4 | Dígito (antigo) **ou** letra (Mercosul) |
| 5, 6 | Dígito |

**Mapas de confusão**

| Letra → dígito | Dígito → letra |
|----------------|----------------|
| O→0, Q→0, D→0 | 0→O |
| I→1, L→1 | 1→I |
| Z→2 | 2→Z |
| S→5 | 5→S |
| G→6 | 6→G |
| B→8 | 8→B |
| T→7 | 7→T |

**Algoritmo**

1. Limpar: maiúsculas, remover espaços, hífens e caracteres fora de `A-Z0-9`.
2. Se o comprimento for 7, gerar duas hipóteses para a posição 4: **antigo** (dígito) e **Mercosul** (letra).
3. Para cada hipótese, aplicar o mapa de confusão em cada posição e contar as **substituições**.
4. Escolher a hipótese com **menos substituições**. Em empate, preferir a que o OCR já lia como válida; se ainda empatar, Mercosul (mais comum em placas recentes).
5. Se o comprimento for diferente de 7, não corrigir: devolver o texto limpo com formato `invalido`.

**Exemplos**

| Bruto | Final | Formato | Observação |
|-------|-------|---------|------------|
| `ABC1O23` | `ABC1O23` | Mercosul | Posição 4: Mercosul exige 0 trocas; antigo exigiria 1 (O→0) |
| `A8C1D23` | `ABC1D23` | Mercosul | `8` na posição 1 → `B` |
| `ABC12S4` | `ABC1254` | Antigo | `S` na posição 5 → `5` |
| `AB1234` | `AB1234` | Inválido | Comprimento errado |

> Estes exemplos devem virar **casos de teste**, inclusive os empates: na posição 4, a hipótese com menos substituições vence.

## 6. Modo "livre" para placas não brasileiras

O dataset de treino é internacional, e os exemplos do notebook incluem placas de outros países (por exemplo, `51F-075.68` e `AFR 2011`). Para evitar descartar leituras corretas:

- A validação brasileira é uma **opção** (marcada por padrão).
- Com ela desligada, o sistema só limpa o texto (maiúsculas, caracteres válidos) e mostra o formato como "livre".
- A avaliação principal usa o **modo brasileiro**, com placas brasileiras no conjunto de teste.

## 7. Testes automatizados (`tests/test_postprocess.py`)

- Limpeza de texto (espaços, hífen, minúsculas).
- Validação dos dois formatos.
- Correção por posição, incluindo os empates descritos acima.
- Comprimentos inválidos não são "corrigidos".
- Idempotência: aplicar duas vezes dá o mesmo resultado.

## 8. Saída do módulo

```python
@dataclass
class LeituraOCR:
    texto_bruto: str
    confianca: float | None
    psm: int
    variante: str            # "cinza" | "clahe" | "otsu" | "adaptativa"

@dataclass
class TextoFinal:
    texto: str
    formato: str             # "antigo" | "mercosul" | "invalido" | "livre"
    substituicoes: int
```
