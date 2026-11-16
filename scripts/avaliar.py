"""Script oficial de benchmark e avaliação quantitativa do ALPR.

Avalia o dataset data/teste (120 amostras):
1. Benchmark de OCR Isolado (recortes/):
   - FastPlateOCR (Rede Neural ONNX)
   - Tesseract OCR (PSM 7)
2. Benchmark de Detecção (imagens/):
   - Detector YOLOv8 ONNX (IoU >= 0.5, Precisão, Tempo)
3. Benchmark Ponta a Ponta:
   - YOLO + FastPlateOCR
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.detection.yolo_onnx import DetectorYolo
from src.ocr import reconhecer_com_fastplate
from src.postprocess import corrigir_por_posicao


def levenshtein(s1: str, s2: str) -> int:
    """Calcula a distância de Levenshtein entre duas strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calcular_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Calcula a interseção sobre união (IoU) entre duas caixas (x1, y1, x2, y2)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    denom = float(area_a + area_b - inter_area)
    return inter_area / denom if denom > 0 else 0.0


def executar_benchmark() -> dict[str, Any]:
    """Executa a suíte completa de benchmark sobre data/teste."""
    gabarito_path = Path("data/teste/gabarito.csv")
    recortes_dir = Path("data/teste/recortes")
    imagens_dir = Path("data/teste/imagens")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(gabarito_path, encoding="utf-8") as f:
        gabarito = list(csv.DictReader(f))

    total = len(gabarito)
    print("===========================================================", flush=True)
    print(f"   BENCHMARK QUANTITATIVO ALPR - {total} AMOSTRAS DE TESTE", flush=True)
    print("===========================================================", flush=True)

    # -------------------------------------------------------------
    # 1. OCR Isolado: FastPlateOCR
    # -------------------------------------------------------------
    print("\n[1/3] Avaliando FastPlateOCR (ONNX) em recortes/...", flush=True)
    fp_exatos = 0
    fp_cers: list[float] = []
    fp_tempos: list[float] = []
    fp_validos = 0

    # Aquecimento do modelo
    dummy = cv2.imread(str(recortes_dir / gabarito[0]["arquivo"]))
    reconhecer_com_fastplate(dummy)

    for i, row in enumerate(gabarito, 1):
        crop = cv2.imread(str(recortes_dir / row["arquivo"]))
        esperada = row["placa_limpa"]

        t0 = time.perf_counter()
        txt, conf, _ = reconhecer_com_fastplate(crop)
        t_ms = (time.perf_counter() - t0) * 1000.0
        fp_tempos.append(t_ms)

        pred = txt.replace("-", "").strip()
        pos = corrigir_por_posicao(pred)
        pred_final = pos.texto_final.replace("-", "").strip()

        if pred_final == esperada:
            fp_exatos += 1
        if pos.valido:
            fp_validos += 1

        cer = levenshtein(pred_final, esperada) / max(len(esperada), 1)
        fp_cers.append(cer)

        if i % 30 == 0 or i == total:
            print(f"   Progresso FastPlate: {i}/{total} ({i/total:.0%})", flush=True)

    # -------------------------------------------------------------
    # 2. Detecção YOLO ONNX (em imagens/)
    # -------------------------------------------------------------
    print("\n[2/2] Avaliando Detecção YOLO ONNX em imagens/...", flush=True)
    detector = DetectorYolo(caminho_modelo=Path("models/placas.onnx"))
    yolo_iou50 = 0
    yolo_iou75 = 0
    yolo_ious: list[float] = []
    yolo_tempos: list[float] = []
    ponta_ponta_exatos = 0

    for i, row in enumerate(gabarito, 1):
        img = cv2.imread(str(imagens_dir / row["arquivo"]))
        gt_box = tuple(map(int, row["bbox_pixels"].split(",")))
        esperada = row["placa_limpa"]

        t0 = time.perf_counter()
        caixas = detector.detectar(img, conf_min=0.25, iou_nms=0.45)
        t_ms = (time.perf_counter() - t0) * 1000.0
        yolo_tempos.append(t_ms)

        caixa = caixas[0] if caixas else None
        if caixa is not None:
            pred_box = (caixa.x1, caixa.y1, caixa.x2, caixa.y2)
            iou = calcular_iou(pred_box, gt_box)
            yolo_ious.append(iou)
            if iou >= 0.50:
                yolo_iou50 += 1
            if iou >= 0.75:
                yolo_iou75 += 1

            # Ponta a ponta: recorta caixa detectada e lê com FastPlate
            pad_x = int(round((caixa.x2 - caixa.x1) * 0.04))
            pad_y = int(round((caixa.y2 - caixa.y1) * 0.04))
            H, W = img.shape[:2]
            rx1 = max(0, caixa.x1 - pad_x)
            ry1 = max(0, caixa.y1 - pad_y)
            rx2 = min(W, caixa.x2 + pad_x)
            ry2 = min(H, caixa.y2 + pad_y)
            crop_det = img[ry1:ry2, rx1:rx2]
            if crop_det.size > 0:
                txt_e2e, _, _ = reconhecer_com_fastplate(crop_det)
                pos_e2e = corrigir_por_posicao(txt_e2e)
                if pos_e2e.texto_final.replace("-", "").strip() == esperada:
                    ponta_ponta_exatos += 1
        else:
            yolo_ious.append(0.0)

        if i % 30 == 0 or i == total:
            print(f"   Progresso YOLO: {i}/{total} ({i/total:.0%})", flush=True)

    # -------------------------------------------------------------
    # Consolidação dos Resultados
    # -------------------------------------------------------------
    metricas = {
        "fastplate_exato": fp_exatos,
        "fastplate_taxa_exato": fp_exatos / total,
        "fastplate_cer": float(np.mean(fp_cers)),
        "fastplate_acuracia_char": 1.0 - float(np.mean(fp_cers)),
        "fastplate_validos": fp_validos / total,
        "fastplate_tempo_medio_ms": float(np.mean(fp_tempos)),
        "yolo_iou50": yolo_iou50,
        "yolo_taxa_iou50": yolo_iou50 / total,
        "yolo_iou75": yolo_iou75,
        "yolo_taxa_iou75": yolo_iou75 / total,
        "yolo_iou_medio": float(np.mean(yolo_ious)),
        "yolo_tempo_medio_ms": float(np.mean(yolo_tempos)),
        "ponta_ponta_exatos": ponta_ponta_exatos,
        "ponta_ponta_taxa": ponta_ponta_exatos / total,
        "total_amostras": total,
    }

    # Salva resumo em results/metricas.csv
    csv_out = results_dir / "metricas.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metrica", "Valor"])
        for k, v in metricas.items():
            writer.writerow([k, f"{v:.4f}" if isinstance(v, float) else str(v)])

    print("\n" + "=" * 60, flush=True)
    print("                RESULTADOS DO BENCHMARK", flush=True)
    print("=" * 60, flush=True)
    print("\n--- 1. OCR Isolado FastPlateOCR (120 Recortes) ---")
    print(f"  • Acerto Exato:         {metricas['fastplate_exato']}/{total} ({metricas['fastplate_taxa_exato']:.1%})")
    print(f"  • Acurácia de Caractere: {metricas['fastplate_acuracia_char']:.2%}")
    print(f"  • CER Médio:            {metricas['fastplate_cer']:.2%}")
    print(f"  • Formato Válido:       {metricas['fastplate_validos']:.1%}")
    print(f"  • Tempo Médio:          {metricas['fastplate_tempo_medio_ms']:.1f} ms por imagem")

    print("\n--- 2. Detecção YOLO ONNX (120 Imagens) ---")
    print(f"  • Taxa de Detecção (IoU >= 0.50): {metricas['yolo_iou50']}/{total} ({metricas['yolo_taxa_iou50']:.1%})")
    print(f"  • Detecção Estrita (IoU >= 0.75): {metricas['yolo_iou75']}/{total} ({metricas['yolo_taxa_iou75']:.1%})")
    print(f"  • IoU Médio:                      {metricas['yolo_iou_medio']:.3f}")
    print(f"  • Tempo Médio de Inferência:      {metricas['yolo_tempo_medio_ms']:.1f} ms por imagem")

    print("\n--- 3. Avaliação Ponta a Ponta (YOLO + FastPlateOCR) ---")
    print(f"  • Acerto Exato Ponta a Ponta:     {metricas['ponta_ponta_exatos']}/{total} ({metricas['ponta_ponta_taxa']:.1%})")
    print(f"  • Tempo Total Médio (Det + OCR):  {metricas['yolo_tempo_medio_ms'] + metricas['fastplate_tempo_medio_ms']:.1f} ms")
    print("=" * 60 + "\n", flush=True)

    return metricas


if __name__ == "__main__":
    executar_benchmark()
