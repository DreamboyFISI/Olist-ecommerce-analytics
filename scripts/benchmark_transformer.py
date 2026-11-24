"""
Benchmark opcional de un transformer preentrenado para sentimientos (PT-BR).
================================================================================
No forma parte del pipeline desplegable: requiere torch y transformers, que no
están en requirements-web. Su salida se guarda en
`artifacts/nlp/transformer_benchmark.json` y el módulo clásico la incorpora
como referencia externa si existe.

Ejemplo:
    pip install -r requirements-nlp.txt
    python scripts/benchmark_transformer.py \
        --model Eduardo211000/bertimbau-sentimento-b2w \
        --sample-per-class 500

El modelo por defecto es un BERTimbau fine-tuneado en reseñas de otro
e-commerce brasileño (B2W). No se entrenó con Olist, así que la evaluación es
zero-shot y no hay fuga de datos. Fine-tunear BERTimbau con Olist queda como
mejora futura (requiere GPU para ser práctico).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.sentiment import (  # noqa: E402
    ARTIFACTS_DIR,
    LABELS,
    LABEL_NAMES,
    holdout_metrics,
    load_reviews,
    make_splits,
    stratified_sample,
)


def _map_model_label(name: str) -> int | None:
    """Traduce la etiqueta del transformer a negativo/neutro/positivo."""
    text = str(name).strip().lower()
    if any(token in text for token in ("neg", "ruim", "bad")):
        return 0
    if any(token in text for token in ("neu", "mid", "ok")):
        return 1
    if any(token in text for token in ("pos", "bom", "good")):
        return 2
    match = re.search(r"([1-5])", text)
    if match:
        stars = int(match.group(1))
        if stars <= 2:
            return 0
        if stars == 3:
            return 1
        return 2
    return None


def _label_mapping(id2label: dict) -> tuple[dict[int, int], list[str]]:
    mapping: dict[int, int] = {}
    unknown: list[str] = []
    for raw_id, raw_label in id2label.items():
        target = _map_model_label(raw_label)
        if target is None:
            unknown.append(str(raw_label))
            continue
        mapping[int(raw_id)] = target
    if not mapping:
        raise ValueError(
            f"No se pudo interpretar id2label={id2label}. Ajusta _map_model_label."
        )
    return mapping, unknown


def _load_transformer(model_id: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return tokenizer, model, torch


def _predict(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    torch: Any,
    id_to_target: dict[int, int],
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (predicción, confianza, probabilidades de 3 clases)."""
    predictions: list[int] = []
    confidences: list[float] = []
    probabilities: list[list[float]] = []
    total = len(texts)
    started = time.time()

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            batch_proba = torch.softmax(logits, dim=-1).cpu().numpy()

        for row in batch_proba:
            mapped = np.zeros(len(LABELS), dtype=float)
            for raw_index, target in id_to_target.items():
                if raw_index < len(row):
                    mapped[target] += float(row[raw_index])
            mapped = mapped / max(mapped.sum(), 1e-12)
            label = int(np.argmax(mapped))
            predictions.append(label)
            confidences.append(float(mapped[label]))
            probabilities.append(mapped.tolist())

        if (start // batch_size) % 10 == 0:
            elapsed = time.time() - started
            done = min(start + len(batch), total)
            speed = done / elapsed if elapsed else 0.0
            print(f"  {done:,}/{total:,} reseñas ({speed:.1f}/s)", flush=True)

    return (
        np.asarray(predictions, dtype=int),
        np.asarray(confidences, dtype=float),
        np.asarray(probabilities, dtype=float),
    )


def run_benchmark(
    model_id: str,
    sample_per_class: int,
    output: Path,
    batch_size: int,
    max_length: int,
    device: str | None,
    data_dir: Path | None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Modelo: {model_id}")
    print(f"Dispositivo: {device}")
    print("Cargando y preparando reseñas...")
    reviews = load_reviews(data_dir)
    _, test = make_splits(reviews)
    sample = stratified_sample(test, n_per_class=sample_per_class)
    print(
        f"Muestra de evaluación: {len(sample):,} reseñas "
        f"({sample_per_class} por clase, test estratificado balanceado)"
    )

    config = AutoConfig.from_pretrained(model_id)
    id_to_target, unknown = _label_mapping(config.id2label)
    print(f"Etiquetas del modelo: {config.id2label}")
    print(f"Mapeo a Olist (0=neg, 1=neu, 2=pos): {id_to_target}")

    tokenizer, model, torch = _load_transformer(model_id, device)
    started = time.time()
    predicted, confidence, probabilities = _predict(
        sample["text_clean"].tolist(),
        tokenizer,
        model,
        torch,
        id_to_target,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    runtime = time.time() - started

    metrics = holdout_metrics(sample["label"].to_numpy(), predicted, probabilities)
    payload = {
        "model_id": model_id,
        "model_url": f"https://huggingface.co/{model_id}",
        "model_name": "BERTimbau fine-tuneado en reseñas B2W (referencia externa)",
        "task": "text-classification",
        "language": "Portugués brasileño (e-commerce)",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "runtime_sec": round(runtime, 2),
        "throughput_resenas_por_seg": round(len(sample) / runtime, 2) if runtime else None,
        "sample": {
            "n": int(len(sample)),
            "per_class": int(sample_per_class),
            "strategy": "Test estratificado balanceado (misma partición que el modelo clásico)",
            "note": (
                "Muestra balanceada para comparar macro-F1 entre clases. "
                "No representa la prevalencia natural del test."
            ),
        },
        "label_mapping": {
            "model_id2label": {str(key): value for key, value in config.id2label.items()},
            "olist_target": {str(key): value for key, value in id_to_target.items()},
            "unknown_labels": unknown,
        },
        "metrics": metrics,
        "notes": [
            "Evaluación zero-shot: el modelo no vio reseñas de Olist durante su entrenamiento.",
            "El modelo se entrenó con reseñas de B2W, otro e-commerce brasileño; el dominio es cercano, pero no idéntico.",
            "No se fine-tuneó BERTimbau con Olist para evitar fuga de datos y porque el costo en CPU es alto.",
            "El modelo no se despliega en la API: se usa solo como referencia comparativa.",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {output}")
    print(
        f"Macro-F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f} | "
        f"PR-AUC negativa: {metrics.get('negative_pr_auc', 0):.4f}"
    )
    print(
        "F1 por clase: "
        + ", ".join(
            f"{name}={metrics['per_class'][name]['f1']:.3f}" for name in LABEL_NAMES.values()
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark zero-shot de un transformer para sentimientos en portugués."
    )
    parser.add_argument(
        "--model",
        default="Eduardo211000/bertimbau-sentimento-b2w",
        help="Modelo de Hugging Face con etiquetas de sentimiento.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=500,
        help="Reseñas de test por clase (la muestra se balancea).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "transformer_benchmark.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default=None, help="cpu, cuda o vacío para autodetectar.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directorio con olist_order_reviews_dataset.csv.",
    )
    args = parser.parse_args()
    run_benchmark(
        model_id=args.model,
        sample_per_class=args.sample_per_class,
        output=args.output,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
