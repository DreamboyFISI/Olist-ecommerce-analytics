"""Pruebas unitarias del módulo de análisis de sentimientos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.sentiment import (
    ARTIFACTS_DIR,
    LABEL_NAMES,
    SCHEMA_PATH,
    build_features,
    build_pipeline,
    clean_text,
    holdout_metrics,
    load_reviews,
    make_splits,
    normalize_for_dedup,
    score_to_label,
    stratified_sample,
    top_terms,
)


def test_clean_text_conserva_acentos_y_negaciones():
    assert clean_text("  <b>ÓTIMO</b> produto! https://x.com  ") == "ótimo produto!"
    limpio = clean_text("NÃO recebi o produto, uma pena!!!")
    assert "não" in limpio
    assert "recebi" in limpio
    assert "!!!" in limpio


def test_normalize_for_dedup_ignora_puntuacion_y_mayusculas():
    assert normalize_for_dedup("Produto ótimo!") == normalize_for_dedup("produto ótimo")
    assert normalize_for_dedup("  Muito   bom  ") == "muito bom"


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1, 0), (2, 0), (3, 1), (4, 2), (5, 2)],
)
def test_score_to_label(score, expected):
    assert score_to_label(score) == expected


@pytest.mark.parametrize("score", [0, 6, -1, None, "abc"])
def test_score_to_label_invalido(score):
    with pytest.raises(ValueError):
        score_to_label(score)


def test_load_reviews_deduplica_y_etiqueta(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "order_id": "a",
                "review_id": "r1",
                "review_score": 5,
                "review_comment_title": "Ótimo",
                "review_comment_message": "Produto excelente",
                "review_creation_date": "2018-01-01",
                "review_answer_timestamp": "2018-01-02 10:00:00",
            },
            {
                "order_id": "a",
                "review_id": "r2",
                "review_score": 1,
                "review_comment_title": "Ruim",
                "review_comment_message": "Não recebi",
                "review_creation_date": "2018-01-01",
                "review_answer_timestamp": "2018-01-03 10:00:00",
            },
            {
                "order_id": "b",
                "review_id": "r3",
                "review_score": 3,
                "review_comment_title": "",
                "review_comment_message": "Produto ok",
                "review_creation_date": "2018-02-01",
                "review_answer_timestamp": "2018-02-02 10:00:00",
            },
            {
                "order_id": "c",
                "review_id": "r4",
                "review_score": 4,
                "review_comment_title": "",
                "review_comment_message": "Produto ok!",
                "review_creation_date": "2018-03-01",
                "review_answer_timestamp": "2018-03-02 10:00:00",
            },
            {
                "order_id": "d",
                "review_id": "r5",
                "review_score": 2,
                "review_comment_title": "",
                "review_comment_message": "",
                "review_creation_date": "2018-04-01",
                "review_answer_timestamp": "2018-04-02 10:00:00",
            },
            {
                "order_id": "e",
                "review_id": "r6",
                "review_score": 3,
                "review_comment_title": "Ainda não sei",
                "review_comment_message": "Entrega rápida, produto ok",
                "review_creation_date": "2018-05-01",
                "review_answer_timestamp": "2018-05-02 10:00:00",
            },
        ]
    )
    path = tmp_path / "olist_order_reviews_dataset.csv"
    frame.to_csv(path, index=False)

    reviews = load_reviews(tmp_path)

    assert len(reviews) == 3
    assert set(reviews["order_id"]) == {"a", "c", "e"}
    # Para el pedido "a" se conserva la respuesta más reciente (1 estrella).
    assert int(reviews.loc[reviews["order_id"] == "a", "label"].iloc[0]) == 0
    # "Produto ok" y "Produto ok!" se consideran el mismo texto; gana el último.
    assert "c" in set(reviews["order_id"])
    assert "b" not in set(reviews["order_id"])
    assert reviews["text_clean"].str.len().gt(0).all()
    assert set(reviews["label"]).issubset({0, 1, 2})


def test_features_y_pipeline_entrenan_con_texto_pequeno():
    texts = pd.Series(
        [
            "produto excelente",
            "chegou rápido e perfeito",
            "recomendo muito",
            "produto ruim",
            "não recebi o pedido",
            "péssima qualidade",
            "produto ok",
            "mais ou menos",
            "ainda não sei",
        ]
        * 10
    )
    labels = pd.Series([2, 2, 2, 0, 0, 0, 1, 1, 1] * 10)

    features = build_features(min_df=1, max_features_word=500, max_features_char=1000)
    matrix = features.fit_transform(texts)
    assert matrix.shape[0] == len(texts)
    assert matrix.shape[1] > 0

    from sklearn.linear_model import LogisticRegression

    pipeline = build_pipeline(
        LogisticRegression(C=1.0, max_iter=500, random_state=42),
        min_df=1,
        max_features_word=500,
        max_features_char=1000,
    )
    pipeline.fit(texts, labels)
    predicted = pipeline.predict(texts)
    proba = pipeline.predict_proba(texts)
    assert predicted.shape == (len(texts),)
    assert proba.shape == (len(texts), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)

    terms = top_terms(pipeline, top_n=5)
    assert set(terms) == set(LABEL_NAMES.values())
    assert any(terms[label]["asociados"] for label in terms)


def test_holdout_metrics_perfectas():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    proba = np.eye(3)[y_true]
    metrics = holdout_metrics(y_true, y_pred, proba)

    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["cohen_kappa"] == pytest.approx(1.0)
    assert metrics["negative_pr_auc"] == pytest.approx(1.0)
    assert metrics["negative_roc_auc"] == pytest.approx(1.0)
    assert metrics["negative_f1"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_make_splits_y_muestreo_estratificado():
    frame = pd.DataFrame(
        {
            "label": [0] * 20 + [1] * 10 + [2] * 20,
            "text_clean": [f"texto {i}" for i in range(50)],
        }
    )
    train, test = make_splits(frame, test_size=0.2, random_state=42)
    assert len(train) + len(test) == len(frame)
    assert set(test["label"]) == {0, 1, 2}

    sample = stratified_sample(frame, n_per_class=5, random_state=42)
    assert sample["label"].value_counts().to_dict() == {0: 5, 1: 5, 2: 5}


@pytest.mark.skipif(not (ARTIFACTS_DIR / "manifest.json").exists(), reason="falta ejecutar el pipeline de sentimientos")
def test_manifest_cumple_contrato():
    import jsonschema

    manifest = json.loads((ARTIFACTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=manifest, schema=schema)
    assert manifest["status"] == "Completado"
    assert manifest["metrics"]["macro_f1"] > 0
    assert manifest["metrics"]["n_test"] > 0


@pytest.mark.skipif(not (ARTIFACTS_DIR / "chart_data.json").exists(), reason="falta ejecutar el pipeline de sentimientos")
def test_errores_muestran_la_nota_original():
    chart = json.loads((ARTIFACTS_DIR / "chart_data.json").read_text(encoding="utf-8"))
    errors = chart.get("errores", [])
    assert errors, "El análisis de errores debería tener casos"
    first = errors[0]
    assert "review_score" in first
    assert 1 <= int(first["review_score"]) <= 5
    assert first["real"] in LABEL_NAMES.values()
    assert first["prediccion"] in LABEL_NAMES.values()


@pytest.mark.skipif(not (ARTIFACTS_DIR / "model.joblib").exists(), reason="falta el modelo de sentimientos")
def test_servicio_de_inferencia():
    from backend.services.sentiment import predict_sentiment

    result = predict_sentiment("O produto chegou rápido e em perfeito estado. Recomendo!")
    assert result["sentimiento"] in LABEL_NAMES.values()
    assert 0.0 <= result["confianza"] <= 1.0
    assert abs(sum(result["probabilidades"].values()) - 1.0) < 1e-6
