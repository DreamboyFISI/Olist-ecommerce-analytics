"""
Módulo productivo de Análisis de Sentimientos de reseñas — DataSmart Olist.
=============================================================================
Responsable: Módulo de NLP / Sentimientos (rama feature/nlp)
Proyecto Integrador · Minería de Datos · UNMSM FISI 2026

Objetivo
--------
Clasificar el sentimiento del texto libre de las reseñas de Olist en tres
categorías construidas a partir de la nota del cliente:

- Negativo: 1–2 estrellas.
- Neutro:   3 estrellas.
- Positivo: 4–5 estrellas.

El texto está en portugués brasileño, es corto, coloquial y contiene ruido
(errores de tipeo, mayúsculas, emojis, URLs). Por eso se combinan n-gramas de
palabra y de carácter, se conservan las negaciones y se comparan varios
algoritmos con validación cruzada estratificada.

Por qué esta evaluación
-----------------------
- La nota de la reseña es una etiqueta débil: no es una anotación humana de
  sentimiento, pero es la señal más cercana y disponible para todo el dataset.
- Las clases están desbalanceadas (la positiva domina), así que la métrica
  principal es **macro-F1**, que da el mismo peso a negativo, neutro y positivo.
- Para el negocio, lo crítico es detectar reseñas negativas: se reportan
  además PR-AUC, ROC-AUC y F1 de la clase negativa, con un umbral ajustado
  fuera de muestra.
- Como la nota original tiene orden (1 a 5), se incluye una evaluación auxiliar
  de 5 clases con **QWK** (kappa ponderada cuadrática) y accuracy ±1 estrella.

El modelo desplegable es un Pipeline de scikit-learn (TF-IDF + clasificador),
serializado en `artifacts/nlp/model.joblib`. El benchmark con un transformer
preentrenado es opcional y se guarda aparte para no añadir torch a la API.

Punto de entrada ejecutable:
    python -m src.models.sentiment
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import jsonschema
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Configuración y rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts" / "nlp"
SCHEMA_PATH = ROOT / "docs" / "module_contract.schema.json"
TRANSFORMER_BENCHMARK_PATH = ARTIFACTS_DIR / "transformer_benchmark.json"

RANDOM_STATE = 42
LABELS = [0, 1, 2]
LABEL_NAMES = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
LABEL_SHORT = {0: "neg", 1: "neu", 2: "pos"}
REVIEW_TEST_SIZE = 0.20
TEXT_COLUMN = "text_clean"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger("sentiment")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# 1. Limpieza y carga de datos
# ---------------------------------------------------------------------------

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+", flags=re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\sÀ-ÿ]", flags=re.UNICODE)
_HAS_LETTER_RE = re.compile(r"[A-Za-zÀ-ÿ]")


def clean_text(value: Any) -> str:
    """Normaliza el texto conservando acentos, negaciones y puntuación útil.

    No se eliminan stopwords: en portugués, palabras como "não", "nunca" o
    "sem" cambian por completo el sentido de la reseña. Los signos de
    exclamación e interrogación también aportan intensidad y se conservan.
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _CONTROL_RE.sub(" ", text)
    text = text.lower()
    return _SPACE_RE.sub(" ", text).strip()


def normalize_for_dedup(value: Any) -> str:
    """Normalización agresiva usada solo para detectar textos duplicados."""
    text = clean_text(value)
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def score_to_label(score: Any) -> int:
    """Convierte la nota de 1–5 estrellas a la clase de sentimiento."""
    try:
        value = int(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nota de reseña inválida: {score!r}") from exc
    if value < 1 or value > 5:
        raise ValueError(f"Nota de reseña fuera de rango: {score!r}")
    if value <= 2:
        return 0
    if value == 3:
        return 1
    return 2


def _combine_title_and_message(frame: pd.DataFrame) -> pd.Series:
    title = frame.get("review_comment_title", pd.Series("", index=frame.index))
    message = frame.get("review_comment_message", pd.Series("", index=frame.index))
    title = title.fillna("").astype(str).str.strip()
    message = message.fillna("").astype(str).str.strip()
    combined = np.where(title.ne(""), title + ". " + message, message)
    return pd.Series(combined, index=frame.index).str.strip(" .")


def load_reviews(data_dir: Path | None = None) -> pd.DataFrame:
    """Carga, deduplica y etiqueta las reseñas con texto.

    Reglas:
    1. Una reseña por pedido (la última respondida), igual que clasificación.
    2. Se conservan las reseñas con título o mensaje y al menos una letra.
    3. Se eliminan duplicados exactos del texto normalizado. Esto evita que
       frases plantilla como "Bom" o "Não recebi" aparezcan en train y test e
       inflen artificialmente las métricas.
    """
    path = (data_dir or RAW_DIR) / "olist_order_reviews_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de reseñas: {path}")

    logger.info("Cargando reseñas desde: %s", path)
    frame = pd.read_csv(
        path,
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )
    frame = frame.sort_values("review_answer_timestamp").drop_duplicates(
        "order_id", keep="last"
    )
    frame["text"] = _combine_title_and_message(frame)
    frame = frame[
        frame["text"].map(lambda value: bool(_HAS_LETTER_RE.search(str(value))))
    ].copy()
    frame["text_clean"] = frame["text"].map(clean_text)
    frame["text_dedup"] = frame["text"].map(normalize_for_dedup)
    frame = frame[frame["text_clean"].str.len() > 0]

    before = len(frame)
    frame = frame.drop_duplicates("text_dedup", keep="last").copy()
    logger.info(
        "Reseñas con texto: %d | tras deduplicar: %d (%d duplicados exactos)",
        before,
        len(frame),
        before - len(frame),
    )

    frame["review_score"] = frame["review_score"].astype(int)
    frame["label"] = frame["review_score"].map(score_to_label)
    frame["label_name"] = frame["label"].map(LABEL_NAMES)
    frame["n_words"] = frame["text_clean"].str.split().str.len().astype(int)
    frame["text_len"] = frame["text_clean"].str.len().astype(int)
    frame = frame.sort_values("review_creation_date").reset_index(drop=True)

    columns = [
        "order_id",
        "review_id",
        "review_score",
        "label",
        "label_name",
        "text",
        "text_clean",
        "n_words",
        "text_len",
        "review_creation_date",
        "review_answer_timestamp",
    ]
    return frame[columns]


def load_master(path: Path | None = None) -> pd.DataFrame | None:
    """Carga el tablón de pedidos para cruzar sentimiento con logística."""
    master_path = path or (PROCESSED_DIR / "master_pedidos.csv")
    if not master_path.exists():
        logger.warning(
            "No se encontró %s; se omiten los cruces con logística y categoría.",
            master_path,
        )
        return None
    logger.info("Cargando tablón de pedidos: %s", master_path)
    return pd.read_csv(master_path, low_memory=False)


def make_splits(
    frame: pd.DataFrame,
    test_size: float = REVIEW_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición estratificada por clase de sentimiento."""
    train, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame["label"],
        random_state=random_state,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def stratified_sample(
    frame: pd.DataFrame,
    n_per_class: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Muestra hasta `n_per_class` filas por clase, sin perder la etiqueta."""
    parts = [
        subset.sample(n=min(len(subset), n_per_class), random_state=random_state)
        for _, subset in frame.groupby("label")
    ]
    if not parts:
        return frame.head(0).copy()
    return (
        pd.concat(parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 2. Representación del texto y modelos candidatos
# ---------------------------------------------------------------------------


def build_features(
    min_df: int = 3,
    max_features_word: int = 60_000,
    max_features_char: int = 80_000,
) -> FeatureUnion:
    """TF-IDF de palabras (1–2) + TF-IDF de caracteres (3–5).

    Los n-gramas de carácter capturan errores de tipeo, conjugaciones y
    variantes morfológicas del portugués; los de palabra aportan contexto y
    negaciones compuestas ("não recebi", "muito ruim").
    """
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=max_features_word,
        token_pattern=r"(?u)\b\w\w+\b",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        sublinear_tf=True,
        max_features=max_features_char,
        dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])


def build_pipeline(classifier: Any, **feature_kwargs: Any) -> Pipeline:
    """Pipeline completo: texto crudo limpio -> TF-IDF -> clasificador."""
    return Pipeline(
        [
            ("features", build_features(**feature_kwargs)),
            ("clf", classifier),
        ]
    )


def model_candidates() -> dict[str, Pipeline]:
    """Algoritmos comparados. La regresión logística ajustada se añade aparte."""
    return {
        "Baseline (clase mayoritaria)": build_pipeline(
            DummyClassifier(strategy="most_frequent")
        ),
        "Naive Bayes (ComplementNB)": build_pipeline(ComplementNB(alpha=0.3)),
        "SVM lineal calibrado": build_pipeline(
            CalibratedClassifierCV(
                LinearSVC(
                    C=0.3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                cv=3,
                method="sigmoid",
            )
        ),
    }


def default_logistic() -> Pipeline:
    return build_pipeline(
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=RANDOM_STATE,
        )
    )


def tune_logistic(
    texts: pd.Series,
    labels: pd.Series,
    cv_splits: int = 3,
) -> tuple[Pipeline, pd.DataFrame]:
    """Búsqueda pequeña y reproducible de C para la regresión logística."""
    search = GridSearchCV(
        default_logistic(),
        {"clf__C": [0.3, 0.5, 0.7, 1.0]},
        scoring="f1_macro",
        cv=StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE
        ),
        n_jobs=1,
        refit=True,
    )
    search.fit(texts, labels)
    detail = pd.DataFrame(search.cv_results_)[
        ["param_clf__C", "mean_test_score", "std_test_score", "rank_test_score"]
    ].copy()
    detail = detail.rename(
        columns={
            "param_clf__C": "C",
            "mean_test_score": "macro_f1_medio",
            "std_test_score": "macro_f1_std",
            "rank_test_score": "ranking",
        }
    )
    detail = detail.sort_values("ranking").reset_index(drop=True)
    logger.info(
        "Mejor C de regresión logística: %s (macro-F1 CV %.4f)",
        search.best_params_["clf__C"],
        search.best_score_,
    )
    return search.best_estimator_, detail


# ---------------------------------------------------------------------------
# 3. Evaluación
# ---------------------------------------------------------------------------


def _score_dict() -> dict[str, Any]:
    return {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "cohen_kappa": make_scorer(cohen_kappa_score, weights="quadratic"),
    }


def cross_validate_models(
    texts: pd.Series,
    labels: pd.Series,
    models: dict[str, Pipeline],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Validación cruzada estratificada con macro-F1 como criterio principal."""
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    rows: list[dict[str, Any]] = []
    for name, pipeline in models.items():
        started = time.time()
        result = cross_validate(
            clone(pipeline),
            texts,
            labels,
            cv=cv,
            scoring=_score_dict(),
            n_jobs=1,
            error_score="raise",
        )
        row: dict[str, Any] = {"modelo": name}
        for metric in _score_dict():
            values = np.asarray(result[f"test_{metric}"], dtype=float)
            row[f"{metric}_cv"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        row["tiempo_cv_seg"] = round(time.time() - started, 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1_cv", ascending=False).reset_index(drop=True)


def holdout_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: np.ndarray | None = None,
    labels: list[int] = LABELS,
) -> dict[str, Any]:
    """Métricas completas de la evaluación final."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    target_names = [LABEL_NAMES[label] for label in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "n": int(len(y_true)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in target_names
        },
    }

    if y_proba is not None:
        y_neg = (y_true == 0).astype(int)
        p_neg = np.asarray(y_proba)[:, 0]
        y_neg_pred = (y_pred == 0).astype(int)
        metrics.update(
            {
                "negative_pr_auc": float(average_precision_score(y_neg, p_neg)),
                "negative_roc_auc": float(roc_auc_score(y_neg, p_neg)),
                "negative_brier": float(brier_score_loss(y_neg, p_neg)),
                "negative_precision": float(
                    precision_score(y_neg, y_neg_pred, zero_division=0)
                ),
                "negative_recall": float(
                    recall_score(y_neg, y_neg_pred, zero_division=0)
                ),
                "negative_f1": float(f1_score(y_neg, y_neg_pred, zero_division=0)),
            }
        )
    return metrics


def optimize_negative_threshold(
    pipeline: Pipeline,
    texts: pd.Series,
    labels: pd.Series,
    n_splits: int = 5,
) -> dict[str, float]:
    """Umbral de probabilidad de negativa que maximiza F1 fuera de muestra."""
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    oof = cross_val_predict(
        clone(pipeline), texts, labels, cv=cv, method="predict_proba", n_jobs=1
    )
    p_neg = np.asarray(oof)[:, 0]
    y_neg = (np.asarray(labels) == 0).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_neg, p_neg)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None
    )
    best = int(np.argmax(f1_values)) if len(f1_values) else 0
    threshold = float(thresholds[best]) if best < len(thresholds) else 0.5
    return {
        "threshold": threshold,
        "oof_f1": float(f1_values[best]) if len(f1_values) else 0.0,
        "oof_precision": float(precision[best]) if len(precision) else 0.0,
        "oof_recall": float(recall[best]) if len(recall) else 0.0,
        "oof_pr_auc": float(average_precision_score(y_neg, p_neg)),
    }


def evaluate_ordinal(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Evaluación auxiliar de la nota exacta (1–5) como problema ordinal."""
    pipeline = build_pipeline(
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=RANDOM_STATE,
        )
    )
    pipeline.fit(train[TEXT_COLUMN], train["review_score"])
    predicted = pipeline.predict(test[TEXT_COLUMN])
    proba = pipeline.predict_proba(test[TEXT_COLUMN])
    y_true = test["review_score"].to_numpy()
    classes = [int(value) for value in pipeline.named_steps["clf"].classes_]
    report = classification_report(
        y_true, predicted, output_dict=True, zero_division=0
    )
    return {
        "exact_accuracy": float(accuracy_score(y_true, predicted)),
        "off_by_one_accuracy": float(np.mean(np.abs(y_true - predicted) <= 1)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, predicted, weights="quadratic")
        ),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "negative_pr_auc": float(
            average_precision_score((y_true <= 2).astype(int), proba[:, 0] + proba[:, 1])
        ),
        "classes": classes,
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=classes).tolist(),
        "per_class": {
            str(label): {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in classes
        },
    }


# ---------------------------------------------------------------------------
# 4. Interpretabilidad y cruces de negocio
# ---------------------------------------------------------------------------

_STOPWORDS_DISPLAY = {
    "de", "da", "do", "das", "dos", "que", "e", "o", "a", "os", "as",
    "em", "um", "uma", "uns", "umas", "para", "por", "com", "sem", "no",
    "na", "nos", "nas", "ao", "aos", "se", "foi", "ser", "sao", "mais",
    "mas", "ou", "como", "ja", "muito", "bem", "tambem", "pelo", "pela",
    "meu", "minha", "seu", "sua", "ele", "ela", "isso", "esse", "essa",
}


def _valid_display_term(term: str) -> bool:
    term = term.strip()
    if len(term) < 3 or term in _STOPWORDS_DISPLAY:
        return False
    if term.isdigit():
        return False
    if not _HAS_LETTER_RE.search(term):
        return False
    return True


def _unwrap_classifier(pipeline: Pipeline) -> Any:
    classifier = pipeline.named_steps["clf"]
    if isinstance(classifier, CalibratedClassifierCV):
        classifier = classifier.calibrated_classifiers_[0].estimator
    return classifier


def top_terms(pipeline: Pipeline, top_n: int = 20) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Términos con mayor peso por clase a partir del vectorizador de palabras."""
    features = pipeline.named_steps["features"]
    word_vectorizer = features.named_transformers["word"]
    names = word_vectorizer.get_feature_names_out()
    n_word = len(names)

    classifier = _unwrap_classifier(pipeline)
    if hasattr(classifier, "coef_"):
        weights = np.asarray(classifier.coef_)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        if weights.shape[0] == 1:
            weights = np.vstack([-weights, np.zeros_like(weights), weights])
    elif hasattr(classifier, "feature_log_prob_"):
        weights = np.asarray(classifier.feature_log_prob_)
    else:  # pragma: no cover - los modelos actuales siempre exponen pesos
        return {}

    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for index, label in enumerate(LABELS):
        if index >= weights.shape[0]:
            continue
        row = weights[index, :n_word]
        order = np.argsort(row)
        positive = []
        for position in order[::-1]:
            term = str(names[position])
            if _valid_display_term(term):
                positive.append({"termino": term, "peso": float(row[position])})
            if len(positive) >= top_n:
                break
        negative = []
        for position in order:
            term = str(names[position])
            if _valid_display_term(term):
                negative.append({"termino": term, "peso": float(row[position])})
            if len(negative) >= top_n:
                break
        result[LABEL_NAMES[label]] = {
            "asociados": positive,
            "opuestos": negative,
        }
    return result


def _round_rates(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rounded = frame.copy()
    for column in columns:
        rounded[column] = rounded[column].astype(float).round(4)
    return rounded.to_dict(orient="records")


def build_business_insights(
    predictions: pd.DataFrame,
    master: pd.DataFrame | None,
) -> dict[str, Any]:
    """Cruza el sentimiento predicho con logística, categoría y tiempo."""
    frame = predictions.copy()
    if master is not None:
        available = [
            "order_id",
            "categoria_principal",
            "dias_retraso",
            "entrega_tardia",
            "order_status",
            "order_delivered_customer_date",
            "customer_state",
        ]
        columns = [column for column in available if column in master.columns]
        master_small = master[columns].drop_duplicates("order_id")
        frame = frame.merge(master_small, on="order_id", how="left", suffixes=("", "_master"))

    frame["es_negativa"] = frame["label_pred"] == 0
    frame["es_positiva"] = frame["label_pred"] == 2
    frame["mes"] = (
        pd.to_datetime(frame["review_creation_date"]).dt.to_period("M").astype(str)
    )
    frame["tramo_longitud"] = pd.cut(
        frame["n_words"],
        bins=[-np.inf, 5, 10, 20, 40, np.inf],
        labels=["1–5", "6–10", "11–20", "21–40", ">40"],
        include_lowest=True,
    ).astype(str)

    if "order_delivered_customer_date" in frame.columns:
        sin_entrega = pd.to_datetime(
            frame["order_delivered_customer_date"], errors="coerce"
        ).isna()
    elif "order_status" in frame.columns:
        sin_entrega = frame["order_status"].ne("delivered")
    else:
        sin_entrega = pd.Series(False, index=frame.index)

    if "dias_retraso" in frame.columns:
        retraso = pd.to_numeric(frame["dias_retraso"], errors="coerce")
    elif "entrega_tardia" in frame.columns:
        retraso = pd.to_numeric(frame["entrega_tardia"], errors="coerce")
    else:
        retraso = pd.Series(np.nan, index=frame.index)
    frame["estado_entrega"] = np.select(
        [sin_entrega, retraso > 0],
        ["Sin entrega", "Entrega tardía"],
        default="Entrega a tiempo",
    )

    def group_summary(column: str, min_rows: int = 20, limit: int | None = None) -> list[dict[str, Any]]:
        if column not in frame.columns:
            return []
        grouped = (
            frame.groupby(column, observed=True)
            .agg(
                resenas=("order_id", "size"),
                tasa_negativa=("es_negativa", "mean"),
                tasa_positiva=("es_positiva", "mean"),
            )
            .reset_index()
        )
        grouped.columns = [column, "resenas", "tasa_negativa", "tasa_positiva"]
        grouped = grouped[grouped["resenas"] >= min_rows]
        if column == "categoria_principal":
            grouped = grouped[grouped[column].astype(str).ne("sin_categoria")]
        grouped = grouped.sort_values(["resenas", "tasa_negativa"], ascending=[False, False])
        if limit:
            grouped = grouped.head(limit)
        return _round_rates(grouped, ["tasa_negativa", "tasa_positiva"])

    return {
        "resumen": {
            "resenas": int(len(frame)),
            "tasa_negativa_predicha": float(frame["es_negativa"].mean()),
            "tasa_positiva_predicha": float(frame["es_positiva"].mean()),
            "confianza_media": float(frame["confianza"].mean()),
        },
        "por_estado_entrega": _round_rates(
            frame.groupby("estado_entrega", observed=True)
            .agg(
                resenas=("order_id", "size"),
                tasa_negativa=("es_negativa", "mean"),
                tasa_positiva=("es_positiva", "mean"),
            )
            .reset_index(),
            ["tasa_negativa", "tasa_positiva"],
        ),
        "por_categoria": group_summary("categoria_principal", min_rows=30, limit=15),
        "por_mes": group_summary("mes", min_rows=20),
        "por_estado_cliente": group_summary("customer_state", min_rows=30, limit=12),
        "por_longitud": group_summary("tramo_longitud", min_rows=20),
    }


def build_results_frame(
    test: pd.DataFrame,
    predicted: np.ndarray,
    proba: np.ndarray,
    master: pd.DataFrame | None,
) -> pd.DataFrame:
    """Tabla de predicciones con texto y contexto para dashboard y artefactos."""
    results = test[
        [
            "order_id",
            "review_score",
            "label",
            "label_name",
            "text",
            "n_words",
            "review_creation_date",
        ]
    ].copy()
    results["label_pred"] = predicted
    results["label_pred_name"] = pd.Series(predicted).map(LABEL_NAMES).to_numpy()
    results["prob_negativo"] = np.asarray(proba)[:, 0]
    results["prob_neutro"] = np.asarray(proba)[:, 1]
    results["prob_positivo"] = np.asarray(proba)[:, 2]
    results["confianza"] = np.asarray(proba).max(axis=1)
    results["acierto"] = results["label"] == results["label_pred"]

    if master is not None:
        available = [
            "order_id",
            "categoria_principal",
            "dias_retraso",
            "entrega_tardia",
            "order_status",
            "order_delivered_customer_date",
            "customer_state",
        ]
        columns = [column for column in available if column in master.columns]
        results = results.merge(
            master[columns].drop_duplicates("order_id"),
            on="order_id",
            how="left",
        )

    return results


def calibration_points(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Curva de calibración de la probabilidad de clase negativa."""
    y_neg = (np.asarray(y_true) == 0).astype(int)
    p_neg = np.asarray(proba)[:, 0]
    frac, mean_pred = calibration_curve(
        y_neg, p_neg, n_bins=n_bins, strategy="quantile"
    )
    return [
        {"probabilidad_media": float(mean_pred[i]), "fraccion_negativas": float(frac[i])}
        for i in range(len(frac))
    ]


# ---------------------------------------------------------------------------
# 5. Exportación de artefactos
# ---------------------------------------------------------------------------


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _sample_results(results: pd.DataFrame, size: int = 500) -> pd.DataFrame:
    parts = []
    for label in LABELS:
        subset = results[results["label"] == label]
        if subset.empty:
            continue
        parts.append(subset.sample(n=min(len(subset), size // len(LABELS)), random_state=RANDOM_STATE))
    sample = pd.concat(parts) if parts else results.head(size)
    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    sample["text"] = sample["text"].astype(str).str.slice(0, 500)
    return sample


def export_artifacts(
    reviews: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cv_table: pd.DataFrame,
    comparison: pd.DataFrame,
    winner_name: str,
    winner_pipeline: Pipeline,
    winner_metrics: dict[str, Any],
    threshold_info: dict[str, float],
    ordinal: dict[str, Any],
    terms: dict[str, Any],
    results: pd.DataFrame,
    insights: dict[str, Any],
    tuning_detail: pd.DataFrame | None,
    transformer: dict[str, Any] | None,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Guarda CSV, JSON, modelo y manifest validado contra el contrato."""
    out_dir = artifacts_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exportando artefactos a: %s", out_dir)

    # 1. results.csv (muestra estratificada del test)
    sample = _sample_results(results, size=500)
    results_path = out_dir / "results.csv"
    export_columns = [
        "order_id",
        "review_score",
        "label",
        "label_name",
        "label_pred",
        "label_pred_name",
        "prob_negativo",
        "prob_neutro",
        "prob_positivo",
        "confianza",
        "acierto",
        "n_words",
        "review_creation_date",
        "categoria_principal",
        "dias_retraso",
        "order_status",
        "text",
    ]
    export_columns = [column for column in export_columns if column in sample.columns]
    sample[export_columns].to_csv(results_path, index=False, encoding="utf-8")
    logger.info("results.csv: %s (%d filas)", results_path, len(sample))

    # 2. comparacion_modelos.csv
    comparison_path = out_dir / "comparacion_modelos.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    logger.info("comparacion_modelos.csv: %s", comparison_path)

    # 3. metricas_por_clase.csv
    per_class = pd.DataFrame(winner_metrics["per_class"]).T.reset_index()
    per_class.columns = ["clase", "precision", "recall", "f1", "soporte"]
    per_class_path = out_dir / "metricas_por_clase.csv"
    per_class.to_csv(per_class_path, index=False, encoding="utf-8")

    # 4. terminos_por_clase.csv
    term_rows = []
    for label_name, groups in terms.items():
        for group_name, values in groups.items():
            for item in values:
                term_rows.append(
                    {
                        "clase": label_name,
                        "grupo": group_name,
                        "termino": item["termino"],
                        "peso": item["peso"],
                    }
                )
    terms_path = out_dir / "terminos_por_clase.csv"
    pd.DataFrame(term_rows).to_csv(terms_path, index=False, encoding="utf-8")

    if tuning_detail is not None:
        tuning_detail.to_csv(out_dir / "ajuste_logistico.csv", index=False, encoding="utf-8")

    # 5. chart_data.json
    class_distribution = {
        split_name: [
            {
                "clase": LABEL_NAMES[label],
                "resenas": int((split["label"] == label).sum()),
                "porcentaje": float((split["label"] == label).mean()),
            }
            for label in LABELS
        ]
        for split_name, split in (("train", train), ("test", test))
    }
    errors = results[~results["acierto"]].sort_values("confianza", ascending=False).head(40)
    error_rows = [
        {
            "texto": str(row["text"])[:400],
            "real": LABEL_NAMES[int(row["label"])],
            "prediccion": LABEL_NAMES[int(row["label_pred"])],
            "confianza": float(row["confianza"]),
            "prob_negativo": float(row["prob_negativo"]),
            "prob_neutro": float(row["prob_neutro"]),
            "prob_positivo": float(row["prob_positivo"]),
        }
        for _, row in errors.iterrows()
    ]
    length_bins = (
        results.groupby(["label_name", "n_words"], observed=True)
        .size()
        .reset_index(name="resenas")
    )
    chart_data = {
        "clases": [
            {"id": label, "nombre": LABEL_NAMES[label], "corto": LABEL_SHORT[label]}
            for label in LABELS
        ],
        "distribucion_clases": class_distribution,
        "comparacion_modelos": comparison.to_dict(orient="records"),
        "matriz_confusion": winner_metrics["confusion_matrix"],
        "metricas_por_clase": winner_metrics["per_class"],
        "metricas_ganador": winner_metrics,
        "umbral_negativo": threshold_info,
        "terminos": terms,
        "insights_negocio": insights,
        "calibracion": calibration_points(
            test["label"].to_numpy(),
            results["label_pred"].to_numpy(),
            results[["prob_negativo", "prob_neutro", "prob_positivo"]].to_numpy(),
        ),
        "errores": error_rows,
        "longitud_texto": length_bins.to_dict(orient="records"),
        "ordinal_5_estrellas": ordinal,
        "ajuste_logistico": (
            tuning_detail.to_dict(orient="records") if tuning_detail is not None else []
        ),
        "transformer_referencia": transformer,
        "resumen": {
            "n_total": int(len(reviews)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_clases": len(LABELS),
            "palabras_medias": float(reviews["n_words"].mean()),
            "periodo": {
                "desde": str(reviews["review_creation_date"].min().date()),
                "hasta": str(reviews["review_creation_date"].max().date()),
            },
        },
    }
    chart_path = out_dir / "chart_data.json"
    with open(chart_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(chart_data), handle, ensure_ascii=False, indent=2)
    logger.info("chart_data.json: %s", chart_path)

    # 6. model.joblib
    winner_label = (
        f"{winner_name} + TF-IDF palabra/carácter"
        if winner_name == "Regresión Logística"
        else winner_name
    )
    model_payload = {
        "model_name": "Análisis de sentimientos de reseñas (PT-BR)",
        "version": "1.0.0",
        "winner": winner_label,
        "pipeline": winner_pipeline,
        "labels": LABELS,
        "label_names": LABEL_NAMES,
        "negative_threshold": threshold_info["threshold"],
        "metrics": winner_metrics,
        "features": {
            "text_column": TEXT_COLUMN,
            "word_ngrams": [1, 2],
            "char_ngrams": [3, 5],
            "max_features_word": 60_000,
            "max_features_char": 80_000,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(train)),
    }
    model_path = out_dir / "model.joblib"
    joblib.dump(model_payload, model_path, compress=3)
    logger.info("model.joblib: %s", model_path)

    # 7. manifest.json
    best_cv = cv_table.iloc[0]
    transformer_metrics = (transformer or {}).get("metrics", {})
    metrics = {
        "macro_f1": round(float(winner_metrics["macro_f1"]), 4),
        "weighted_f1": round(float(winner_metrics["weighted_f1"]), 4),
        "accuracy": round(float(winner_metrics["accuracy"]), 4),
        "balanced_accuracy": round(float(winner_metrics["balanced_accuracy"]), 4),
        "cohen_kappa": round(float(winner_metrics["cohen_kappa"]), 4),
        "negative_pr_auc": round(float(winner_metrics.get("negative_pr_auc", 0.0)), 4),
        "negative_roc_auc": round(float(winner_metrics.get("negative_roc_auc", 0.0)), 4),
        "negative_f1": round(float(winner_metrics.get("negative_f1", 0.0)), 4),
        "negative_precision": round(float(winner_metrics.get("negative_precision", 0.0)), 4),
        "negative_recall": round(float(winner_metrics.get("negative_recall", 0.0)), 4),
        "umbral_negativo": round(float(threshold_info["threshold"]), 4),
        "macro_f1_cv": round(float(best_cv["macro_f1_cv"]), 4),
        "qwk_5_estrellas": round(float(ordinal["quadratic_weighted_kappa"]), 4),
        "exactitud_5_estrellas": round(float(ordinal["exact_accuracy"]), 4),
        "off_by_one_5_estrellas": round(float(ordinal["off_by_one_accuracy"]), 4),
        "n_total": int(len(reviews)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "tasa_negativa_test": round(float((test["label"] == 0).mean()), 4),
        "transformer_macro_f1": (
            round(float(transformer_metrics.get("macro_f1", 0.0)), 4)
            if transformer_metrics
            else None
        ),
    }
    delivery = insights.get("por_estado_entrega", [])
    delivery_map = {row["estado_entrega"]: row for row in delivery}
    late_rate = delivery_map.get("Entrega tardía", {}).get("tasa_negativa")
    on_time_rate = delivery_map.get("Entrega a tiempo", {}).get("tasa_negativa")
    missing_rate = delivery_map.get("Sin entrega", {}).get("tasa_negativa")
    categories = insights.get("por_categoria", [])
    worst_category = max(categories, key=lambda row: row["tasa_negativa"]) if categories else None
    neutral_f1 = winner_metrics["per_class"]["Neutro"]["f1"]
    best_classical_balanced = comparison.loc[
        comparison["tipo"] == "clásico", "macro_f1_balanceado"
    ].max()

    if transformer_metrics:
        gap = float(transformer_metrics.get("macro_f1", 0.0)) - float(best_classical_balanced)
        if gap > 0.01:
            verdict = (
                "El transformer supera al clásico en esta muestra, pero no se despliega por costo y "
                "latencia; fine-tunear BERTimbau en GPU es la siguiente mejora."
            )
        elif gap < -0.01:
            verdict = (
                "El clásico mantiene el liderazgo; se despliega por latencia, costo e interpretabilidad."
            )
        else:
            verdict = (
                "Empate técnico: el clásico se despliega por latencia, costo e interpretabilidad, y "
                "fine-tunear BERTimbau en GPU queda como mejora pendiente."
            )
        transformer_sentence = (
            f"El transformer externo zero-shot alcanzó macro-F1 "
            f"{transformer_metrics.get('macro_f1', 0):.3f} en la muestra balanceada, frente a "
            f"{best_classical_balanced:.3f} del mejor modelo clásico. {verdict}"
        )
    else:
        transformer_sentence = (
            "El benchmark de transformer no está disponible; el módulo clásico es el modelo desplegado."
        )

    business_conclusions = [
        f"El modelo detecta reseñas negativas con PR-AUC {metrics['negative_pr_auc']:.3f} y ROC-AUC "
        f"{metrics['negative_roc_auc']:.3f}; con el umbral {metrics['umbral_negativo']:.2f} logra "
        f"precisión {metrics['negative_precision'] * 100:.1f}% y recall {metrics['negative_recall'] * 100:.1f}%.",
        f"El sentimiento está muy ligado a la logística: las reseñas de pedidos con entrega tardía "
        f"tienen {late_rate * 100:.1f}% de negativas predichas frente a {on_time_rate * 100:.1f}% de los "
        f"entregados a tiempo, y {missing_rate * 100:.1f}% cuando no hay entrega registrada."
        if late_rate is not None and on_time_rate is not None and missing_rate is not None
        else "El cruce con logística no está disponible porque falta data/processed/master_pedidos.csv.",
        f"Entre las 15 categorías con más reseñas, la de mayor tasa de negativas es "
        f"{worst_category['categoria_principal']} ({worst_category['tasa_negativa'] * 100:.1f}% de "
        f"{worst_category['resenas']} reseñas), una prioridad para revisar producto, embalaje y expectativas."
        if worst_category
        else "El cruce por categoría no está disponible en este entorno.",
        f"La clase neutra es la más difícil (F1 {neutral_f1:.3f}): muchas reseñas de 3 estrellas son "
        f"ambivalentes o descriptivas. Para decisiones operativas conviene usar la probabilidad de "
        f"negativa y no solo la clase argmax.",
        transformer_sentence,
    ]
    limitations = [
        "La nota de 1–5 estrellas es una etiqueta débil: mide satisfacción global y no sentimiento puro; "
        "las reseñas de 3 estrellas son especialmente ambiguas.",
        "Solo ~41% de las reseñas tienen comentario. La muestra está sesgada hacia clientes que deciden "
        "escribir, que suelen ser los muy satisfechos o los muy insatisfechos.",
        "El texto es portugués coloquial, con errores de tipeo, ironía, emojis y abreviaturas; los n-gramas "
        "de carácter ayudan, pero no resuelven la ironía ni el sarcasmo.",
        "Se eliminaron duplicados exactos de texto para evitar fuga de información; near-duplicates y "
        "plantillas casi idénticas pueden seguir inflando levemente la evaluación.",
        "La partición train/test es aleatoria y estratificada, no temporal: mide generalización a "
        "reseñas similares, no deriva futura. Para producción conviene añadir un holdout temporal.",
        "Las tasas por categoría se calculan sobre el test y algunas categorías tienen menos de 200 "
        "reseñas; son señales exploratorias, no estimaciones precisas.",
        "El benchmark con transformer es zero-shot sobre un modelo entrenado en otro e-commerce (B2W). "
        "No se fine-tuneó BERTimbau en Olist por costo computacional; con GPU el techo sería mayor.",
        "El modelo se entrenó con reseñas de 2016–2018. Cambios de catálogo, logística o lenguaje "
        "requieren monitoreo de deriva y reentrenamiento periódico.",
        "Es un modelo descriptivo, no causal: una asociación entre retraso y negatividad no prueba que "
        "el retraso sea la única causa.",
    ]
    manifest = {
        "status": "Completado",
        "model_name": "Análisis de sentimientos de reseñas",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Olist Brazilian E-Commerce — olist_order_reviews_dataset.csv",
        "period": f"{reviews['review_creation_date'].min().date()} a {reviews['review_creation_date'].max().date()}",
        "metrics": metrics,
        "winner": winner_label,
        "results_path": "artifacts/nlp/results.csv",
        "chart_data_path": "artifacts/nlp/chart_data.json",
        "business_conclusions": business_conclusions,
        "limitations": limitations,
        "serialized_model_path": "artifacts/nlp/model.joblib",
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(manifest), handle, ensure_ascii=False, indent=2)
    logger.info("manifest.json: %s", manifest_path)

    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, encoding="utf-8") as handle:
            schema = json.load(handle)
        jsonschema.validate(instance=manifest, schema=schema)
        logger.info("manifest.json cumple docs/module_contract.schema.json")

    return manifest


# ---------------------------------------------------------------------------
# 6. Orquestación
# ---------------------------------------------------------------------------


def run_pipeline(
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    quick: bool = False,
    skip_tuning: bool = False,
) -> dict[str, Any]:
    """Ejecuta de principio a fin el módulo de sentimientos."""
    started = time.time()
    logger.info("Iniciando pipeline de análisis de sentimientos...")

    reviews = load_reviews(data_dir)
    if quick:
        reviews = stratified_sample(reviews, n_per_class=900)
        logger.info("Modo rápido: %d reseñas", len(reviews))

    train, test = make_splits(reviews)
    logger.info(
        "Partición: train=%d | test=%d | clases train=%s",
        len(train),
        len(test),
        train["label"].value_counts().sort_index().to_dict(),
    )

    n_splits = 3 if quick else 5
    if skip_tuning or quick:
        logistic = default_logistic()
        tuning_detail = None
    else:
        logistic, tuning_detail = tune_logistic(
            train[TEXT_COLUMN], train["label"], cv_splits=3
        )

    models = model_candidates()
    models["Regresión Logística"] = logistic

    logger.info("Comparando %d modelos con validación cruzada...", len(models))
    cv_table = cross_validate_models(
        train[TEXT_COLUMN], train["label"], models, n_splits=n_splits
    )
    winner_name = str(cv_table.iloc[0]["modelo"])
    logger.info(
        "Ganador por macro-F1 CV: %s (%.4f)",
        winner_name,
        cv_table.iloc[0]["macro_f1_cv"],
    )

    balanced_test = stratified_sample(test, n_per_class=500)

    comparison_rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in models.items():
        logger.info("Ajustando y evaluando en test: %s", name)
        pipe = clone(pipeline).fit(train[TEXT_COLUMN], train["label"])
        fitted[name] = pipe
        predicted = pipe.predict(test[TEXT_COLUMN])
        proba = pipe.predict_proba(test[TEXT_COLUMN])
        metrics = holdout_metrics(test["label"], predicted, proba)
        balanced_pred = pipe.predict(balanced_test[TEXT_COLUMN])
        cv_row = cv_table[cv_table["modelo"] == name].iloc[0]
        comparison_rows.append(
            {
                "modelo": name,
                "tipo": "clásico",
                "macro_f1_cv": round(float(cv_row["macro_f1_cv"]), 4),
                "macro_f1_test": round(float(metrics["macro_f1"]), 4),
                "weighted_f1_test": round(float(metrics["weighted_f1"]), 4),
                "accuracy_test": round(float(metrics["accuracy"]), 4),
                "balanced_accuracy_test": round(float(metrics["balanced_accuracy"]), 4),
                "cohen_kappa_test": round(float(metrics["cohen_kappa"]), 4),
                "negative_pr_auc_test": round(float(metrics.get("negative_pr_auc", 0.0)), 4),
                "negative_roc_auc_test": round(float(metrics.get("negative_roc_auc", 0.0)), 4),
                "macro_f1_balanceado": round(
                    float(f1_score(balanced_test["label"], balanced_pred, average="macro", zero_division=0)),
                    4,
                ),
                "n_test": int(len(test)),
                "n_balanceado": int(len(balanced_test)),
                "tiempo_cv_seg": float(cv_row["tiempo_cv_seg"]),
            }
        )

    winner_pipeline = fitted[winner_name]
    winner_pred = winner_pipeline.predict(test[TEXT_COLUMN])
    winner_proba = winner_pipeline.predict_proba(test[TEXT_COLUMN])
    winner_metrics = holdout_metrics(test["label"], winner_pred, winner_proba)

    threshold_info = optimize_negative_threshold(
        models[winner_name], train[TEXT_COLUMN], train["label"], n_splits=n_splits
    )
    y_neg_true = (test["label"].to_numpy() == 0).astype(int)
    y_neg_pred = (winner_proba[:, 0] >= threshold_info["threshold"]).astype(int)
    winner_metrics.update(
        {
            "negative_precision": float(precision_score(y_neg_true, y_neg_pred, zero_division=0)),
            "negative_recall": float(recall_score(y_neg_true, y_neg_pred, zero_division=0)),
            "negative_f1": float(f1_score(y_neg_true, y_neg_pred, zero_division=0)),
            "negative_threshold": threshold_info["threshold"],
        }
    )

    logger.info("Evaluando la nota exacta como problema ordinal (5 clases)...")
    ordinal = evaluate_ordinal(train, test)

    logger.info("Extrayendo términos interpretables del modelo ganador...")
    terms = top_terms(winner_pipeline)

    logger.info("Cruzando sentimiento con logística y categoría...")
    master = load_master()
    results = build_results_frame(test, winner_pred, winner_proba, master)
    insights = build_business_insights(results, None)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "macro_f1_test", ascending=False
    ).reset_index(drop=True)

    transformer = None
    transformer_path = (artifacts_dir or ARTIFACTS_DIR) / "transformer_benchmark.json"
    if transformer_path.exists():
        with open(transformer_path, encoding="utf-8") as handle:
            transformer = json.load(handle)
        transformer_metrics = transformer.get("metrics", {})
        if transformer_metrics:
            comparison = pd.concat(
                [
                    comparison,
                    pd.DataFrame(
                        [
                            {
                                "modelo": f"Transformer BERTimbau (referencia zero-shot)",
                                "tipo": "transformer externo",
                                "macro_f1_cv": None,
                                "macro_f1_test": None,
                                "weighted_f1_test": transformer_metrics.get("weighted_f1"),
                                "accuracy_test": transformer_metrics.get("accuracy"),
                                "balanced_accuracy_test": transformer_metrics.get("balanced_accuracy"),
                                "cohen_kappa_test": transformer_metrics.get("cohen_kappa"),
                                "negative_pr_auc_test": None,
                                "negative_roc_auc_test": None,
                                "macro_f1_balanceado": transformer_metrics.get("macro_f1"),
                                "n_test": transformer_metrics.get("n"),
                                "n_balanceado": transformer_metrics.get("n"),
                                "tiempo_cv_seg": transformer.get("runtime_sec"),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    else:
        logger.info("Sin benchmark de transformer en %s (opcional).", transformer_path)

    manifest = export_artifacts(
        reviews=reviews,
        train=train,
        test=test,
        cv_table=cv_table,
        comparison=comparison,
        winner_name=winner_name,
        winner_pipeline=winner_pipeline,
        winner_metrics=winner_metrics,
        threshold_info=threshold_info,
        ordinal=ordinal,
        terms=terms,
        results=results,
        insights=insights,
        tuning_detail=tuning_detail,
        transformer=transformer,
        artifacts_dir=artifacts_dir,
    )

    logger.info(
        "Pipeline finalizado en %.1f s | macro-F1 test=%.4f | accuracy=%.4f | PR-AUC neg=%.4f",
        time.time() - started,
        winner_metrics["macro_f1"],
        winner_metrics["accuracy"],
        winner_metrics["negative_pr_auc"],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el análisis de sentimientos de reseñas Olist."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DIR,
        help="Directorio con olist_order_reviews_dataset.csv.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directorio de salida de artefactos.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Submuestra y menos folds para pruebas rápidas.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Usa C=0.5 sin búsqueda de hiperparámetros.",
    )
    args = parser.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        quick=args.quick,
        skip_tuning=args.skip_tuning,
    )


if __name__ == "__main__":
    main()
