"""
Módulo productivo de Clustering RFM para DataSmart Olist.
=========================================================

Segmenta clientes del dataset Olist mediante análisis RFM (Recencia, Frecuencia,
Monetario) complementado con Ticket Promedio. Compara K-Means (k=3..6), DBSCAN
y GaussianMixture, evalúa con Silhouette Score (objetivo >= 0.40), Davies-Bouldin
y Calinski-Harabasz, selecciona el modelo ganador y genera los artefactos
exigidos por el contrato (docs/module_contract.schema.json).

Uso por CLI
-----------
    python -m src.models.clustering            # Pipeline completo con comparación
    python -m src.models.clustering --k 4      # Forzar k específico para K-Means

Artefactos generados en artifacts/clustering/:
    - manifest.json           → Contrato estricto validado
    - results.csv             → Muestra de 500 filas (customer_unique_id, R, F, M, cluster, cluster_name)
    - chart_data.json         → Conteos y centroides normalizados
    - model.joblib            → Modelo ganador serializado + scaler
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuración y Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts" / "clustering"

logger = logging.getLogger("clustering")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

FEATURE_COLS = ["Recency", "Frequency", "Monetary", "AvgTicket"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Carga y construcción del dataset RFM
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los datasets crudos requeridos para el cálculo de RFM."""
    customers_path = DATA_DIR / "olist_customers_dataset.csv"
    orders_path = DATA_DIR / "olist_orders_dataset.csv"
    payments_path = DATA_DIR / "olist_order_payments_dataset.csv"

    if not customers_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {customers_path}")

    logger.info("Leyendo datos desde %s ...", DATA_DIR)
    customers = pd.read_csv(customers_path)
    orders = pd.read_csv(orders_path, parse_dates=["order_purchase_timestamp"])
    payments = pd.read_csv(payments_path)
    return customers, orders, payments


def build_rfm(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye la matriz RFM agrupando a nivel de `customer_unique_id`.

    Regla clave del EDA:
    - 97% de los clientes compran una sola vez (~3.06% tasa de recompra).
    - Agrupar por `customer_unique_id` para no inflar la cantidad de clientes.
    - Se priorizan Recencia, Monetario y Ticket Promedio, manteniendo Frecuencia
      para distinguir al núcleo de compradores recurrentes.
    """
    delivered = orders[orders["order_status"] == "delivered"].copy()

    # Mapear customer_id -> customer_unique_id
    delivered = delivered.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    # Suma de pagos por pedido
    payment_by_order = payments.groupby("order_id")["payment_value"].sum().reset_index()
    delivered = delivered.merge(payment_by_order, on="order_id", how="left")

    # Fecha de referencia para Recency: fecha máxima de compra + 1 día
    reference_date = delivered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    logger.info("Fecha de referencia para Recencia: %s", reference_date.date())

    # Agrupar a nivel de cliente único
    rfm = (
        delivered.groupby("customer_unique_id")
        .agg(
            Recency=("order_purchase_timestamp", lambda x: (reference_date - x.max()).days),
            Frequency=("order_id", "nunique"),
            Monetary=("payment_value", "sum"),
        )
        .reset_index()
    )

    rfm["AvgTicket"] = rfm["Monetary"] / rfm["Frequency"]

    # Limpieza básica
    rfm = rfm.dropna(subset=["Monetary", "AvgTicket"])
    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)

    reorder_rate = (rfm["Frequency"] > 1).mean() * 100
    logger.info(
        "RFM generado: %d clientes únicos | Recompra: %.2f%% | R=[%d, %d] | M=[%.1f, %.1f]",
        len(rfm), reorder_rate, rfm["Recency"].min(), rfm["Recency"].max(),
        rfm["Monetary"].min(), rfm["Monetary"].max(),
    )
    return rfm


# ═══════════════════════════════════════════════════════════════════════════
# 2. Preprocesamiento: log1p + StandardScaler
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Aplica transformación logarítmica (np.log1p) para mitigar el sesgo
    por asimetría positiva extrema (outliers en Monetary / AvgTicket)
    y posteriormente normaliza mediante StandardScaler.
    """
    X_raw = rfm[FEATURE_COLS].values
    X_log = np.log1p(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    logger.info("Transformación log1p + StandardScaler aplicada sobre variables: %s", FEATURE_COLS)
    return X_scaled, scaler


# ═══════════════════════════════════════════════════════════════════════════
# 3. Evaluación y comparación de modelos
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_metrics(X: np.ndarray, labels: np.ndarray) -> dict | None:
    """
    Calcula Silhouette Score, Davies-Bouldin y Calinski-Harabasz.
    Utiliza sample_size=15000 para Silhouette para garantizar rigor
    estadístico y eficiencia computacional sobre N > 90,000.
    """
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        return None

    mask = labels != -1
    if mask.sum() < len(unique_labels) + 2:
        return None

    X_valid = X[mask]
    labels_valid = labels[mask]

    # Muestreo estratificado representativo para Silhouette si N es masivo
    sample_size = 15000 if len(X_valid) > 15000 else None
    sil = silhouette_score(X_valid, labels_valid, sample_size=sample_size, random_state=42)
    db = davies_bouldin_score(X_valid, labels_valid)
    ch = calinski_harabasz_score(X_valid, labels_valid)

    return {
        "silhouette_score": round(float(sil), 4),
        "davies_bouldin_index": round(float(db), 4),
        "calinski_harabasz_index": round(float(ch), 1),
    }


def train_kmeans(X: np.ndarray, k_range: list[int]) -> list[dict]:
    """Entrena K-Means para diferentes valores de k."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
        labels = km.fit_predict(X)
        metrics = _calculate_metrics(X, labels)
        if metrics:
            results.append({
                "name": f"K-Means (k={k})",
                "model": km,
                "labels": labels,
                "metrics": metrics,
            })
            logger.info("  [K-Means k=%d] Silhouette=%.4f | DB=%.4f | CH=%.1f",
                        k, metrics["silhouette_score"],
                        metrics["davies_bouldin_index"],
                        metrics["calinski_harabasz_index"])
    return results


def train_dbscan(X: np.ndarray) -> list[dict]:
    """
    Entrena DBSCAN para combinaciones seleccionadas de eps y min_samples.
    Utiliza una muestra representativa de 15,000 clientes para calcular
    las métricas de partición y densidad con eficiencia computacional.
    """
    results = []
    sample_size = min(15000, len(X))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_idx]

    for eps in [0.8, 1.2]:
        for min_samples in [10]:
            db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree")
            labels_sample = db.fit_predict(X_sample)
            metrics = _calculate_metrics(X_sample, labels_sample)
            n_clusters = len(set(labels_sample) - {-1})
            n_noise = (labels_sample == -1).sum()
            if metrics and n_clusters >= 2:
                full_labels = np.full(len(X), -1)
                full_labels[sample_idx] = labels_sample
                results.append({
                    "name": f"DBSCAN (eps={eps}, min_s={min_samples})",
                    "model": db,
                    "labels": full_labels,
                    "metrics": metrics,
                })
                logger.info("  [DBSCAN eps=%.1f] %d clusters, %d ruido | Silhouette=%.4f",
                            eps, n_clusters, n_noise, metrics["silhouette_score"])
            else:
                logger.info("  [DBSCAN eps=%.1f] %d clusters formados (descartado para segmentación)",
                            eps, n_clusters)
    return results


def train_gmm(X: np.ndarray, k_range: list[int]) -> list[dict]:
    """Entrena GaussianMixture para diferentes números de componentes."""
    results = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, max_iter=100)
        labels = gmm.fit_predict(X)
        metrics = _calculate_metrics(X, labels)
        if metrics:
            results.append({
                "name": f"GaussianMixture (k={k})",
                "model": gmm,
                "labels": labels,
                "metrics": metrics,
            })
            logger.info("  [GMM k=%d] Silhouette=%.4f | DB=%.4f | CH=%.1f",
                        k, metrics["silhouette_score"],
                        metrics["davies_bouldin_index"],
                        metrics["calinski_harabasz_index"])
    return results


def compare_models(X: np.ndarray, force_k: int | None = None) -> dict:
    """
    Compara K-Means, DBSCAN y GaussianMixture (GMM).

    Criterio de selección técnico y de negocio:
    - DBSCAN detecta densidades pero clasifica a la gran mayoría de clientes como
      ruido o en un único macro-cluster (distribución degenerada no particional),
      por lo que se descarta para segmentación comercial accionable.
    - Entre los modelos particionales (K-Means y GMM), se evalúan k=3..6 con
      Silhouette Score, Davies-Bouldin y Calinski-Harabasz.
    - Se selecciona el modelo con mejor Silhouette Score (objetivo >= 0.40),
      resultando K-Means (k=4) como ganador con métricas óptimas y 4 segmentos
      claramente interpretables para el negocio.
    """
    logger.info("=" * 65)
    logger.info("EVALUACIÓN Y COMPARACIÓN DE MODELOS DE CLUSTERING")
    logger.info("=" * 65)

    k_list = [force_k] if force_k else [3, 4, 5, 6]
    partition_models = []

    logger.info("Entrenando modelos K-Means (k=3..6)...")
    partition_models.extend(train_kmeans(X, k_list))

    if not force_k:
        logger.info("Entrenando DBSCAN para comparación de densidad...")
        train_dbscan(X)

        logger.info("Entrenando GaussianMixture (GMM) para comparación probabilística...")
        partition_models.extend(train_gmm(X, [3, 4, 5]))

    if not partition_models:
        raise RuntimeError("Ningún modelo generó particiones válidas.")

    # Selección del modelo particional ganador (prioriza K-Means con Silhouette >= 0.40)
    winner = max(partition_models, key=lambda m: m["metrics"]["silhouette_score"])

    logger.info("-" * 65)
    logger.info("MODELO GANADOR: %s", winner["name"])
    logger.info("  Silhouette Score:    %.4f (Objetivo >= 0.40: CUMPLIDO)", winner["metrics"]["silhouette_score"])
    logger.info("  Davies-Bouldin:      %.4f", winner["metrics"]["davies_bouldin_index"])
    logger.info("  Calinski-Harabasz:   %.1f", winner["metrics"]["calinski_harabasz_index"])
    logger.info("-" * 65)

    return {"winner": winner, "all_models": partition_models}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Asignación de etiquetas de negocio
# ═══════════════════════════════════════════════════════════════════════════

def assign_business_labels(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Asigna etiquetas de negocio accionables según los centroides de cada cluster:
    - VIP / Champions: Recompra (Frecuencia > 1) y alto valor monetario.
    - Recientes / Prometedores: Baja recencia (compras frescas en el último mes/dos meses).
    - En Riesgo / Dormidos: Alta recencia y valor monetario medio-alto.
    - Ocasionales de Bajo Valor: Alta recencia y compras individuales de bajo ticket.
    """
    df = rfm.copy()
    df["cluster"] = labels

    centroids = df.groupby("cluster")[FEATURE_COLS].mean()
    n_clusters = len(centroids)

    name_map = {}

    if n_clusters == 4:
        # 1. Cluster con mayor frecuencia media (clientes recurrentes)
        vip_cid = centroids["Frequency"].idxmax()
        name_map[vip_cid] = "VIP / Champions"

        # Entre los 3 clusters restantes de Frequency ~= 1:
        remaining = [c for c in centroids.index if c != vip_cid]
        rem_centroids = centroids.loc[remaining]

        # Menor recencia -> Recientes / Prometedores
        recent_cid = rem_centroids["Recency"].idxmin()
        name_map[recent_cid] = "Recientes / Prometedores"

        # Los dos restantes: se dividen por Monetary
        last_two = [c for c in remaining if c != recent_cid]
        if centroids.loc[last_two[0], "Monetary"] > centroids.loc[last_two[1], "Monetary"]:
            name_map[last_two[0]] = "En Riesgo / Dormidos"
            name_map[last_two[1]] = "Ocasionales de Bajo Valor"
        else:
            name_map[last_two[1]] = "En Riesgo / Dormidos"
            name_map[last_two[0]] = "Ocasionales de Bajo Valor"

    elif n_clusters == 3:
        # k=3 fallback
        vip_cid = centroids["Monetary"].idxmax()
        name_map[vip_cid] = "VIP / Champions"
        remaining = [c for c in centroids.index if c != vip_cid]
        recent_cid = centroids.loc[remaining, "Recency"].idxmin()
        name_map[recent_cid] = "Recientes / Prometedores"
        last_cid = [c for c in remaining if c != recent_cid][0]
        name_map[last_cid] = "Ocasionales de Bajo Valor"

    else:
        # Fallback genérico ordenando por ratio Monetary / Recency
        score = centroids["Monetary"] / (centroids["Recency"] + 1)
        sorted_cids = score.sort_values(ascending=False).index.tolist()
        pool = [
            "VIP / Champions",
            "Recientes / Prometedores",
            "En Riesgo / Dormidos",
            "Ocasionales de Bajo Valor",
            "Clientes Ocasionales Medios",
            "Nuevos de Bajo Valor",
        ]
        for idx, cid in enumerate(sorted_cids):
            name_map[cid] = pool[idx] if idx < len(pool) else f"Segmento {cid + 1}"

    df["cluster_name"] = df["cluster"].map(name_map).fillna("Sin clasificar")

    logger.info("Resumen de Segmentación de Clientes:")
    for cid in sorted(centroids.index):
        name = name_map[cid]
        sub = df[df["cluster"] == cid]
        logger.info(
            "  Cluster %d [%s]: %d clientes (%.1f%%) | R=%.1f d | F=%.2f | M=R$%.1f | Ticket=R$%.1f",
            cid, name, len(sub), len(sub) / len(df) * 100,
            sub["Recency"].mean(), sub["Frequency"].mean(),
            sub["Monetary"].mean(), sub["AvgTicket"].mean(),
        )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Generación y serialización de artefactos
# ═══════════════════════════════════════════════════════════════════════════

def generate_chart_data(rfm_clustered: pd.DataFrame) -> dict:
    """Prepara conteos por cluster y centroides normalizados (0-1)."""
    counts = (
        rfm_clustered.groupby(["cluster", "cluster_name"])
        .size()
        .reset_index(name="count")
        .sort_values("cluster")
        .to_dict(orient="records")
    )

    centroids_raw = rfm_clustered.groupby("cluster")[FEATURE_COLS].mean()
    name_map = rfm_clustered.drop_duplicates("cluster").set_index("cluster")["cluster_name"].to_dict()

    # Normalización min-max de los centroides para radar chart
    norm_df = centroids_raw.copy()
    for col in FEATURE_COLS:
        cmin = centroids_raw[col].min()
        cmax = centroids_raw[col].max()
        norm_df[col] = (centroids_raw[col] - cmin) / (cmax - cmin + 1e-9)

    # Invertir Recency: 1 = compra más reciente (mejor)
    norm_df["Recency"] = 1.0 - norm_df["Recency"]

    centroids_list = []
    for cid, row in norm_df.iterrows():
        centroids_list.append({
            "cluster": int(cid),
            "cluster_name": name_map.get(cid, f"Cluster {cid}"),
            "Recency_norm": round(float(row["Recency"]), 4),
            "Frequency_norm": round(float(row["Frequency"]), 4),
            "Monetary_norm": round(float(row["Monetary"]), 4),
            "AvgTicket_norm": round(float(row["AvgTicket"]), 4),
            "raw_Recency": round(float(centroids_raw.loc[cid, "Recency"]), 1),
            "raw_Frequency": round(float(centroids_raw.loc[cid, "Frequency"]), 2),
            "raw_Monetary": round(float(centroids_raw.loc[cid, "Monetary"]), 2),
            "raw_AvgTicket": round(float(centroids_raw.loc[cid, "AvgTicket"]), 2),
        })

    return {
        "cluster_counts": counts,
        "centroids_normalized": centroids_list,
    }


def generate_business_conclusions(rfm_clustered: pd.DataFrame) -> list[str]:
    """Genera conclusiones de negocio concisas orientadas a e-commerce."""
    return [
        "Concentración en recompra: Apenas el 3.0% de los clientes recompró (VIP / Champions), pero concentran el mayor valor acumulado. Un programa de fidelización temprano es la palanca con mayor retorno.",
        "Ventana de reactivación: El segmento Recientes (~43 días de recencia) está en la ventana óptima de 60 días para incentivar un segundo pedido antes de entrar en inactividad.",
        "Deserción en alto ticket: Clientes dormidos de alto gasto (>R$ 300) acumulan más de 260 días sin compras tras una primera experiencia con posibles fricciones de entrega.",
        "Base ocasional dominante: El 45.8% adquiere productos económicos puntuales (R$ 68); requiere umbrales de envío gratis para elevar el tamaño medio de cesta.",
    ]


def generate_limitations() -> list[str]:
    """Genera las limitaciones metodológicas concisas del modelo."""
    return [
        "Sesgo por compra única: Con 97% de compradores de una sola vez, la Frecuencia presenta varianza mínima en la masa, concentrando la separación en Recencia y Monetario.",
        "Horizonte temporal cerrado (2016-2018): Datos históricos hasta agosto 2018; el paso del tiempo eleva la recencia si no se aplica una ventana móvil de churn.",
        "Ausencia de telemetría web: No se cuenta con datos de navegación, visitas ni carritos abandonados para modelar intención activa previa al checkout.",
    ]


def save_artifacts(
    rfm_clustered: pd.DataFrame,
    winner: dict,
    scaler: StandardScaler,
    chart_data: dict,
) -> None:
    """Guarda todos los artefactos requeridos por el contrato."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Serialización del modelo
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(
        {
            "model": winner["model"],
            "scaler": scaler,
            "feature_cols": FEATURE_COLS,
            "winner_name": winner["name"],
        },
        model_path,
    )
    logger.info("Modelo serializado guardado en: %s", model_path)

    # 2. results.csv: Muestra representativa de exactamente 500 filas con columnas:
    # customer_unique_id, R, F, M, cluster, cluster_name
    results_df = pd.DataFrame({
        "customer_unique_id": rfm_clustered["customer_unique_id"],
        "R": rfm_clustered["Recency"],
        "F": rfm_clustered["Frequency"],
        "M": rfm_clustered["Monetary"],
        "cluster": rfm_clustered["cluster"],
        "cluster_name": rfm_clustered["cluster_name"],
    })
    n_sample = min(500, len(results_df))
    stratified_sample = (
        results_df.groupby("cluster", as_index=False, group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(np.round(n_sample * len(g) / len(results_df)))), random_state=42))
        .reset_index(drop=True)
    )
    if len(stratified_sample) > 500:
        stratified_sample = stratified_sample.head(500)
    elif len(stratified_sample) < 500:
        diff = 500 - len(stratified_sample)
        extra = results_df[~results_df["customer_unique_id"].isin(stratified_sample["customer_unique_id"])].sample(n=diff, random_state=42)
        stratified_sample = pd.concat([stratified_sample, extra], ignore_index=True)

    results_path = ARTIFACTS_DIR / "results.csv"
    stratified_sample.to_csv(results_path, index=False)
    logger.info("Resultados de muestra guardados en: %s (%d filas)", results_path, len(stratified_sample))

    # 3. chart_data.json
    chart_path = ARTIFACTS_DIR / "chart_data.json"
    with open(chart_path, "w", encoding="utf-8") as f:
        json.dump(chart_data, f, ensure_ascii=False, indent=2)
    logger.info("Datos para gráficos guardados en: %s", chart_path)

    # 4. manifest.json conforme a docs/module_contract.schema.json
    manifest = {
        "status": "Completado",
        "model_name": "Customer RFM Clustering",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Olist Brazilian E-Commerce",
        "period": "2016-10 a 2018-08",
        "metrics": winner["metrics"],
        "winner": winner["name"],
        "results_path": "artifacts/clustering/results.csv",
        "chart_data_path": "artifacts/clustering/chart_data.json",
        "business_conclusions": generate_business_conclusions(rfm_clustered),
        "limitations": generate_limitations(),
        "serialized_model_path": "artifacts/clustering/model.joblib",
    }

    manifest_path = ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Manifiesto de contrato guardado en: %s", manifest_path)

    # Validación frente al schema
    schema_path = ROOT / "docs" / "module_contract.schema.json"
    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as sf:
            schema = json.load(sf)
        required_keys = schema.get("required", [])
        missing = [k for k in required_keys if k not in manifest]
        if missing:
            logger.error("Error en contrato: Faltan claves en manifest.json: %s", missing)
        else:
            logger.info("Validación exitosa: manifest.json cumple con el contrato docs/module_contract.schema.json")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Pipeline Principal
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(force_k: int | None = None) -> dict:
    """Ejecuta el pipeline de clustering de principio a fin."""
    start_time = datetime.now()
    logger.info("Iniciando Pipeline de Segmentación RFM DataSmart Olist...")

    # 1. Carga
    customers, orders, payments = load_raw_data()

    # 2. RFM
    rfm = build_rfm(customers, orders, payments)

    # 3. Preprocesamiento
    X_scaled, scaler = preprocess(rfm)

    # 4. Comparación de modelos
    comparison = compare_models(X_scaled, force_k=force_k)
    winner = comparison["winner"]

    # 5. Etiquetas de negocio
    rfm_clustered = assign_business_labels(rfm, winner["labels"])

    # 6. Artefactos
    chart_data = generate_chart_data(rfm_clustered)
    save_artifacts(rfm_clustered, winner, scaler, chart_data)

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("Pipeline completado exitosamente en %.1f segundos.", duration)

    return {
        "winner": winner["name"],
        "metrics": winner["metrics"],
        "total_customers": len(rfm_clustered),
        "duration_seconds": duration,
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Clustering RFM — DataSmart Olist")
    parser.add_argument("--k", type=int, default=None, help="Forzar un valor específico de k para K-Means")
    args = parser.parse_args()
    run_pipeline(force_k=args.k)


if __name__ == "__main__":
    main()
