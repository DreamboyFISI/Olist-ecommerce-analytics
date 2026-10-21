import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDA_SUMMARY_PATH = ROOT / "docs" / "hallazgos_eda.json"
ARTIFACTS_DIR = ROOT / "artifacts"

MODULES = (
    {"slug": "association", "title": "Asociación", "branch": "feature/association", "question": "¿Qué productos o categorías se compran conjuntamente?", "expected_models": ["Apriori", "FP-Growth", "ECLAT"]},
    {"slug": "clustering", "title": "Clustering", "branch": "feature/clustering", "question": "¿Qué segmentos de clientes existen y cuáles aportan mayor valor?", "expected_models": ["K-Means", "DBSCAN"]},
    {"slug": "timeseries", "title": "Series temporales", "branch": "feature/timeseries", "question": "¿Cómo evolucionarán las ventas en los próximos periodos?", "expected_models": ["SARIMA", "Prophet"]},
    {"slug": "graphs", "title": "Grafos", "branch": "feature/graphs", "question": "¿Qué productos, categorías o vendedores están más conectados?", "expected_models": ["NetworkX", "Louvain"]},
    {"slug": "regression", "title": "Regresión", "branch": "feature/regression", "question": "¿Cuántos días tardará en entregarse un pedido?", "expected_models": ["Regresión lineal", "Random Forest", "XGBoost"]},
    {"slug": "classification", "title": "Clasificación", "branch": "feature/classification", "question": "¿Qué pedidos tienen riesgo de recibir una reseña negativa?", "expected_models": ["Random Forest", "XGBoost", "SVM", "Naive Bayes"]},
)


def _read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_eda_summary() -> dict:
    return _read_json(EDA_SUMMARY_PATH)


def load_module_registry() -> list[dict]:
    registry = []
    for definition in MODULES:
        item = deepcopy(definition)
        manifest_path = ARTIFACTS_DIR / item["slug"] / "manifest.json"
        item.update({"status": "Pendiente", "version": None, "generated_at": None, "winner": None, "metrics": {}, "manifest_available": False})
        if manifest_path.exists():
            item.update(_read_json(manifest_path))
            item["manifest_available"] = True
        registry.append(item)
    return registry
