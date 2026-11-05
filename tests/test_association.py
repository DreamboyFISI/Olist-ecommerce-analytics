"""
Pruebas unitarias para el modulo de Reglas de Asociacion - DataSmart Olist.
==========================================================================
Responsable: Modulo de Asociacion (rama feature/association)
Proyecto Integrador - Mineria de Datos - UNMSM FISI 2026

Pruebas con datos sinteticos minimos y validacion de artefactos:
  1. Construccion y filtrado correcto de canastas por pedido y cliente.
  2. Consistencia entre FP-Growth y ECLAT en soporte de itemsets.
  3. Generacion de reglas validando umbrales de Lift >= 1.0.
  4. Validacion formal de manifest.json contra docs/module_contract.schema.json.
  5. Existencia e integridad de artefactos precalculados.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jsonschema
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

from src.models.association import (
    construir_canastas,
    generate_rules,
    run_eclat,
)
ARTIFACTS_DIR = ROOT / "artifacts" / "association"
SCHEMA_PATH = ROOT / "docs" / "module_contract.schema.json"


class TestAssociationModule(unittest.TestCase):
    """Conjunto de pruebas unitarias para asociacion y venta cruzada."""

    @classmethod
    def setUpClass(cls) -> None:
        """Crea dataset sintetico transaccional para pruebas controladas."""
        cls.orders = pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_id": ["c1", "c2", "c3", "c4"],
            "order_status": ["delivered", "delivered", "delivered", "canceled"],
        })

        cls.items = pd.DataFrame({
            "order_id": ["o1", "o1", "o2", "o3", "o4"],
            "product_id": ["p1", "p2", "p1", "p3", "p1"],
        })

        cls.products = pd.DataFrame({
            "product_id": ["p1", "p2", "p3"],
            "product_category_name": ["beleza_saude", "cama_mesa_banho", "desconocido_original"],
        })

        cls.translations = pd.DataFrame({
            "product_category_name": ["beleza_saude", "cama_mesa_banho"],
            "product_category_name_english": ["health_beauty", "bed_bath_table"],
        })

        cls.customers = pd.DataFrame({
            "customer_id": ["c1", "c2", "c3", "c4"],
            "customer_unique_id": ["u1", "u1", "u2", "u3"],
        })

    def test_construir_canastas(self) -> None:
        """Verifica el filtrado de pedidos entregados y creacion de matrices binarias."""
        baskets = construir_canastas(
            orders=self.orders,
            items=self.items,
            products=self.products,
            translations=self.translations,
            customers=self.customers,
        )

        basket_order = baskets["order"]
        basket_cust = baskets["customer"]

        # Pedido cancelado no debe ser incluido
        self.assertNotIn("o4", basket_order.index)
        self.assertEqual(len(basket_order), 3)
        self.assertTrue(basket_order.dtypes.all())

        # Pedido multicategoria o1
        self.assertTrue(basket_order.loc["o1", "health_beauty"])
        self.assertTrue(basket_order.loc["o1", "bed_bath_table"])

        # Categoria sin traduccion rellenada con 'desconocido'
        self.assertIn("desconocido", basket_order.columns)
        self.assertTrue(basket_order.loc["o3", "desconocido"])

        # Cliente u1 acumula pedidos
        self.assertIn("u1", basket_cust.index)
        self.assertEqual(len(basket_cust), 2)

    def test_fpgrowth_eclat_consistency(self) -> None:
        """Valida que FP-Growth y ECLAT identifiquen los mismos soportes."""
        df_basket = pd.DataFrame({
            "item_A": [True, True, False, True, True],
            "item_B": [True, True, True, False, True],
            "item_C": [False, True, True, True, False],
        })

        fi_fp = fpgrowth(df_basket, min_support=0.4, use_colnames=True)
        fi_ec = run_eclat(df_basket, min_support=0.4)

        self.assertEqual(len(fi_fp), len(fi_ec))

        fp_dict = {frozenset(row["itemsets"]): round(row["support"], 4) for _, row in fi_fp.iterrows()}
        ec_dict = {frozenset(row["itemsets"]): round(row["support"], 4) for _, row in fi_ec.iterrows()}
        self.assertEqual(fp_dict, ec_dict)

    def test_generate_rules_lift_threshold(self) -> None:
        """Verifica que las reglas generadas cumplan con Lift >= 1.0."""
        df_basket = pd.DataFrame({
            "A": [True, True, True, False, False] * 20,
            "B": [True, True, True, False, False] * 20,
            "C": [False, False, True, True, True] * 20,
        })

        fi = fpgrowth(df_basket, min_support=0.1, use_colnames=True)
        rules = generate_rules(fi, metric="lift", min_threshold=1.0)

        self.assertFalse(rules.empty)
        self.assertTrue((rules["lift"] >= 1.0).all())
        expected_cols = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
        for col in expected_cols:
            self.assertIn(col, rules.columns)

    def test_manifest_schema_validation(self) -> None:
        """Valida formalmente el archivo manifest.json contra module_contract.schema.json."""
        manifest_path = ARTIFACTS_DIR / "manifest.json"
        self.assertTrue(manifest_path.exists(), f"No existe {manifest_path}")

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        with open(SCHEMA_PATH, encoding="utf-8") as sf:
            schema = json.load(sf)

        jsonschema.validate(instance=manifest, schema=schema)
        self.assertEqual(manifest["status"], "Completado")
        self.assertEqual(manifest["winner"], "FP-Growth - Canasta por Pedido")
        self.assertEqual(len(manifest["business_conclusions"]), 4)
        self.assertEqual(len(manifest["limitations"]), 4)

    def test_artifacts_integrity(self) -> None:
        """Verifica que todos los artefactos precalculados existan y no esten vacios."""
        results_path = ARTIFACTS_DIR / "results.csv"
        comp_path = ARTIFACTS_DIR / "algorithm_comparison.csv"
        chart_path = ARTIFACTS_DIR / "chart_data.json"
        model_path = ARTIFACTS_DIR / "model.joblib"

        self.assertTrue(results_path.exists() and results_path.stat().st_size > 0)
        self.assertTrue(comp_path.exists() and comp_path.stat().st_size > 0)
        self.assertTrue(chart_path.exists() and chart_path.stat().st_size > 0)
        self.assertTrue(model_path.exists() and model_path.stat().st_size > 0)

        df_results = pd.read_csv(results_path)
        self.assertGreaterEqual(len(df_results), 2)
        self.assertIn("lift", df_results.columns)

        with open(chart_path, encoding="utf-8") as f:
            chart_data = json.load(f)
        self.assertIn("top_10_rules", chart_data)
        self.assertIn("cooccurrence_matrix", chart_data)


if __name__ == "__main__":
    unittest.main()
