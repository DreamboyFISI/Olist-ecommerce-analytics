"""Entrada de línea de comandos del monitoreo: python -m src.monitoring [--retrain] [--dry-run]."""

from __future__ import annotations

import argparse
import json
import logging

from src.monitoring.monitor import run
from src.monitoring.retrain import execute_plan, plan_retraining


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoreo de modelos DataSmart: desempeño, deriva, versionado y auditoría.")
    parser.add_argument("--no-write", action="store_true", help="No escribir reporte, registro ni bitácora.")
    parser.add_argument("--retrain", action="store_true", help="Reentrenar los módulos en Alerta.")
    parser.add_argument("--include-watch", action="store_true", help="Incluir también los módulos en Vigilar.")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar el plan de reentrenamiento sin ejecutarlo.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = run(write=not args.no_write)
    print(json.dumps({"generated_at": report["generated_at"], "resumen": report["resumen"],
                      "modulos": {m["modulo"]: m["estado"] for m in report["modulos"]}}, ensure_ascii=False, indent=2))

    if args.retrain or args.dry_run:
        plan = plan_retraining(report, include_watch=args.include_watch)
        if not plan:
            print("Ningún módulo requiere reentrenamiento.")
            return
        resultados = execute_plan(plan, dry_run=args.dry_run or not args.retrain)
        print(json.dumps(resultados, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
