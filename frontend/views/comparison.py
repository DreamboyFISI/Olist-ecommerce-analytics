import pandas as pd
import streamlit as st

from frontend.components import page_header, render_empty_state


def render(modules: list[dict]) -> None:
    page_header("Evaluación transversal", "Comparación de modelos", "Una matriz común para revisar el mejor resultado de cada módulo sin mezclar métricas de problemas diferentes.")
    rows = [{"Módulo": m["title"], "Estado": m["status"], "Ganador": m.get("winner") or "Pendiente", "Versión": m.get("version") or "—", "Generado": m.get("generated_at") or "—"} for m in modules]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if not any(module.get("manifest_available") for module in modules):
        render_empty_state("Aún no hay modelos para comparar", "Cada integrante debe publicar su manifest y sus métricas. Esta vista se completará progresivamente al integrar las ramas.")
    st.warning("MAE, Silhouette, lift, modularidad y F1 no son intercambiables. La comparación correcta se realiza contra el baseline y la métrica propia de cada problema.")
