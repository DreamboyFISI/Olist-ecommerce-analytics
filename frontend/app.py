import sys
from pathlib import Path

import streamlit as st

# Streamlit añade la carpeta del script al path, no siempre la raíz del repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.components import inject_styles
from frontend.data_client import get_eda_summary, get_health, get_modules
from frontend.views import comparison, eda, modules, overview

st.set_page_config(page_title="DataSmart · Olist", page_icon="📦", layout="wide", initial_sidebar_state="expanded")
inject_styles()
api_connected, api_label = get_health()
eda_summary, eda_source = get_eda_summary()
module_registry = get_modules()
module_by_title = {module["title"]: module for module in module_registry}

with st.sidebar:
    st.markdown("## DataSmart")
    st.caption("Olist analytics")
    navigation = st.radio("Navegación principal", ["Resumen", "EDA y calidad", "Asociación", "Clustering", "Series temporales", "Grafos", "Regresión", "Clasificación", "Evaluación comparativa"], label_visibility="collapsed")
    st.divider()
    marker = "●" if api_connected else "○"
    st.caption(f"{marker} {api_label}")
    st.caption("Bondi · San Marino · Eden · Sinbad · Janna")

if navigation == "Resumen":
    overview.render(eda_summary, module_registry, api_label)
elif navigation == "EDA y calidad":
    eda.render(eda_summary, eda_source)
elif navigation == "Evaluación comparativa":
    comparison.render(module_registry)
else:
    modules.render(module_by_title[navigation])
