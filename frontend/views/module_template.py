import pandas as pd
import streamlit as st

from frontend.components import page_header, render_empty_state, section_intro, status_badge

CONTRACT_FIELDS = [
    "Estado del entrenamiento", "Nombre y versión del modelo", "Fecha de generación",
    "Dataset o periodo utilizado", "Métricas de entrenamiento y prueba", "Modelo ganador",
    "Tabla pequeña de resultados o predicciones", "Datos preparados para las gráficas",
    "Conclusiones de negocio", "Limitaciones", "Modelo serializado cuando corresponda",
]


def render_module(module: dict, expected_output: str) -> None:
    page_header("Módulo de modelamiento", module["title"], module["question"])
    st.markdown(f"{status_badge(module['status'])} &nbsp; <code>{module['branch']}</code>", unsafe_allow_html=True)
    if not module.get("manifest_available"):
        render_empty_state("Módulo preparado para su rama", "La navegación, el contrato y el espacio visual ya están listos. Esta pantalla se activará al agregar " + f"artifacts/{module['slug']}/manifest.json y los resultados reales.")
    else:
        cols = st.columns(3)
        cols[0].metric("Estado", module.get("status", "Sin informar"))
        cols[1].metric("Versión", module.get("version") or "Sin informar")
        cols[2].metric("Modelo ganador", module.get("winner") or "Sin informar")
        if module.get("metrics"):
            st.dataframe(pd.DataFrame([module["metrics"]]), hide_index=True, use_container_width=True)
    section_intro("Salida esperada", expected_output)
    st.write("Modelos o técnicas previstos: " + ", ".join(module["expected_models"]) + ".")
    with st.expander("Contrato común del módulo", expanded=True):
        for field in CONTRACT_FIELDS:
            st.checkbox(field, value=bool(module.get("manifest_available")), disabled=True, key=f"{module['slug']}-{field}")
    st.info("No entrenes el modelo desde esta pantalla. Publica artefactos reproducibles y deja que Streamlit se concentre en consultar y visualizar.")
