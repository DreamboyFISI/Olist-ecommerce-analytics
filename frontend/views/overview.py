import pandas as pd
import streamlit as st

from frontend.components import page_header, section_intro, status_badge


def render(eda: dict, modules: list[dict], api_label: str) -> None:
    page_header("DataSmart · Olist", "Decisiones más claras para cada pedido", "Un espacio único para explorar la calidad de los datos, descubrir patrones y comparar los modelos del equipo.")
    cols = st.columns(4)
    cols[0].metric("Pedidos analizados", f"{eda['tablas']['orders']['filas']:,}")
    cols[1].metric("Ticket promedio", f"R$ {eda['ticket_promedio_BRL']:,.2f}")
    cols[2].metric("Reseñas negativas", f"{eda['review_negativa_%(<=2)']:.2f}%")
    cols[3].metric("Entregas tardías", f"{eda['%_entregas_tardias']:.2f}%")
    st.markdown(f"<div class='module-note'><strong>Fuente de datos:</strong> {api_label}. El dashboard puede seguir mostrando el EDA en modo local si la API no está activa.</div>", unsafe_allow_html=True)
    section_intro("Ruta analítica", "El EDA está completo; los seis módulos quedan aislados para el trabajo por ramas.")
    rows = [{"Módulo": m["title"], "Pregunta de negocio": m["question"], "Rama": m["branch"], "Estado": m["status"]} for m in modules]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    ready = sum(m["status"].lower() in {"completado", "entrenado", "listo"} for m in modules)
    st.markdown(f"{status_badge('EDA completado')} &nbsp; {status_badge(f'{ready} de {len(modules)} módulos listos')}", unsafe_allow_html=True)
