"""Página del módulo de Clasificación: riesgo de reseña negativa.

Lee los artefactos de ``artifacts/classification`` a través de la API (o en
modo local) y ofrece dos pestañas: evaluar un pedido con el modelo y revisar
cómo se construyó y qué tan bien funciona. No entrena nada en Streamlit.
"""

from datetime import datetime, time, timedelta
from urllib.error import HTTPError, URLError

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import PALETTE, page_header, section_intro, status_badge, style_figure
from frontend.data_client import explain_classification, get_classification_results, predict_classification
from frontend.views.module_template import render_module

TERRACOTA = "#E07B54"

STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia", "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão", "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul", "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
    "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo",
    "SE": "Sergipe", "TO": "Tocantins",
}
PAYMENT_LABELS = {"credit_card": "Tarjeta de crédito", "boleto": "Boleto", "voucher": "Vale",
                  "debit_card": "Tarjeta de débito", "not_defined": "No definido"}
MODEL_LABELS = {
    "Baseline (prior)": "Referencia (adivinar la proporción)",
}


def _state_label(code: str) -> str:
    return f"{code} · {STATE_NAMES.get(code, code)}"


def _category_label(code: str) -> str:
    return code.replace("_", " ").capitalize()


def _pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


# ---------------------------------------------------------------------------
# Pestaña 1: evaluar un pedido
# ---------------------------------------------------------------------------
def _render_prediction(results: dict) -> None:
    section_intro(
        "Evaluar un pedido",
        "Describe el pedido tal como se ve al momento de enviar la encuesta de satisfacción y el modelo "
        "estima la probabilidad de que el cliente deje 1 o 2 estrellas.",
    )
    states = results.get("states") or sorted(STATE_NAMES)
    categories = results.get("categories") or ["sin_categoria"]
    payment_types = results.get("payment_types") or list(PAYMENT_LABELS)
    threshold = results["manifest"]["metrics"].get("umbral", 0.5)

    with st.form("classification-prediction-form"):
        st.markdown("#### Pedido y pago")
        c = st.columns(4)
        n_items = c[0].number_input("Ítems en el pedido", 1, 50, 1)
        n_products = c[1].number_input("Productos distintos", 1, 50, 1)
        n_sellers = c[2].number_input("Vendedores distintos", 1, 10, 1)
        n_categories = c[3].number_input("Categorías distintas", 1, 10, 1)
        c = st.columns(4)
        price_total = c[0].number_input("Precio total (R$)", 1.0, 20_000.0, 120.0, step=10.0)
        freight_total = c[1].number_input("Flete total (R$)", 0.0, 2_000.0, 18.0, step=1.0)
        installments = c[2].number_input("Cuotas", 1, 24, 1)
        payment_type = c[3].selectbox("Medio de pago", payment_types,
                                      format_func=lambda v: PAYMENT_LABELS.get(v, v))
        c = st.columns(3)
        default_cat = categories.index("bed_bath_table") if "bed_bath_table" in categories else 0
        main_category = c[0].selectbox("Categoría principal", categories, index=default_cat,
                                       format_func=_category_label)
        seller_state = c[1].selectbox("Estado del vendedor", states, index=states.index("SP") if "SP" in states else 0,
                                      format_func=_state_label)
        customer_state = c[2].selectbox("Estado del cliente", states, index=states.index("RJ") if "RJ" in states else 0,
                                        format_func=_state_label)

        with st.expander("Detalles del producto (opcionales)"):
            d = st.columns(4)
            distance_km = d[0].number_input("Distancia vendedor–cliente (km)", 0.0, 5_000.0, 400.0, step=10.0)
            weight_total_g = d[1].number_input("Peso total (g)", 0.0, 100_000.0, 1_500.0, step=100.0)
            volume_total_cm3 = d[2].number_input("Volumen total (cm³)", 0.0, 1_000_000.0, 12_000.0, step=500.0)
            photos_mean = d[3].number_input("Fotos por producto", 0.0, 20.0, 2.0, step=1.0)
            d = st.columns(2)
            description_length_mean = d[0].number_input("Largo de descripción (caracteres)", 0.0, 5_000.0, 600.0, step=50.0)
            approval_hours = d[1].number_input("Horas hasta aprobar el pago", 0.0, 240.0, 0.5, step=0.5)

        st.markdown("#### Fechas y estado logístico al enviar la encuesta")
        f = st.columns(4)
        purchase_date = f[0].date_input("Fecha de compra", value=datetime(2018, 7, 10).date())
        purchase_time = f[1].time_input("Hora de compra", value=time(14, 30))
        estimated_delivery_date = f[2].date_input("Fecha prometida al cliente",
                                                  value=datetime(2018, 8, 3).date())
        shipping_limit_date = f[3].date_input("Límite para despachar", value=datetime(2018, 7, 16).date())
        g = st.columns(3)
        delivery_status = g[0].radio("Estado al enviar la encuesta", ["Entregado", "Aún no entregado"],
                                     horizontal=True)
        delivered_carrier_date = g[1].date_input("Fecha de despacho (si ocurrió)",
                                                 value=datetime(2018, 7, 13).date())
        delivered_customer_date = g[2].date_input("Fecha de entrega al cliente",
                                                  value=datetime(2018, 7, 24).date())
        survey_date = st.date_input("Fecha de envío de la encuesta", value=datetime(2018, 7, 25).date(),
                                    help="Olist envía la encuesta tras la entrega o cuando vence la promesa.")
        submitted = st.form_submit_button("Estimar riesgo de reseña negativa", type="primary",
                                          use_container_width=True)

    if not submitted:
        st.caption("Completa los datos y presiona el botón. Los valores por defecto describen un pedido típico.")
        return

    delivered = delivery_status == "Entregado"
    payload = {
        "purchase_timestamp": datetime.combine(purchase_date, purchase_time).isoformat(),
        "survey_timestamp": datetime.combine(survey_date, time(0, 0)).isoformat(),
        "estimated_delivery_date": datetime.combine(estimated_delivery_date, time(0, 0)).isoformat(),
        "shipping_limit_date": datetime.combine(shipping_limit_date, time(23, 59)).isoformat(),
        "delivered_carrier_date": datetime.combine(delivered_carrier_date, time(12, 0)).isoformat(),
        "delivered_customer_date": datetime.combine(delivered_customer_date, time(12, 0)).isoformat() if delivered else None,
        "approval_hours": float(approval_hours),
        "n_items": int(n_items), "n_products": int(n_products), "n_sellers": int(n_sellers),
        "n_categories": int(n_categories),
        "price_total": float(price_total), "freight_total": float(freight_total),
        "weight_total_g": float(weight_total_g), "volume_total_cm3": float(volume_total_cm3),
        "photos_mean": float(photos_mean), "description_length_mean": float(description_length_mean),
        "installments": int(installments), "n_payment_types": 1, "payment_type": payment_type,
        "customer_state": customer_state, "seller_state": seller_state,
        "distance_km": float(distance_km), "main_category": main_category,
    }
    try:
        prediction = predict_classification(payload)
    except (HTTPError, URLError, TimeoutError, ValueError, FileNotFoundError) as exc:
        st.error(f"No se pudo calcular el riesgo: {exc}")
        return

    prob = prediction["probabilidad"]
    nivel = prediction["nivel_riesgo"]
    contexto = prediction["contexto"]
    color = {"Alto": TERRACOTA, "Medio": PALETTE["san_marino"], "Bajo": PALETTE["bondi"]}[nivel]

    st.markdown("#### Resultado")
    r = st.columns(4)
    r[0].metric("Probabilidad de reseña negativa", _pct(prob))
    r[1].metric("Nivel de riesgo", nivel)
    veces = prediction.get("veces_sobre_prevalencia")
    r[2].metric("Frente a un pedido promedio", f"{veces:.1f}×" if veces else "—",
                help=f"Un pedido promedio del periodo de prueba tiene {_pct(prediction['prevalencia_referencia'])} de negativas.")
    r[3].metric("Estado al enviar la encuesta", contexto["estado_en_encuesta"],
                f"{contexto['dias_retraso']:+.1f} días vs promesa", delta_color="inverse")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"color": PALETTE["ink"]}},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": color},
               "steps": [{"range": [0, threshold * 50], "color": "rgba(7,153,182,0.15)"},
                         {"range": [threshold * 50, threshold * 100], "color": "rgba(74,110,176,0.18)"},
                         {"range": [threshold * 100, 100], "color": "rgba(224,123,84,0.18)"}],
               "threshold": {"line": {"color": PALETTE["eden"], "width": 3}, "value": threshold * 100}},
    ))
    st.plotly_chart(style_figure(gauge, f"Umbral de alerta: {threshold * 100:.0f}%", height=300),
                    use_container_width=True, config={"displaylogo": False})

    decil = prediction.get("decil_referencia")
    if nivel == "Alto":
        mensaje = ("**Conviene actuar antes de que el cliente califique:** contactar al cliente, confirmar el "
                   "estado del envío o compensar. ")
    elif nivel == "Medio":
        mensaje = "**Vale la pena vigilar el pedido:** está por encima del promedio pero no supera el umbral de alerta. "
    else:
        mensaje = "**Pedido de bajo riesgo:** no requiere acción especial. "
    if decil and decil.get("tasa_negativas") is not None:
        mensaje += (f"En el periodo de prueba, los pedidos con una probabilidad similar (decil {decil['decil']} de "
                    f"riesgo) terminaron en reseña negativa el {_pct(decil['tasa_negativas'])} de las veces.")
    (st.warning if nivel == "Alto" else st.info)(mensaje)

    st.markdown("#### Por qué")
    try:
        explicacion = explain_classification(payload)
    except (HTTPError, URLError, TimeoutError, ValueError, FileNotFoundError) as exc:
        explicacion = None
        st.caption(f"No se pudo generar la explicación: {exc}")
    if explicacion and explicacion.get("contribuciones"):
        contrib = pd.DataFrame(explicacion["contribuciones"]).sort_values("contribucion")
        fig = go.Figure(go.Bar(
            x=contrib.contribucion, y=contrib.variable, orientation="h",
            marker_color=[TERRACOTA if v > 0 else PALETTE["bondi"] for v in contrib.contribucion],
            customdata=contrib.valor, hovertemplate="%{y} = %{customdata}<br>contribución %{x:.3f}<extra></extra>"))
        unidad = explicacion.get("unidad") or ""
        metodo = {"shap": "SHAP", "lime": "LIME", "perturbacion": "perturbación con valores típicos"}.get(
            explicacion.get("metodo"), explicacion.get("metodo"))
        fig.update_xaxes(title=f"Contribución al riesgo ({unidad}) · naranja empuja hacia arriba, azul hacia abajo")
        st.plotly_chart(style_figure(fig, f"Qué variables explican este resultado · método: {metodo}", height=380),
                        use_container_width=True, config={"displaylogo": False})

    with st.expander("Variables que recibió el modelo"):
        variables = pd.DataFrame({"variable": list(prediction["variables"]),
                                  "valor": list(prediction["variables"].values())})
        st.dataframe(variables, hide_index=True, use_container_width=True)
        st.caption(f"{prediction['modelo']} · versión {prediction.get('version') or '1.0.0'} · "
                   f"{prediction.get('momento_prediccion') or ''}")


# ---------------------------------------------------------------------------
# Pestaña 2: cómo funciona y resultados
# ---------------------------------------------------------------------------
def _comparison_chart(cv_rows: list[dict], prevalence: float, winner: str) -> go.Figure:
    df = pd.DataFrame(cv_rows).sort_values("pr_auc_mean")
    colors = [TERRACOTA if winner.startswith(m) else PALETTE["bondi"] for m in df.modelo]
    fig = go.Figure(go.Bar(
        x=df.pr_auc_mean, y=[MODEL_LABELS.get(m, m) for m in df.modelo], orientation="h",
        error_x={"type": "data", "array": df.pr_auc_std, "visible": True},
        marker_color=colors, text=[f"{v:.3f}" for v in df.pr_auc_mean], textposition="outside",
        hovertemplate="%{y}: PR-AUC %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=prevalence, line_dash="dash", line_color="grey",
                  annotation_text=f"Azar ({prevalence:.3f})", annotation_position="top")
    fig.update_xaxes(title="PR-AUC en validación cruzada (más alto es mejor)", range=[0, max(df.pr_auc_mean.max() * 1.25, 0.2)])
    return style_figure(fig, "Comparación de modelos", height=380)


def _curves_chart(chart: dict) -> go.Figure:
    roc, pr = chart["roc_curve"], chart["pr_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"ROC (AUC {roc['auc']:.3f})",
                             line={"color": PALETTE["san_marino"], "width": 2.6}))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line={"color": "grey", "dash": "dash"}))
    fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                             name=f"Precision-Recall (AP {pr['average_precision']:.3f})",
                             line={"color": TERRACOTA, "width": 2.6}))
    fig.add_hline(y=chart["prevalence_test"], line_dash="dot", line_color="grey",
                  annotation_text="Prevalencia", annotation_position="bottom right")
    fig.update_xaxes(title="Tasa de falsos positivos · Recall")
    fig.update_yaxes(title="Recall · Precision", range=[0, 1.02])
    return style_figure(fig, "Curvas ROC y Precision-Recall en el conjunto de prueba", height=400)


def _gains_chart(gains: list[dict]) -> go.Figure:
    df = pd.DataFrame(gains).sort_values("decil")
    x = [0] + (df.decil * 10).tolist()
    y = [0] + df["captura_acumulada_%"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Modelo",
                             line={"color": TERRACOTA, "width": 2.6},
                             hovertemplate="Contactando el %{x}% más riesgoso se captura el %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Selección aleatoria",
                             line={"color": "grey", "dash": "dash"}))
    fig.update_xaxes(title="% de pedidos contactados, ordenados por riesgo")
    fig.update_yaxes(title="% de reseñas negativas capturadas")
    return style_figure(fig, "Curva de ganancia acumulada", height=380)


def _confusion_chart(cm: dict) -> go.Figure:
    matrix = cm["matrix"]
    labels_real = ["Positiva/neutra", "Negativa"]
    labels_pred = ["Predice sin alerta", "Predice riesgo"]
    total_fila = [sum(f) for f in matrix]
    text = [[f"{v:,}<br>{v / total_fila[i] * 100:.1f}% de la fila" for v in fila] for i, fila in enumerate(matrix)]
    fig = go.Figure(go.Heatmap(z=matrix, x=labels_pred, y=labels_real, text=text, texttemplate="%{text}",
                               colorscale=[[0, "#EAF4F6"], [1, PALETTE["eden"]]], showscale=False))
    fig.update_yaxes(title="Clase real", autorange="reversed")
    return style_figure(fig, "Matriz de confusión en prueba", height=340)


def _importance_chart(rows: list[dict]) -> go.Figure:
    df = pd.DataFrame(rows).sort_values("importancia")
    colors = [TERRACOTA if f.startswith("Log") else PALETTE["bondi"] for f in df.familia]
    fig = go.Figure(go.Bar(x=df.importancia, y=df.variable, orientation="h", marker_color=colors,
                           error_x={"type": "data", "array": df["std"], "visible": True},
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="Caída de PR-AUC al desordenar la variable (importancia por permutación)")
    return style_figure(fig, "Qué variables pesan más (naranja: logística observada · azul: conocida al comprar)",
                        height=460)


def _render_evaluation(results: dict) -> None:
    manifest, chart = results["manifest"], results["chart_data"]
    m = manifest["metrics"]
    prevalence = chart.get("prevalence_test") or m.get("prevalencia_test") or 0.0
    gains = chart.get("gains_by_decile", [])
    top10 = next((g["captura_acumulada_%"] for g in gains if g["decil"] == 1), None)

    section_intro("Cómo funciona",
                  "El modelo mira el pedido y su logística en el momento en que Olist envía la encuesta y "
                  "ordena los pedidos del más al menos riesgoso.")
    pasos = st.columns(4)
    pasos[0].info("**1 · Qué predice**\n\nSi la reseña tendrá 1 o 2 estrellas. En entrenamiento ocurre en "
                  f"{m.get('n_train', 0):,} pedidos; en prueba, el {_pct(prevalence)} de {m.get('n_test', 0):,}.")
    pasos[1].info("**2 · Con qué información**\n\n33 datos del pedido, el pago, el producto y la logística "
                  "observada: días de retraso, si llegó tarde o si seguía pendiente al enviar la encuesta.")
    pasos[2].info("**3 · Cómo se probó**\n\nSe entrena con compras de 2017 a mayo 2018 y se prueba con "
                  "junio a agosto 2018, meses que el modelo nunca vio. Seis algoritmos compitieron.")
    pasos[3].info(f"**4 · Modelo elegido**\n\n{manifest['winner']}. Entre los empatados se eligió el más "
                  f"simple; el umbral de alerta ({m.get('umbral', 0.5):.2f}) se fijó fuera de muestra.")

    section_intro("Resultados en pedidos nunca vistos", f"{manifest['period']}.")
    k = st.columns(5)
    k[0].metric("PR-AUC", f"{m['pr_auc']:.3f}", f"{m['pr_auc'] / prevalence:.1f}× el azar" if prevalence else None,
                help="Qué tan bien encuentra las reseñas negativas siendo pocas. Adivinar al azar daría la prevalencia.")
    k[1].metric("ROC-AUC", f"{m['roc_auc']:.3f}", help="Probabilidad de ordenar un pedido malo por encima de uno bueno.")
    k[2].metric("Detecta (recall)", _pct(m["recall"]), help="Porcentaje de reseñas negativas que el modelo marca con el umbral.")
    k[3].metric("Acierta (precisión)", _pct(m["precision"]), help="De los pedidos marcados, cuántos sí terminan en negativa.")
    k[4].metric("Capta en el 10% más riesgoso", f"{top10:.0f}%" if top10 is not None else "—",
                help="Si solo se contacta al 10% de pedidos más riesgosos, qué parte de las negativas se alcanza.")

    izq, der = st.columns(2)
    with izq:
        st.plotly_chart(_comparison_chart(chart["model_comparison_cv"], prevalence, manifest["winner"]),
                        use_container_width=True, config={"displaylogo": False})
    with der:
        st.plotly_chart(_curves_chart(chart), use_container_width=True, config={"displaylogo": False})

    izq, der = st.columns(2)
    with izq:
        st.plotly_chart(_gains_chart(gains), use_container_width=True, config={"displaylogo": False})
    with der:
        st.plotly_chart(_confusion_chart(chart["confusion_matrix"]), use_container_width=True,
                        config={"displaylogo": False})

    st.plotly_chart(_importance_chart(chart["feature_importance"]), use_container_width=True,
                    config={"displaylogo": False})

    estado = chart.get("status_at_survey")
    if estado:
        st.markdown("#### Lo que más pesa: el estado de la entrega al enviar la encuesta")
        tabla = pd.DataFrame(estado).rename(columns={"estado": "Estado", "pedidos": "Pedidos",
                                                     "tasa_negativas": "% de reseñas negativas"})
        tabla["% de reseñas negativas"] = tabla["% de reseñas negativas"].map(lambda v: _pct(v))
        st.dataframe(tabla, hide_index=True, use_container_width=True)

    st.markdown("#### Escenarios: ¿cuánto se puede anticipar al comprar?")
    esc = pd.DataFrame(chart["scenario_comparison"]).rename(columns={
        "escenario": "Escenario", "variables": "Variables", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
        "lift_pr_auc": "Veces sobre el azar", "captura_top10_%": "Captura en el 10% más riesgoso"})
    st.dataframe(esc.style.format({"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Veces sobre el azar": "{:.1f}×",
                                   "Captura en el 10% más riesgoso": "{:.1f}%", "Variables": "{:.0f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Con solo los datos del momento de la compra el modelo casi no acierta: el riesgo nace en la logística.")

    with st.expander("Detalle técnico: desbalance, comparación en prueba y calibración"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tratamiento del desbalance (validación cruzada)**")
            st.dataframe(pd.DataFrame(chart["imbalance_strategies"]).round(4), hide_index=True,
                         use_container_width=True)
        with c2:
            st.markdown("**Todos los modelos en el conjunto de prueba**")
            st.dataframe(pd.DataFrame(chart["model_comparison_test"]).round(4), hide_index=True,
                         use_container_width=True)
        cal = chart.get("calibration")
        if cal:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                                     line={"color": "grey", "dash": "dash"}))
            fig.add_trace(go.Scatter(x=cal["prob_media"], y=cal["fraccion_positivos"], mode="lines+markers",
                                     name="Modelo", line={"color": TERRACOTA, "width": 2.4}))
            fig.update_xaxes(title="Probabilidad predicha (media del decil)")
            fig.update_yaxes(title="Fracción real de negativas")
            st.plotly_chart(style_figure(fig, "Calibración: la probabilidad se puede leer como frecuencia", height=340),
                            use_container_width=True, config={"displaylogo": False})

    izq, der = st.columns(2)
    with izq:
        st.subheader("Conclusiones de negocio")
        for linea in manifest["business_conclusions"]:
            st.markdown(f"- {linea}")
    with der:
        st.subheader("Limitaciones")
        for linea in manifest["limitations"]:
            st.markdown(f"- {linea}")

    with st.expander("Muestra de predicciones del conjunto de prueba"):
        muestra = pd.DataFrame(results["results"])
        st.dataframe(muestra, hide_index=True, use_container_width=True)
        st.download_button("Descargar muestra (CSV)", muestra.to_csv(index=False).encode("utf-8"),
                           file_name="clasificacion_resultados.csv", mime="text/csv")


def render(module: dict) -> None:
    page_header("Modelo predictivo", "Riesgo de reseña negativa",
                "Identifica los pedidos con más probabilidad de recibir una mala calificación para actuar antes.")
    try:
        results = get_classification_results()
    except (HTTPError, URLError, TimeoutError, ValueError, FileNotFoundError):
        render_module(module, "Clase real, probabilidad, predicción, matriz de confusión y métricas sensibles al desbalance.")
        return

    manifest = results["manifest"]
    st.markdown(f"{status_badge(manifest['status'])} &nbsp; <code>{module['branch']}</code> &nbsp; "
                f"v{manifest['version']} &nbsp;·&nbsp; {manifest['winner']}", unsafe_allow_html=True)

    prediction_tab, evaluation_tab = st.tabs(["Evaluar un pedido", "Cómo funciona y resultados"])
    with prediction_tab:
        _render_prediction(results)
    with evaluation_tab:
        _render_evaluation(results)
