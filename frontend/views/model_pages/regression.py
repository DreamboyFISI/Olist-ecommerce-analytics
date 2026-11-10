from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from urllib.error import HTTPError, URLError

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from frontend.components import PALETTE, page_header, section_intro, status_badge, style_figure
from frontend.data_client import (
    get_location_cities,
    get_locations,
    get_regression_results,
    predict_regression,
)


STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
    "BA": "Bahía", "CE": "Ceará", "DF": "Distrito Federal",
    "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MT": "Mato Grosso", "MS": "Mato Grosso del Sur", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco",
    "PI": "Piauí", "RJ": "Río de Janeiro", "RN": "Río Grande del Norte",
    "RS": "Río Grande del Sur", "RO": "Rondônia", "RR": "Roraima",
    "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe",
    "TO": "Tocantins",
}

STATE_CENTERS = {
    "AC": (-9.0, -70.0), "AL": (-9.6, -36.7), "AP": (1.4, -51.8),
    "AM": (-3.4, -65.0), "BA": (-12.5, -41.7), "CE": (-5.2, -39.5),
    "DF": (-15.8, -47.9), "ES": (-19.6, -40.5), "GO": (-15.9, -49.8),
    "MA": (-5.0, -45.3), "MT": (-12.7, -55.9), "MS": (-20.5, -54.6),
    "MG": (-18.5, -44.6), "PA": (-3.8, -52.5), "PB": (-7.1, -36.7),
    "PR": (-24.6, -51.6), "PE": (-8.4, -37.9), "PI": (-7.7, -42.7),
    "RJ": (-22.2, -42.7), "RN": (-5.8, -36.6), "RS": (-30.0, -53.2),
    "RO": (-10.9, -62.8), "RR": (2.1, -61.4), "SC": (-27.2, -50.2),
    "SP": (-22.2, -48.6), "SE": (-10.6, -37.4), "TO": (-10.2, -48.3),
}


def _state_label(code: str) -> str:
    return f"{STATE_NAMES.get(code, code)} ({code})"


def _category_label(code: str) -> str:
    return code.replace("_", " ").capitalize()


def _risk_label(days: float) -> str:
    if days <= 7:
        return "Entrega rápida"
    if days <= 14:
        return "Entrega estándar"
    return "Entrega prolongada"


def _haversine_km(origin: dict, destination: dict) -> float:
    lat1, lng1 = radians(origin["lat"]), radians(origin["lng"])
    lat2, lng2 = radians(destination["lat"]), radians(destination["lng"])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 6_371.0088 * 2 * asin(sqrt(value))


def _location_selector(role: str, state: str, title: str) -> dict:
    cities = get_location_cities(role, state)
    if not cities:
        raise ValueError(f"No hay ubicaciones disponibles para {title.lower()} en {_state_label(state)}.")
    city = st.selectbox(
        f"Ciudad del {title.lower()}",
        cities,
        key=f"{role}-city-{state}",
    )
    locations = get_locations(role, state, city)
    if not locations:
        raise ValueError(f"No hay registros disponibles en {city}.")
    labels = {location["id"]: location["label"] for location in locations}
    selected_id = st.selectbox(
        title,
        labels,
        format_func=lambda identifier: labels[identifier],
        key=f"{role}-location-{state}-{city}",
        help="Los identificadores están anonimizados en el conjunto público de Olist.",
    )
    return next(location for location in locations if location["id"] == selected_id)


def _route_map(seller: dict, customer: dict | None, state: str, allow_click: bool) -> dict | None:
    center = [customer["lat"], customer["lng"]] if customer else STATE_CENTERS.get(state, (-14.2, -51.9))
    zoom = 8 if customer else 6
    route_map = folium.Map(location=center, zoom_start=zoom, control_scale=True)
    folium.Marker(
        [seller["lat"], seller["lng"]],
        tooltip="Vendedor",
        icon=folium.Icon(color="blue", icon="home"),
    ).add_to(route_map)
    if customer:
        folium.Marker(
            [customer["lat"], customer["lng"]],
            tooltip="Cliente",
            icon=folium.Icon(color="green", icon="user"),
        ).add_to(route_map)
        folium.PolyLine(
            [[seller["lat"], seller["lng"]], [customer["lat"], customer["lng"]]],
            color="#0799b6",
            weight=3,
            dash_array="8",
            tooltip="Distancia en línea recta",
        ).add_to(route_map)
    output = st_folium(
        route_map,
        height=390,
        use_container_width=True,
        returned_objects=["last_clicked"] if allow_click else [],
        key=f"route-map-{state}-{'new' if allow_click else 'existing'}",
    )
    return output.get("last_clicked") if allow_click and output else None


def _render_prediction(results: dict) -> None:
    section_intro(
        "Simulador de entrega",
        "Indica la promesa de entrega y los datos básicos del pedido. XGBoost calculará una fecha probable y un rango orientativo; no necesita GPU para predecir.",
    )
    states = results["states"]
    categories = results["categories"]

    st.markdown("#### Origen y destino")
    state_cols = st.columns(2)
    seller_state = state_cols[0].selectbox(
        "Estado del vendedor",
        states,
        index=states.index("SP"),
        format_func=_state_label,
    )
    customer_state = state_cols[1].selectbox(
        "Estado del cliente",
        states,
        index=states.index("SP"),
        format_func=_state_label,
    )

    try:
        seller = _location_selector("seller", seller_state, "Vendedor")
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        st.error(f"No se pudo cargar el origen: {exc}")
        return

    customer_mode = st.radio(
        "¿Cómo deseas indicar el destino?",
        ["Elegir un cliente existente", "Ubicar un cliente nuevo en el mapa"],
        horizontal=True,
    )
    customer = None
    if customer_mode == "Elegir un cliente existente":
        try:
            customer = _location_selector("customer", customer_state, "Cliente")
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            st.error(f"No se pudo cargar el destino: {exc}")
            return
        _route_map(seller, customer, customer_state, allow_click=False)
    else:
        st.caption(
            f"Haz clic en el mapa para ubicar al cliente nuevo en {_state_label(customer_state)}. "
            "La coordenada se usa solo para esta simulación y no se guarda."
        )
        customer = st.session_state.get("new-customer-point")
        if customer and customer.get("state") != customer_state:
            customer = None
        clicked = _route_map(seller, customer, customer_state, allow_click=True)
        if clicked:
            selected_point = {
                "lat": float(clicked["lat"]),
                "lng": float(clicked["lng"]),
                "state": customer_state,
            }
            if customer != selected_point:
                st.session_state["new-customer-point"] = selected_point
                st.rerun()
        if customer:
            st.success(f"Destino seleccionado: {customer['lat']:.5f}, {customer['lng']:.5f}")
        else:
            st.warning("Selecciona un punto en el mapa para poder calcular la ruta.")

    automatic_distance = _haversine_km(seller, customer) if customer else None
    manual_distance = st.checkbox(
        "Ajustar manualmente la distancia",
        help="Úsalo solo si conoces una distancia logística más precisa que la línea recta.",
        disabled=automatic_distance is None,
    )
    if automatic_distance is not None:
        if manual_distance:
            distance_km = st.number_input(
                "Distancia logística (km)",
                min_value=0.0,
                max_value=10_000.0,
                value=float(round(automatic_distance, 1)),
                step=10.0,
            )
        else:
            distance_km = automatic_distance
            st.metric("Distancia geográfica calculada", f"{distance_km:,.1f} km")
            st.caption("Calculada en línea recta entre las coordenadas promedio del CEP de origen y destino.")
    else:
        distance_km = 0.0

    with st.form("regression-prediction-form"):
        st.markdown("#### Momento del pedido")
        route_cols = st.columns(3)
        purchase_date = route_cols[0].date_input("Fecha de compra")
        purchase_time = route_cols[1].time_input("Hora de compra")
        promised_date = route_cols[2].date_input(
            "Fecha prometida al cliente",
            value=purchase_date + pd.Timedelta(days=15),
            help=(
                "Es la fecha límite comunicada al cliente al registrar el pedido. "
                "Puede ser una fecha histórica, pero debe ser posterior a la compra."
            ),
        )
        st.markdown("#### Pedido y producto")
        order_cols = st.columns(3)
        n_items = order_cols[0].number_input("Cantidad de ítems", min_value=1, max_value=100, value=1)
        n_products = order_cols[1].number_input("Productos diferentes", min_value=1, max_value=100, value=1)
        n_sellers = order_cols[2].number_input("Vendedores", min_value=1, max_value=20, value=1)
        main_category = st.selectbox("Categoría principal", categories, index=categories.index("beleza_saude"), format_func=_category_label)

        value_cols = st.columns(4)
        price_total = value_cols[0].number_input("Precio total (BRL)", min_value=0.0, max_value=100_000.0, value=120.0, step=10.0)
        freight_total = value_cols[1].number_input("Flete total (BRL)", min_value=0.0, max_value=20_000.0, value=20.0, step=1.0)
        weight_total_g = value_cols[2].number_input("Peso total (g)", min_value=0.0, max_value=1_000_000.0, value=1_000.0, step=100.0)
        volume_total_cm3 = value_cols[3].number_input("Volumen total (cm³)", min_value=0.0, max_value=100_000_000.0, value=8_000.0, step=500.0)
        submitted = st.form_submit_button("Calcular tiempo de entrega", type="primary", use_container_width=True)

    if not submitted:
        return
    if customer is None:
        st.error("Primero ubica al cliente en el mapa.")
        return
    if promised_date <= purchase_date:
        st.error("La fecha prometida debe ser posterior a la fecha de compra.")
        return

    payload = {
        "purchase_timestamp": datetime.combine(purchase_date, purchase_time).isoformat(),
        "estimated_delivery_date": promised_date.isoformat(),
        "n_items": n_items,
        "n_products": n_products,
        "n_sellers": n_sellers,
        "price_total": price_total,
        "freight_total": freight_total,
        "weight_total_g": weight_total_g,
        "volume_total_cm3": volume_total_cm3,
        "distance_km": distance_km,
        "customer_state": customer_state,
        "main_seller_state": seller_state,
        "main_category": main_category,
    }
    try:
        with st.spinner("Calculando la estimación..."):
            prediction = predict_regression(payload)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        st.error(f"La API rechazó la predicción: {detail}")
        return
    except (URLError, TimeoutError, ValueError) as exc:
        st.error(f"No se pudo conectar con el modelo: {exc}")
        return

    predicted_days = prediction["predicted_days"]
    arrival = datetime.fromisoformat(prediction["estimated_arrival"])
    operational_arrival = (
        datetime.fromisoformat(prediction["operational_arrival"])
        if prediction.get("operational_arrival")
        else None
    )
    interval = prediction["interval"]
    earliest = datetime.fromisoformat(interval["earliest_arrival"])
    latest = datetime.fromisoformat(interval["latest_arrival"])
    metric_cols = st.columns(4)
    metric_cols[0].metric("Entrega estimada", f"{predicted_days:.2f} días")
    metric_cols[1].metric("Fecha probable", arrival.strftime("%d/%m/%Y"))
    interval_label = (
        f"Hasta {interval['upper_days']:.1f} días"
        if interval["lower_days"] <= 0
        else f"{interval['lower_days']:.1f}–{interval['upper_days']:.1f} días"
    )
    metric_cols[2].metric("Rango orientativo", interval_label)
    metric_cols[3].metric("Modelo utilizado", prediction["model"])
    if operational_arrival:
        st.warning(
            f"La fecha calculada cae en sábado, domingo o feriado nacional. "
            f"**Fecha operativa sugerida: {operational_arrival.strftime('%d/%m/%Y')}**."
        )
    st.info(
        f"En las pruebas históricas, cerca del 80% de los pedidos quedó dentro de un margen similar: entre el "
        f"**{earliest.strftime('%d/%m/%Y')}** y el **{latest.strftime('%d/%m/%Y')}**. "
        f"Lectura operativa: **{_risk_label(predicted_days)}**."
    )

    comparison = pd.DataFrame({
        "Referencia": ["Predicción del modelo", "Mediana histórica"],
        "Días": [predicted_days, prediction["baseline_days"]],
    })
    fig = px.bar(comparison, x="Referencia", y="Días", color="Referencia", text_auto=".2f", color_discrete_sequence=[PALETTE["bondi"], PALETTE["san_marino"]])
    st.plotly_chart(style_figure(fig, "Predicción frente a la referencia histórica", 360), use_container_width=True)
    st.caption("El rango usa el percentil 80 del error histórico. Es una referencia analítica, no una garantía logística.")


def _render_evaluation(results: dict) -> None:
    manifest = results["manifest"]
    section_intro("Evaluación del modelo", "Comparación temporal sobre pedidos recientes no utilizados para entrenar.")
    metrics = pd.DataFrame(results["metrics"]).sort_values("MAE")
    st.dataframe(metrics.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}), hide_index=True, use_container_width=True)

    fig_metrics = px.bar(metrics.sort_values("MAE", ascending=False), x="MAE", y="modelo", orientation="h", color="modelo", text_auto=".2f", color_discrete_sequence=[PALETTE["bondi"], PALETTE["san_marino"], PALETTE["sinbad"], PALETTE["janna"]])
    st.plotly_chart(style_figure(fig_metrics, "MAE por modelo · menor es mejor"), use_container_width=True)

    history = pd.DataFrame(manifest.get("metric_history", []))
    if not history.empty:
        st.markdown("#### Evolución de XGBoost")
        st.caption("Cada versión se comparó con las tres métricas. MAE y RMSE deben bajar; R² debe subir.")
        st.dataframe(
            history.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
            hide_index=True,
            use_container_width=True,
        )
        history_long = history.melt(
            id_vars=["version", "cambios"],
            value_vars=["MAE", "RMSE"],
            var_name="Métrica",
            value_name="Días",
        )
        fig_history = px.line(
            history_long,
            x="version",
            y="Días",
            color="Métrica",
            markers=True,
            hover_data=["cambios"],
            color_discrete_sequence=[PALETTE["bondi"], PALETTE["san_marino"]],
        )
        st.plotly_chart(style_figure(fig_history, "Reducción del error entre versiones", 380), use_container_width=True)

    sample = pd.DataFrame(results["prediction_sample"])
    fig_scatter = px.scatter(sample, x="delivery_days", y="prediction", color="absolute_error", color_continuous_scale=[[0, PALETTE["sinbad"]], [1, PALETTE["eden"]]], labels={"delivery_days": "Días reales", "prediction": "Días predichos", "absolute_error": "Error absoluto"})
    maximum = float(max(sample["delivery_days"].max(), sample["prediction"].max()))
    fig_scatter.add_trace(go.Scatter(x=[0, maximum], y=[0, maximum], mode="lines", name="Predicción perfecta", line=dict(color=PALETTE["san_marino"], dash="dash")))
    st.plotly_chart(style_figure(fig_scatter, "Real frente a predicho"), use_container_width=True)

    errors = pd.DataFrame(results["error_distribution"])
    fig_errors = px.bar(errors, x="range", y="orders", text_auto=True, color_discrete_sequence=[PALETTE["bondi"]], labels={"range": "Error absoluto (días)", "orders": "Pedidos"})
    st.plotly_chart(style_figure(fig_errors, "Distribución del error"), use_container_width=True)

    with st.expander("Conclusiones y limitaciones"):
        st.markdown("**Conclusiones de negocio**")
        for conclusion in manifest["business_conclusions"]:
            st.markdown(f"- {conclusion}")
        st.markdown("**Limitaciones**")
        for limitation in manifest["limitations"]:
            st.markdown(f"- {limitation}")


def render(module: dict) -> None:
    page_header("Modelo predictivo", "Regresión", "Estima cuántos días tardará en entregarse un pedido de Olist.")
    try:
        results = get_regression_results()
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        st.error(f"No se pudieron cargar los resultados de regresión: {exc}")
        return

    manifest = results["manifest"]
    st.markdown(f"{status_badge(manifest['status'])} &nbsp; <code>{manifest['winner']} · v{manifest['version']}</code>", unsafe_allow_html=True)
    top_cols = st.columns(4)
    top_cols[0].metric("Modelo ganador", manifest["winner"])
    top_cols[1].metric("MAE", f"{manifest['metrics']['MAE']:.2f} días")
    top_cols[2].metric("R²", f"{manifest['metrics']['R2']:.3f}")
    top_cols[3].metric("Pedidos de prueba", f"{results['test_orders']:,}")

    prediction_tab, evaluation_tab = st.tabs(["Predecir una entrega", "Resultados y gráficas"])
    with prediction_tab:
        _render_prediction(results)
    with evaluation_tab:
        _render_evaluation(results)
