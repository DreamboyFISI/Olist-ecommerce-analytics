from datetime import datetime
import json
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from urllib.error import HTTPError, URLError

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from frontend.components import PALETTE, page_header, section_intro, style_figure
from frontend.data_client import (
    get_location_cities,
    get_locations,
    get_regression_results,
    predict_regression,
    resolve_location_state,
)


BRAZIL_BOUNDARY_PATH = Path(__file__).resolve().parents[2] / "assets" / "brazil_boundary.geojson"
with BRAZIL_BOUNDARY_PATH.open(encoding="utf-8") as boundary_source:
    BRAZIL_BOUNDARY = json.load(boundary_source)


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


def _haversine_km(origin: dict, destination: dict) -> float:
    lat1, lng1 = radians(origin["lat"]), radians(origin["lng"])
    lat2, lng2 = radians(destination["lat"]), radians(destination["lng"])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 6_371.0088 * 2 * asin(sqrt(value))


def _point_in_ring(lat: float, lng: float, ring: list[list[float]]) -> bool:
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > lat) != (y2 > lat)
        if crosses:
            border_lng = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lng < border_lng:
                inside = not inside
        previous = current
    return inside


def _distance_to_segment(lat: float, lng: float, start: list[float], end: list[float]) -> float:
    x, y = lng, lat
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return sqrt((x - x1) ** 2 + (y - y1) ** 2)
    position = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx**2 + dy**2)))
    projected_x, projected_y = x1 + position * dx, y1 + position * dy
    return sqrt((x - projected_x) ** 2 + (y - projected_y) ** 2)


def _near_country_border(lat: float, lng: float, ring: list[list[float]], tolerance: float = 0.15) -> bool:
    return any(
        _distance_to_segment(lat, lng, ring[index - 1], ring[index]) <= tolerance
        for index in range(len(ring))
    )


def _is_inside_brazil(lat: float, lng: float) -> bool:
    geometry = BRAZIL_BOUNDARY["features"][0]["geometry"]
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return any(
        (
            _point_in_ring(lat, lng, polygon[0])
            and not any(_point_in_ring(lat, lng, hole) for hole in polygon[1:])
        )
        or _near_country_border(lat, lng, polygon[0])
        for polygon in polygons
    )


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
    folium.GeoJson(
        BRAZIL_BOUNDARY,
        name="Límite de Brasil",
        style_function=lambda _: {
            "color": "#0799b6",
            "weight": 2,
            "fillColor": "#9cd2d3",
            "fillOpacity": 0.08,
        },
    ).add_to(route_map)
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
        "Calculadora de flete",
        "Ubica el origen y el destino, completa los datos del pedido y obtén una estimación en reales brasileños.",
    )
    states = results["states"]
    categories = results["categories"]

    customer_state_key = "regression-customer-state"
    pending_customer_state = st.session_state.pop("pending-customer-state", None)
    if pending_customer_state in states:
        st.session_state[customer_state_key] = pending_customer_state

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
        key=customer_state_key,
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
            "Haz clic en cualquier punto dentro de Brasil. El estado del cliente se ajustará "
            "automáticamente y la coordenada se usará solo para esta simulación."
        )
        customer = st.session_state.get("new-customer-point")
        if customer and customer.get("state") != customer_state:
            customer = None
        clicked = _route_map(seller, customer, customer_state, allow_click=True)
        if clicked:
            selected_point = {
                "lat": float(clicked["lat"]),
                "lng": float(clicked["lng"]),
            }
            if not _is_inside_brazil(selected_point["lat"], selected_point["lng"]):
                st.session_state.pop("new-customer-point", None)
                customer = None
                st.error("El punto seleccionado está fuera de Brasil. Selecciona una ubicación dentro del contorno marcado.")
            else:
                try:
                    resolved = resolve_location_state(
                        selected_point["lat"], selected_point["lng"]
                    )
                except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                    st.error(f"No se pudo identificar el estado del punto: {exc}")
                else:
                    selected_point["state"] = resolved["state"]
                    if customer != selected_point:
                        st.session_state["new-customer-point"] = selected_point
                        if resolved["state"] != customer_state:
                            st.session_state["pending-customer-state"] = resolved["state"]
                        st.rerun()
        if customer:
            st.success(
                f"Destino seleccionado en {_state_label(customer['state'])}: "
                f"{customer['lat']:.5f}, {customer['lng']:.5f}"
            )
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
        route_cols = st.columns(2)
        purchase_date = route_cols[0].date_input(
            "Fecha de compra",
            value=datetime(2018, 7, 10).date(),
            help="Los datos históricos de Olist abarcan de 2016 a 2018.",
        )
        purchase_time = route_cols[1].time_input(
            "Hora de compra",
            value=datetime(2018, 7, 10, 10, 30).time(),
        )
        st.markdown("#### Pedido y producto")
        order_cols = st.columns(3)
        n_items = order_cols[0].number_input("Cantidad de ítems", min_value=1, max_value=100, value=1)
        n_products = order_cols[1].number_input("Productos diferentes", min_value=1, max_value=100, value=1)
        n_sellers = order_cols[2].number_input("Vendedores", min_value=1, max_value=20, value=1)
        default_category = categories.index("beleza_saude") if "beleza_saude" in categories else 0
        main_category = st.selectbox(
            "Categoría principal",
            categories,
            index=default_category,
            format_func=_category_label,
        )

        value_cols = st.columns(3)
        price_total = value_cols[0].number_input("Precio total (BRL)", min_value=0.0, max_value=100_000.0, value=120.0, step=10.0)
        weight_total_g = value_cols[1].number_input("Peso total (g)", min_value=0.0, max_value=1_000_000.0, value=1_000.0, step=100.0)
        volume_total_cm3 = value_cols[2].number_input("Volumen total (cm³)", min_value=0.0, max_value=100_000_000.0, value=8_000.0, step=500.0)
        submitted = st.form_submit_button("Estimar costo del flete", type="primary", use_container_width=True)

    if not submitted:
        return
    if customer is None:
        st.error("Primero ubica al cliente en el mapa.")
        return
    if n_products > n_items:
        st.error("Los productos diferentes no pueden superar la cantidad de ítems.")
        return
    if n_sellers > n_items:
        st.error("La cantidad de vendedores no puede superar la cantidad de ítems.")
        return

    payload = {
        "purchase_timestamp": datetime.combine(purchase_date, purchase_time).isoformat(),
        "n_items": n_items,
        "n_products": n_products,
        "n_sellers": n_sellers,
        "price_total": price_total,
        "weight_total_g": weight_total_g,
        "volume_total_cm3": volume_total_cm3,
        "distance_km": distance_km,
        "seller_id": seller["id"],
        "seller_lat": seller["lat"],
        "seller_lng": seller["lng"],
        "customer_lat": customer["lat"],
        "customer_lng": customer["lng"],
        "customer_state": customer_state,
        "main_seller_state": seller_state,
        "main_category": main_category,
    }
    try:
        with st.spinner("Estimando el costo del envío..."):
            prediction = predict_regression(payload)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        st.error(f"La API rechazó la predicción: {detail}")
        return
    except (URLError, TimeoutError, ValueError) as exc:
        st.error(f"No se pudo conectar con el modelo: {exc}")
        return

    predicted_freight = prediction["predicted_freight"]
    high_probability = prediction["high_freight_probability"] * 100
    metric_cols = st.columns(4)
    metric_cols[0].metric("Flete estimado", f"R$ {predicted_freight:,.2f}")
    metric_cols[1].metric("Tipo de flete", prediction["segment"])
    metric_cols[2].metric("Señal de costo alto", f"{high_probability:.1f}%")
    metric_cols[3].metric("Margen histórico típico", f"R$ {prediction['typical_error']:,.2f}")

    route_text = f"R$ {prediction['route_reference']:,.2f}"
    seller_text = f"R$ {prediction['seller_reference']:,.2f}"
    st.info(
        f"El pedido fue comparado con rutas y vendedores similares. Como referencia, la ruta registra "
        f"un flete histórico de **{route_text}** y el vendedor de **{seller_text}**."
    )

    comparison = pd.DataFrame({
        "Referencia": ["Estimación", "Historial de la ruta", "Historial del vendedor"],
        "Flete (BRL)": [
            predicted_freight,
            prediction["route_reference"],
            prediction["seller_reference"],
        ],
    })
    fig = px.bar(
        comparison,
        x="Referencia",
        y="Flete (BRL)",
        color="Referencia",
        text_auto=".2f",
        color_discrete_sequence=[PALETTE["bondi"], PALETTE["san_marino"], PALETTE["sinbad"]],
    )
    st.plotly_chart(style_figure(fig, "Estimación frente a referencias históricas", 360), use_container_width=True)
    st.caption(
        "Es una estimación de apoyo, no una tarifa definitiva. Los fletes caros presentan mayor variación."
    )


def _render_evaluation(results: dict) -> None:
    manifest = results["manifest"]
    section_intro(
        "Cómo evolucionó el modelo",
        "Se evaluaron dos formas de estimar el flete sobre pedidos recientes que no se usaron para entrenar.",
    )

    final_metrics = manifest["test_metrics"]
    top_cols = st.columns(4)
    top_cols[0].metric("Error promedio", f"R$ {final_metrics['MAE']:.2f}")
    top_cols[1].metric("Error relativo total", f"{final_metrics['WAPE']:.2f}%")
    top_cols[2].metric("Variación explicada", f"{final_metrics['R2'] * 100:.1f}%")
    top_cols[3].metric("Pedidos de prueba", f"{results['test_orders']:,}")

    st.markdown("#### Las dos versiones")
    versions = pd.DataFrame(results["version_comparison"])
    st.dataframe(
        versions.rename(columns={
            "version": "Versión",
            "enfoque": "Cómo trabaja",
            "MAE": "Error promedio",
            "RMSE": "Penaliza errores grandes",
            "R2": "Variación explicada",
            "WAPE": "Error relativo",
            "MAE_fletes_altos": "Error en fletes altos",
            "sesgo_fletes_altos": "Subestimación en fletes altos",
        }).style.format({
            "Error promedio": "R$ {:.2f}",
            "Penaliza errores grandes": "R$ {:.2f}",
            "Variación explicada": "{:.1%}",
            "Error relativo": "{:.2f}%",
            "Error en fletes altos": "R$ {:.2f}",
            "Subestimación en fletes altos": "R$ {:.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    version_errors = versions[["version", "MAE", "MAE_fletes_altos"]].melt(
        id_vars="version",
        var_name="Medición",
        value_name="Error (BRL)",
    )
    version_errors["Medición"] = version_errors["Medición"].replace({
        "MAE": "Todos los fletes",
        "MAE_fletes_altos": "Fletes mayores de R$30",
    })
    fig_versions = px.bar(
        version_errors,
        x="version",
        y="Error (BRL)",
        color="Medición",
        barmode="group",
        text_auto=".2f",
        color_discrete_sequence=[PALETTE["bondi"], PALETTE["san_marino"]],
        labels={"version": "Versión"},
    )
    st.plotly_chart(
        style_figure(fig_versions, "Error promedio: general y en fletes altos", 410),
        use_container_width=True,
    )
    st.success(
        "El sistema híbrido mantiene casi igual el error general y mejora los pedidos costosos: "
        "primero reconoce si el flete parece normal o alto y después usa el estimador más adecuado."
    )

    st.markdown("#### Qué significan las métricas")
    explanation_cols = st.columns(3)
    explanation_cols[0].info(
        "**Error promedio (MAE)**\n\nLa estimación se aleja, en promedio, R$ 5.45 del valor real."
    )
    explanation_cols[1].info(
        "**Error relativo (WAPE)**\n\nEl error acumulado equivale aproximadamente al 22% del costo real total."
    )
    explanation_cols[2].info(
        "**Variación explicada (R²)**\n\nEl sistema recoge cerca del 66% de las diferencias de precio entre pedidos."
    )

    components = results["component_metrics"]
    st.markdown("#### Las tres piezas del sistema")
    component_cols = st.columns(3)
    component_cols[0].metric("Detector normal / alto", f"{components['classifier']['f1'] * 100:.1f}%")
    component_cols[0].caption("Equilibra correctamente la detección de fletes caros y evita demasiadas falsas alarmas.")
    component_cols[1].metric("Estimador normal", f"R$ {components['normal_regressor']['MAE']:.2f}")
    component_cols[1].caption("Error promedio cuando el pedido pertenece al grupo de costo normal.")
    component_cols[2].metric("Estimador de costo alto", f"R$ {components['high_regressor']['MAE']:.2f}")
    component_cols[2].caption("Error promedio dentro del grupo más variable y difícil de anticipar.")

    sample = pd.DataFrame(results["prediction_sample"])
    fig_scatter = px.scatter(
        sample,
        x="freight_total",
        y="prediction",
        color="segmento_predicho",
        opacity=0.45,
        labels={
            "freight_total": "Flete real (BRL)",
            "prediction": "Flete estimado (BRL)",
            "segmento_predicho": "Grupo elegido",
        },
        color_discrete_sequence=[PALETTE["bondi"], PALETTE["san_marino"]],
    )
    maximum = float(max(sample["freight_total"].max(), sample["prediction"].max()))
    fig_scatter.add_trace(go.Scatter(
        x=[0, maximum],
        y=[0, maximum],
        mode="lines",
        name="Coincidencia ideal",
        line=dict(color=PALETTE["eden"], dash="dash"),
    ))
    st.plotly_chart(style_figure(fig_scatter, "Flete real frente a estimado"), use_container_width=True)
    st.caption("Para que el gráfico principal sea legible, se visualiza el 99% de los casos menos extremos.")

    error_ranges = pd.DataFrame(results["error_by_range"])
    fig_errors = px.bar(
        error_ranges,
        x="rango_flete",
        y="MAE",
        text_auto=".2f",
        color="MAE",
        color_continuous_scale=[[0, PALETTE["sinbad"]], [1, PALETTE["san_marino"]]],
        labels={"rango_flete": "Costo real del flete", "MAE": "Error promedio (BRL)"},
        hover_data=["pedidos", "flete_promedio"],
    )
    st.plotly_chart(style_figure(fig_errors, "Dónde resulta más difícil acertar", 390), use_container_width=True)

    importance = pd.DataFrame(results["feature_importance"]).head(8).sort_values("importancia_mae")
    fig_importance = px.bar(
        importance,
        x="importancia_mae",
        y="descripcion",
        orientation="h",
        text_auto=".2f",
        color_discrete_sequence=[PALETTE["bondi"]],
        labels={"importancia_mae": "Aporte a la estimación", "descripcion": "Información"},
    )
    st.plotly_chart(style_figure(fig_importance, "Información que más ayuda al modelo", 430), use_container_width=True)

    st.warning(
        "La principal limitación sigue estando en los fletes muy caros: son menos frecuentes y cambian mucho "
        "entre pedidos. Por eso la estimación debe utilizarse como referencia, especialmente por encima de R$60."
    )


def render(module: dict) -> None:
    page_header(
        "Modelo predictivo",
        "Costo del flete",
        "Estima cuánto podría costar enviar un pedido antes de confirmar la compra.",
    )
    try:
        results = get_regression_results()
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        st.error(f"No se pudieron cargar los resultados de regresión: {exc}")
        return

    prediction_tab, evaluation_tab = st.tabs([
        "Estimar un flete",
        "Cómo funciona y resultados",
    ])
    with prediction_tab:
        _render_prediction(results)
    with evaluation_tab:
        _render_evaluation(results)
