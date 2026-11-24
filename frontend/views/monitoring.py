"""Página de Monitoreo y Auditoría: desempeño en el tiempo, deriva, versionado y bitácora."""

from urllib.error import HTTPError, URLError

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import PALETTE, page_header, render_empty_state, section_intro, style_figure
from frontend.data_client import get_monitoring_report

COLOR_ESTADO = {"OK": "#2E8B57", "Vigilar": "#E0A24A", "Alerta": "#E07B54", "Sin datos": "#9CA3AF"}


def _badge(estado: str) -> str:
    return (f"<span style='background:{COLOR_ESTADO.get(estado, '#9CA3AF')};color:white;padding:2px 10px;"
            f"border-radius:12px;font-weight:600'>{estado}</span>")


def _tendencia(modulo: dict) -> go.Figure | None:
    periodos = pd.DataFrame(modulo.get("periodos", []))
    if periodos.empty or "metrica" not in periodos:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periodos.periodo, y=periodos.metrica, mode="lines+markers",
                             name=modulo["metrica"], line={"color": PALETTE["san_marino"], "width": 2.6},
                             hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    if modulo.get("referencia") is not None:
        fig.add_hline(y=modulo["referencia"], line_dash="dash", line_color=PALETTE["eden"],
                      annotation_text="Referencia del manifiesto", annotation_position="top left")
    if "tasa_objetivo" in periodos:
        fig.add_trace(go.Scatter(x=periodos.periodo, y=periodos.tasa_objetivo, mode="lines+markers",
                                 name="Tasa objetivo", yaxis="y2", line={"color": "#E07B54", "width": 1.8, "dash": "dot"}))
        fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "title": "Tasa objetivo", "showgrid": False})
    fig.update_yaxes(title=modulo["metrica"])
    fig.update_xaxes(title="Periodo")
    mejor = "más alto es mejor" if modulo.get("mayor_es_mejor") else "más bajo es mejor"
    return style_figure(fig, f"{modulo['titulo']} · {modulo['metrica']} por periodo ({mejor})", height=360)


def render() -> None:
    page_header("Operación de los modelos", "Monitoreo y auditoría",
                "Vigila si los modelos siguen funcionando como cuando se entrenaron, registra cada versión y deja "
                "rastro de cada revisión.")
    col_a, col_b = st.columns([3, 1])
    with col_b:
        refrescar = st.button("Recalcular ahora", help="Vuelve a evaluar los artefactos y agrega una fila a la bitácora.")
    try:
        report = get_monitoring_report(refresh=refrescar)
    except (HTTPError, URLError, TimeoutError, ValueError, FileNotFoundError) as exc:
        render_empty_state("Sin reporte de monitoreo",
                           f"Ejecuta `python -m src.monitoring` para generar el primer reporte. Detalle: {exc}")
        return

    resumen = report["resumen"]
    with col_a:
        st.caption(f"Último reporte: {report['generated_at']} · umbrales: PSI vigilar {report['umbrales']['psi_vigilar']} "
                   f"/ alerta {report['umbrales']['psi_alerta']}; degradación vigilar "
                   f"{report['umbrales']['degradacion_vigilar'] * 100:.0f}% / alerta "
                   f"{report['umbrales']['degradacion_alerta'] * 100:.0f}%; antigüedad máxima {report['umbrales']['max_edad_dias']} días.")
    k = st.columns(4)
    k[0].metric("Módulos OK", resumen.get("OK", 0))
    k[1].metric("En vigilancia", resumen.get("Vigilar", 0))
    k[2].metric("En alerta", resumen.get("Alerta", 0))
    k[3].metric("Sin datos", resumen.get("Sin datos", 0))

    estado_tab, tendencia_tab, deriva_tab, registro_tab, auditoria_tab, reentreno_tab = st.tabs([
        "Estado", "Desempeño en el tiempo", "Deriva", "Versionado", "Auditoría", "Reentrenamiento"])

    with estado_tab:
        section_intro("Semáforo por módulo", "Cada módulo se compara con la métrica de su propio manifiesto.")
        for m in report["modulos"]:
            with st.container(border=True):
                c = st.columns([2, 1, 1, 1, 4])
                c[0].markdown(f"**{m['titulo']}** · v{m.get('version') or '—'}")
                c[1].markdown(_badge(m["estado"]), unsafe_allow_html=True)
                c[2].markdown(f"{m['metrica'] or '—'}: **{m['actual'] if m['actual'] is not None else '—'}**")
                ref = m.get("referencia")
                c[3].markdown(f"ref. {ref:.3f}" if isinstance(ref, (int, float)) else "ref. —")
                c[4].caption(" · ".join(m.get("motivos", [])))

    with tendencia_tab:
        section_intro("Desempeño por periodo", "La métrica principal recalculada sobre las predicciones fechadas "
                                               "que cada módulo dejó en sus artefactos.")
        for m in report["modulos"]:
            fig = _tendencia(m)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.caption("Los periodos con menos de 30 predicciones se omiten para no leer ruido como tendencia.")

    with deriva_tab:
        section_intro("Deriva de datos y de la tasa objetivo",
                      "PSI de las probabilidades o errores entre el primer y el último periodo, y cambio de la "
                      "tasa del objetivo. PSI < 0.10 estable, 0.10–0.25 vigilar, > 0.25 alerta.")
        filas = []
        for m in report["modulos"]:
            d = m.get("drift", {})
            if "psi_score" in d:
                p = d["psi_score"]
                filas.append({"Módulo": m["titulo"], "Medición": "PSI del score",
                              "Valor": p["valor"] if p.get("valor") is not None else "—",
                              "Periodo": f"{p['desde']} → {p['hasta']}" if p.get("desde") else p.get("nota", "—"),
                              "Estado": p["estado"]})
            if "tasa_objetivo" in d:
                t = d["tasa_objetivo"]
                filas.append({"Módulo": m["titulo"], "Medición": "Tasa objetivo", "Valor": f"{t['inicial']:.3f} → {t['final']:.3f}",
                              "Periodo": f"cambio {t['cambio_relativo'] * 100:+.0f}%", "Estado": t["estado"]})
        if filas:
            st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
        else:
            st.info("Ningún módulo tiene suficientes periodos para medir deriva.")

    with registro_tab:
        section_intro("Registro de modelos", "Versión, fecha y huella SHA-256 de cada artefacto: lo que permite "
                                             "saber exactamente qué modelo está sirviendo el dashboard.")
        filas = [{"Módulo": r["titulo"], "Modelo": r.get("modelo") or "—", "Versión": r.get("version") or "—",
                  "Generado": (r.get("generado") or "—")[:19], "Edad (días)": r.get("edad_dias"),
                  "Antigüedad": r["antiguedad"], "Artefactos": len(r["artefactos"])} for r in report["registro"]]
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
        modulo = st.selectbox("Ver artefactos de", [r["titulo"] for r in report["registro"]])
        r = next(x for x in report["registro"] if x["titulo"] == modulo)
        st.dataframe(pd.DataFrame(r["artefactos"]), hide_index=True, use_container_width=True)
        if r.get("metricas"):
            st.dataframe(pd.DataFrame([r["metricas"]]).round(4), hide_index=True, use_container_width=True)

    with auditoria_tab:
        section_intro("Bitácora de revisiones", "Cada corrida del monitoreo agrega una fila por módulo con la métrica "
                                                "observada, el estado y el motivo. Nadie la edita a mano.")
        auditoria = pd.DataFrame(report.get("auditoria", []))
        if auditoria.empty:
            st.info("Todavía no hay corridas registradas.")
        else:
            st.dataframe(auditoria.iloc[::-1], hide_index=True, use_container_width=True)
            st.download_button("Descargar bitácora (CSV)", auditoria.to_csv(index=False).encode("utf-8"),
                               file_name="audit_log.csv", mime="text/csv")

    with reentreno_tab:
        section_intro("Plan de reentrenamiento", "Regla: un módulo en Alerta se reentrena; uno en Vigilar se revisa. "
                                                 "El comando se ejecuta desde la terminal, nunca desde el dashboard.")
        plan = report.get("plan_reentrenamiento", [])
        if not plan:
            st.success("Ningún módulo requiere reentrenamiento en este momento.")
        for accion in plan:
            with st.container(border=True):
                st.markdown(f"**{accion['modulo']}** · {_badge(accion['estado'])} · versión actual "
                            f"{accion.get('version_actual') or '—'}", unsafe_allow_html=True)
                for motivo in accion.get("motivos", []):
                    st.markdown(f"- {motivo}")
                if accion.get("ejecutable"):
                    st.code(accion["comando"] or "", language="bash")
                else:
                    st.caption("Este módulo se entrena desde su notebook; ejecútalo y vuelve a correr el monitoreo.")
        st.code("python -m src.monitoring --retrain          # reentrena lo que esté en Alerta\n"
                "python -m src.monitoring --retrain --dry-run  # solo muestra el plan", language="bash")
