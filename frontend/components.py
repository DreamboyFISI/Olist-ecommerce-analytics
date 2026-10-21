from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PALETTE = {"bondi": "#0799B6", "san_marino": "#4A6EB0", "eden": "#114C5F", "sinbad": "#9CD2D3", "janna": "#F2E6CF", "white": "#FFFFFF", "ink": "#17313A"}


def inject_styles() -> None:
    styles = (ROOT / "frontend" / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
    st.markdown('<a class="skip-link" href="#contenido-principal">Ir al contenido principal</a>', unsafe_allow_html=True)


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f"<section id='contenido-principal' class='page-header' tabindex='-1'><p class='eyebrow'>{kicker}</p><h1>{title}</h1><p class='lede'>{description}</p></section>", unsafe_allow_html=True)


def section_intro(title: str, description: str) -> None:
    st.markdown(f"<div class='section-intro'><h2>{title}</h2><p>{description}</p></div>", unsafe_allow_html=True)


def status_badge(status: str) -> str:
    css_class = "status-ready" if status.lower() in {"completado", "entrenado", "listo"} else "status-pending"
    return f"<span class='status-badge {css_class}'>{status}</span>"


def style_figure(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(title={"text": title, "x": 0.01, "xanchor": "left"} if title else None, height=height, margin=dict(l=20, r=20, t=65 if title else 25, b=25), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, sans-serif", color=PALETTE["ink"], size=14), hoverlabel=dict(bgcolor=PALETTE["eden"], font_color=PALETTE["white"]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(gridcolor="rgba(17,76,95,0.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(17,76,95,0.10)", zeroline=False)
    return fig


def render_empty_state(title: str, body: str) -> None:
    st.markdown(f"<div class='empty-state' role='status'><div class='empty-state-mark' aria-hidden='true'>◇</div><div><h3>{title}</h3><p>{body}</p></div></div>", unsafe_allow_html=True)
