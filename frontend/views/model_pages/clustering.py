from frontend.views.module_template import render_module


def render(module: dict) -> None:
    render_module(module, "Cliente, cluster, métricas RFM, perfil del segmento y Silhouette Score.")
