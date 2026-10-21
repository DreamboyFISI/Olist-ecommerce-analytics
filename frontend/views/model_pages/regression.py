from frontend.views.module_template import render_module


def render(module: dict) -> None:
    render_module(module, "Valor real, predicción, error por pedido y métricas frente al baseline de 12.73 días.")
