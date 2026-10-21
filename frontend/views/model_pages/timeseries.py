from frontend.views.module_template import render_module


def render(module: dict) -> None:
    render_module(module, "Fecha, valor real, pronóstico, intervalos de confianza y error de backtesting.")
