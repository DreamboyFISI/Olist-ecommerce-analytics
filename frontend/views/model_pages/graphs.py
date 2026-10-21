from frontend.views.module_template import render_module


def render(module: dict) -> None:
    render_module(module, "Nodos, aristas, peso, comunidad, centralidad y una vista interpretable de las relaciones.")
