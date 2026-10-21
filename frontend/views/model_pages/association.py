from frontend.views.module_template import render_module


def render(module: dict) -> None:
    render_module(module, "Reglas con support, confidence y lift, identificando el algoritmo que produjo cada resultado.")
