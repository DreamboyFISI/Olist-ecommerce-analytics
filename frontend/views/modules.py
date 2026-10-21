from frontend.views.model_pages import association, classification, clustering, graphs, regression, timeseries

PAGES = {
    "association": association,
    "clustering": clustering,
    "timeseries": timeseries,
    "graphs": graphs,
    "regression": regression,
    "classification": classification,
}


def render(module: dict) -> None:
    PAGES[module["slug"]].render(module)
