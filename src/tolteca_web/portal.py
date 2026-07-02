"""TolTEC Web v3 — multi-page Dash portal.

Pages
-----
* ``/``            → :class:`~tolteca_web.obs.DataProdViewerPage`
* ``/sweep``       → :class:`~tolteca_web.sweep.SweepViewerPage`
* ``/reduced-obs`` → :class:`~tolteca_web.obs.ReducedObsViewerPage`
* ``/kids-diag``   → :class:`~tolteca_web.obs.KidsDiagViewerPage`
* ``/tel``         → :class:`~tolteca_web.obs.TelViewerPage`

Configuration (environment variables)
--------------------------------------
``TOLTECA_WEB_CACHE_ROOT``
    Path to the zarr + catalog cache directory.  Defaults to ``./cache``.
``TOLTECA_WEB_CATALOG``
    Path to ``catalog.parquet``.  Defaults to ``<cache_root>/catalog.parquet``.
``TOLTECA_WEB_TEL_ROOT``
    Directory containing ``tel_*.nc`` telescope files.
    Defaults to ``./data_lmt/tel``.
``ZARR_HTTP_MODE``
    Set to ``1`` to open zarr stores via HTTP range-requests.  Default: ``0``.
``TOLTECA_WEB_PORT``
    Server port.  Default: ``8052``.
``TOLTECA_WEB_DEBUG``
    Set to ``1`` to enable Dash debug mode.  Default: ``0``.
``TOLTECA_DB_URL``
    SQLite or DuckDB URL for analysis-group queries.
    Auto-detected from cache root when not set.

Usage
-----
Via CLI::

    tolteca_web run tolteca_web.portal:create_app --port 8080

Or directly::

    TOLTECA_WEB_CACHE_ROOT=/path/to/cache python -m tolteca_web.portal
"""

from __future__ import annotations

import gzip as _gzip
import io as _io
import os
from pathlib import Path

_HERE = Path(__file__).parent


def create_app():
    """Create and return the configured TolTEC Web v3 Dash application."""
    from dash import Dash, Input, Output, dcc, html
    from flask import request as flask_request
    from flask import send_from_directory

    import dash_mantine_components as dmc  # noqa: F401

    from tolteca_web.obs import (
        DataProdViewerPage,
        KidsDiagViewerPage,
        ObsDataService,
        ParquetCatalogBackend,
        ReducedObsViewerPage,
        TelViewerPage,
    )
    from tolteca_web.sweep import SweepViewerPage

    # ── Configuration ────────────────────────────────────────────────────────

    cache_root = Path(os.environ.get("TOLTECA_WEB_CACHE_ROOT", "cache"))
    catalog_path = Path(
        os.environ.get("TOLTECA_WEB_CATALOG", cache_root / "catalog.parquet")
    )
    zarr_dir = cache_root / "zarr"

    use_http_zarr = os.environ.get("ZARR_HTTP_MODE", "0") == "1"
    port = int(os.environ.get("TOLTECA_WEB_PORT", "8052"))
    zarr_http_base = f"http://localhost:{port}/zarr"

    tel_root = Path(os.environ.get("TOLTECA_WEB_TEL_ROOT", "data_lmt/tel"))

    # Auto-detect DB from cache root
    _default_db: str | None = None
    for _suffix, _scheme in [
        ("tolteca.sqlite", "sqlite"),
        ("tolteca_sim.sqlite", "sqlite"),
        ("tolteca_prod.sqlite", "sqlite"),
        ("tolteca.duckdb", "duckdb"),
    ]:
        _candidate = cache_root / _suffix
        if _candidate.exists():
            _default_db = f"{_scheme}:///{_candidate}"
            break
    db_url: str | None = os.environ.get("TOLTECA_DB_URL", _default_db)

    # ── Data service ─────────────────────────────────────────────────────────

    catalog = ParquetCatalogBackend(catalog_path, cache_root)
    svc = ObsDataService(
        cache_root,
        catalog,
        zarr_base_url=zarr_http_base if use_http_zarr else None,
        db_url=db_url,
    )

    # ── Pages ────────────────────────────────────────────────────────────────

    viewer_page = DataProdViewerPage(svc)
    sweep_page = SweepViewerPage(svc)
    reduced_obs_page = ReducedObsViewerPage(svc)
    kids_diag_page = KidsDiagViewerPage(svc)
    tel_page = TelViewerPage(data_root=tel_root)

    # ── Dash app ─────────────────────────────────────────────────────────────

    # Pre-load plotly.min.js before React starts to avoid a race condition
    # through the VSCode port-forwarding proxy (which drops responses >~2 MB).
    _index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script src="/_dash-component-suites/plotly/package_data/plotly.min.js"></script>
            {%renderer%}
        </footer>
    </body>
</html>"""

    # __name__ == "tolteca_web.portal" so Dash resolves assets/ relative to
    # this file → src/tolteca_web/assets/ (where dagComponentFunctions.js lives)
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="TolTEC Data Viewer",
        index_string=_index_string,
    )

    @app.server.route("/zarr/<path:subpath>")
    def serve_zarr_file(subpath: str):
        return send_from_directory(str(zarr_dir), subpath)

    # Gzip plotly.min.js (4.7 MB) so it passes through the VSCode proxy.
    @app.server.after_request
    def _gzip_large_assets(response):
        if "plotly.min.js" in flask_request.path:
            data = response.get_data()
            buf = _io.BytesIO()
            with _gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
                gz.write(data)
            compressed = buf.getvalue()
            response.set_data(compressed)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = len(compressed)
            response.vary.add("Accept-Encoding")
        return response

    # ── Layout ───────────────────────────────────────────────────────────────

    app.layout = html.Div([
        dcc.Location(id="app-url", refresh=False),
        html.Div(viewer_page.layout(),      id="page-viewer"),
        html.Div(sweep_page.layout(),       id="page-sweep"),
        html.Div(reduced_obs_page.layout(), id="page-reduced-obs"),
        html.Div(kids_diag_page.layout(),   id="page-kids-diag"),
        html.Div(tel_page.layout(),         id="page-tel"),
    ])

    viewer_page.register_callbacks(app)
    sweep_page.register_callbacks(app)
    reduced_obs_page.setup_callbacks(app)
    kids_diag_page.setup_callbacks(app)
    tel_page.setup_callbacks(app)

    # ── Routing ──────────────────────────────────────────────────────────────

    @app.callback(
        Output("page-viewer",      "style"),
        Output("page-sweep",       "style"),
        Output("page-reduced-obs", "style"),
        Output("page-kids-diag",   "style"),
        Output("page-tel",         "style"),
        Input("app-url", "pathname"),
    )
    def _route(pathname: str | None) -> tuple[dict, dict, dict, dict, dict]:
        hide: dict = {"display": "none"}
        show: dict = {}
        p = pathname or "/"
        if p == "/sweep":
            return hide, show, hide, hide, hide
        if p == "/reduced-obs":
            return hide, hide, show, hide, hide
        if p == "/kids-diag":
            return hide, hide, hide, show, hide
        if p == "/tel":
            return hide, hide, hide, hide, show
        return show, hide, hide, hide, hide

    return app


def main() -> None:
    """Run the portal server (used by ``python -m tolteca_web.portal``)."""
    port = int(os.environ.get("TOLTECA_WEB_PORT", "8052"))
    debug = os.environ.get("TOLTECA_WEB_DEBUG", "0") == "1"
    cache_root = Path(os.environ.get("TOLTECA_WEB_CACHE_ROOT", "cache"))
    tel_root = Path(os.environ.get("TOLTECA_WEB_TEL_ROOT", "data_lmt/tel"))

    print("\nTolTEC Web v3")
    print(f"  Cache root : {cache_root.resolve()}")
    print(f"  Zarr mode  : {'HTTP (range-requests)' if os.environ.get('ZARR_HTTP_MODE') == '1' else 'local disk'}")
    print(f"  Port       : {port}")
    print(f"\n  Portal     : http://localhost:{port}/")
    print(f"  Sweep      : http://localhost:{port}/sweep")
    print(f"  Kids       : http://localhost:{port}/reduced-obs")
    print(f"  Diagnostics: http://localhost:{port}/kids-diag")
    print(f"  Tel Viewer : http://localhost:{port}/tel")
    print(f"  Tel root   : {tel_root.resolve()}")
    print()

    app = create_app()
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        threaded=True,
        dev_tools_ui=True,
        dev_tools_props_check=False,
        dev_tools_hot_reload=False,
    )


if __name__ == "__main__":
    main()
