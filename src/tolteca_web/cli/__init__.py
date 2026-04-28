"""CLI for tolteca_web — start the Dash server."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

app = typer.Typer(
    name="tolteca_web",
    help="TolTEC web application framework.",
    no_args_is_help=True,
)


@app.command()
def run(
    app_module: Annotated[
        str,
        typer.Argument(
            help=(
                "Python import path of the Dash app factory or module. "
                "E.g. 'tolteca_web.demo:create_app' or 'myproject.app'."
            )
        ),
    ],
    host: Annotated[str, typer.Option("--host", "-H", help="Bind host.")] = "0.0.0.0",  # noqa: S104
    port: Annotated[int, typer.Option("--port", "-p", help="Bind port.")] = 8050,
    debug: Annotated[bool, typer.Option("--debug/--no-debug", help="Debug mode.")] = False,  # noqa: FBT002
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", "-e", help="Dotenv file to load before starting."),
    ] = None,
) -> None:
    """Start the Dash server for the given app module.

    The APP_MODULE argument can be either:

    \b
    * A module path whose top-level ``server`` attribute is a Flask app, or
    * A ``module:factory`` path where ``factory()`` returns a Dash app.

    Examples
    --------
    \b
    $ tolteca_web run tolteca_web.demo:create_app
    $ tolteca_web run tolteca_web.demo:create_app --port 9000 --debug
    """
    if env_file is not None:
        _load_env_file(env_file)

    dash_app = _resolve_app(app_module)
    logger.info("Starting server on http://{}:{}/", host, port)
    dash_app.run(host=host, port=port, debug=debug)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_env_file(path: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv(path)
        logger.info("Loaded env file: {}", path)
    except ImportError:
        logger.warning("python-dotenv not installed; skipping env file {}.", path)


def _resolve_app(spec: str) -> object:
    """Resolve a ``module`` or ``module:factory`` spec to a Dash app."""
    import importlib

    if ":" in spec:
        module_path, factory_name = spec.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        factory = getattr(mod, factory_name)
        return factory()
    mod = importlib.import_module(spec)
    if hasattr(mod, "app"):
        return mod.app
    if hasattr(mod, "server"):
        return mod.server
    msg = f"Module '{spec}' has no 'app' or 'server' attribute and no factory was specified."
    raise typer.BadParameter(msg)


@app.command()
def version() -> None:
    """Print the tolteca_web version and exit."""
    from tolteca_web._version import __version__  # type: ignore[import-not-found]

    typer.echo(f"tolteca_web {__version__}")


if __name__ == "__main__":
    app()

