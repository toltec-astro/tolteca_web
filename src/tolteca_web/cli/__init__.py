"""Console script for tolteca web."""

import argparse
import os
import sys
from contextlib import ContextDecorator
from pathlib import Path

import click
from tollan.utils import envfile
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger

__all__ = ["load_env_helper", "run_site", "run_flask"]


class hookit(ContextDecorator):
    """A context manager that allow inject code to object's method.

    Parameters
    ----------
    obj : object
        The object to alter.

    name : str
        The name of the method to hook.

    """

    def __init__(self, obj, name: str):
        self.obj = obj
        self.name = name
        self.func_hooked = getattr(obj, name)

    def set_post_func(self, func):
        """Call `func` after the hooked function.

        Parameters
        ----------
        func : callable
            The function to call after the hooked function.
        """

        def hooked(obj, *args, **kwargs):
            self.func_hooked(obj, *args, **kwargs)
            func(obj, *args, **kwargs)

        setattr(self.obj, self.name, hooked)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        setattr(self.obj, self.name, self.func_hooked)


def load_env_helper():
    """A helper utility to expand env vars defined in systemd environment
    files in shell.
    """
    parser = argparse.ArgumentParser(description="Load systemd env file.")
    parser.add_argument(
        "env_files", metavar="ENV_FILE", nargs="+", help="Path to systemd env file."
    )
    args = parser.parse_args()
    envs = dict()
    for path in args.env_files:
        envs.update(envfile.env_load(path))
    cmd = " ".join(f'{k}="{v}"' for k, v in envs.items())
    # Print the env vars so that it can be captured by the shell
    print(cmd)


def _add_site_env_arg(parser):
    # note that site overrides DASHA_SITE in envfiles.
    parser.add_argument(
        "--site",
        "-s",
        metavar="NAME",
        default=None,
        help="The module name or path to the site. "
        "Examples: ~/mysite.py, mypackage.mysite",
    )
    parser.add_argument(
        "--env_files",
        "-e",
        metavar="ENV_FILE",
        nargs="*",
        help="Path to systemd env file.",
    )

    def handle_site_env_args(args):
        envs = dict()
        for path in args.env_files or tuple():
            envs.update(envfile.env_load(path))
        if args.site is not None:
            envs["DASHA_SITE"] = args.site
        if len(envs) > 0:
            logger.info(f"loaded envs:\n{pformat_yaml(envs)}")
        for k, v in envs.items():
            os.environ[k] = v or ""

    return parser, handle_site_env_args


def _add_ext_arg(parser):
    _all_ext_procs = [
        "flask",
    ]
    parser.add_argument(
        "extension",
        metavar="EXT",
        choices=_all_ext_procs,
        nargs="?",
        default="flask",
        help="The extension process to run"
        " Available options: {}".format(", ".join(_all_ext_procs)),
    )

    def handle_ext_args(args):
        if args.extension == "flask":
            from ..app import create_app

            app = create_app()
            # get port
            port_default = 8050
            port = os.environ.get("FLASK_RUN_PORT", port_default)
            host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
            try:
                port = int(port)
            except Exception:
                port = port_default
            import flask.cli

            # hook the server banner to include a splash screen with dasha info
            with hookit(flask.cli, "show_server_banner") as hk:

                def dasha_splash_screen(*args, **kwargs):
                    click.echo(
                        f"""
~~~~ dasha is running: http://{host}:{port} ~~~~~
"""
                    )

                hk.set_post_func(dasha_splash_screen)
                app.run(host=host, debug=True, port=port)
        else:
            raise NotImplementedError

    return parser, handle_ext_args


def run_site(args=None):
    """A helper utility to run DashA site."""

    parser = argparse.ArgumentParser(description="Run DashA site.")
    parser, handle_site_env_args = _add_site_env_arg(parser)
    parser, handle_ext_args = _add_ext_arg(parser)
    args = parser.parse_args(args=args)
    handle_site_env_args(args)
    handle_ext_args(args)


def main(args=None):
    """Console script for tolteca_web."""
    return run_site(args)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
