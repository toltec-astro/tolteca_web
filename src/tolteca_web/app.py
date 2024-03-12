import atexit
import os
import sys
from contextlib import ExitStack

import flask
from tollan.utils.general import getobj
from tollan.utils.log import logger, logit, timeit

__all__ = ["create_app"]


# enable logging for the start up if flask development is set
if (
    (os.environ.get("FLASK_ENV", None) == "development")
    or os.environ.get("DASH_DEBUG", None)
    or os.environ.get("FLASK_DEBUG", None)
):
    loglevel = "DEBUG"
else:
    loglevel = "INFO"
logger.remove()
logger.add(sys.stderr, level=loglevel)


exit_stack = ExitStack()
""""
An `~contextlib.ExitStack` instance that can be used to register clean up
functions.
"""


@timeit
def create_app():
    """Flask entry point."""
    envs = {
        "SECRET_KEY": "toltec",
    }
    envs = envs | os.environ

    server = flask.Flask(__package__)
    server.config.update(
        {
            "SECRET_KEY": envs["SECRET_KEY"],
        },
    )

    site = getattr(getobj(envs["DASHA_SITE"]), "DASHA_SITE", None)
    if callable(site):
        site = site()
    if site is None:
        raise ValueError(f'No valid DASHA_SITE found in {envs["DASHA_SITE"]}.')
    # load exts
    from . import db, dasha

    exts = {}
    for ext_name, ext in {"db": db, "dasha": dasha}.items():
        if ext_name in site:
            ext.init_ext(site[ext_name])
            exts[ext_name] = ext

    with server.app_context():
        for ext_name, ext in exts.items():
            ext.init_app(server, site[ext_name])

        post_init = site.get("post_init", None)
        if post_init is not None:
            timeit(post_init)()
        from werkzeug.middleware.proxy_fix import ProxyFix

    server.wsgi_app = ProxyFix(server.wsgi_app, x_proto=1, x_host=1)
    return server


def _exit():
    with logit(logger.info, "dasha clean up"):
        exit_stack.close()


atexit.register(_exit)
