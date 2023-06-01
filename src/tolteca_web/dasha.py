import copy
import inspect

import dash_bootstrap_components as dbc
from dash import Dash
from dash_component_template.template import Template
from tollan.utils.fmt import pformat_yaml
from tollan.utils.general import Deferred, getobj, rupdate
from tollan.utils.log import logger, timeit

__all__ = [
    "DashA",
    "dasha",
    "dash_app",
    "resolve_url",
    "get_url_stem",
    "resolve_template",
    "CSS",
]


dash_app = Deferred(Dash)
"""A proxy to the `Dash` instance."""


dasha = Deferred()
"""A proxy to the `DashA` instance."""


class CSS(object):
    """A set of commonly used CSS."""

    themes = dbc.themes
    fa = (
        "https://cdnjs.cloudflare.com/ajax/libs/"
        "font-awesome/6.0.0-beta2/css/all.min.css"
    )


class DashA:
    """This class sets up a Dash app to serve a component template.

    Parameters
    ----------
    config : dict
        The Dash config and template config to use.
        Dash configurations shall be specified as ALL CAPS.
        This object is passed to `from_dict`
        to create the template instance when `init_app` is called.
    """

    _dash_config_default = {
        "SERVE_LOCALLY": True,
        "REQUESTS_PATHNAME_PREFIX": None,
        "ROUTES_PATHNAME_PREFIX": None,
        "EXTERNAL_STYLESHEETS": list(),
        "EXTERNAL_SCRIPTS": list(),
        "META_TAGS": [
            {
                "name": "viewport",
                "content": "width=device-width, initial-scale=1," " shrink-to-fit=no",
            }
        ],
        # default app title
        "TITLE": None,
    }

    def __init__(self, config):
        self.config = copy.deepcopy(self._dash_config_default)
        rupdate(self.config, config)
        self.dash_app = None

    def init_app(self, server):
        def extract_args(config, args):
            result = dict()
            for name in args:
                key = name.upper()
                if key in config:
                    result[name] = config.pop(key)
            return result, config

        def extract_dash_args(config):
            return extract_args(
                config, set(inspect.getfullargspec(Dash.__init__).args[1:])
            )

        def extract_dasha_args(config):
            return extract_args(config, {"DEBUG", "NO_DEFAULT_STYLESHEETS", "THEME"})

        dash_config, config = extract_dash_args(copy.deepcopy(self.config))
        dasha_config, template_config = extract_dasha_args(config)

        # handle default stylesheets and theme
        css_theme = dasha_config.get("THEME", dbc.themes.BOOTSTRAP)
        if dasha_config.get("NO_DEFAULT_STYLESHEETS", False):
            css = list()
        else:
            css = [CSS.fa, css_theme]
        css.extend(dash_config.get("external_stylesheets", list()))
        dash_config["external_stylesheets"] = css

        logger.info(f"Dash config:\n{pformat_yaml(dash_config)}")
        logger.info(f"DashA config:\n{pformat_yaml(dasha_config)}")
        logger.info(f"Template:\n{pformat_yaml(template_config)}")

        app = dash_app.init(
            name=__package__,
            server=server,
            suppress_callback_exceptions=True,
            **dash_config,
        )

        serve_locally = dash_config["serve_locally"]
        app.scripts.config.serve_locally = serve_locally
        app.css.config.serve_locally = serve_locally

        # dev tools
        if dasha_config.get("DEBUG", False):
            app.enable_dev_tools(debug=True),

        with server.app_context():
            template = resolve_template(config)
            with timeit("setup layout"):
                template.setup_layout(app)
                # try infer a title if title is not set
                if app.title is None:
                    app.title = getattr(template, "title_text", "Dash App")
            with timeit("serve layout"):
                app.layout = template.layout
        return server


def init_ext(config):
    ext = dasha.init(DashA(config))
    return ext


def init_app(server, config):
    return dasha.init_app(server)


def resolve_url(path):
    """Expands an internal URL to include prefix the app is mounted at."""
    routes_prefix = dash_app.config.routes_pathname_prefix or ""
    return f"{routes_prefix}{path}".replace("//", "/")


def _ensure_prefix(s, p):
    """Return a new string with prefix `p` if it does not."""
    if s.startswith(p):
        return s
    return f"{p}{s}"


def get_url_stem(path):
    """The inverse of `resolve_url`."""
    routes_prefix = dash_app.config.routes_pathname_prefix or ""
    if routes_prefix == "":
        return path
    routes_prefix = _ensure_prefix(routes_prefix.strip("/"), "/")
    path = _ensure_prefix(path, "/")
    if path.startswith(routes_prefix):
        path = path.replace(routes_prefix, "", 1)
    return _ensure_prefix(path, "/")


def resolve_template(arg):
    """Return the component template specified in config."""

    if isinstance(arg, Template):
        return arg
    if isinstance(arg, dict):
        arg = copy.copy(arg)
        cls = arg.pop("template")
        if isinstance(cls, str):
            cls = getobj(cls)
        # in case the cls is a module, check for
        # the _resolve_template field
        if inspect.ismodule(cls):
            temp_attr = getattr(cls, "_resolve_template", None)
            if temp_attr is None:
                raise ValueError(f"cannot resolve template in module {cls}")
            if isinstance(temp_attr, str):
                cls = getattr(cls, temp_attr)
            else:
                cls = temp_attr
        if issubclass(cls, Template):
            return cls(**arg)
        raise ValueError(f"invalid template class {cls}")
    raise ValueError(f"cannot resolve template from {arg}")
