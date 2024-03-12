from flask_sqlalchemy import SQLAlchemy
from tollan.utils.general import ObjectProxy
from tollan.utils.log import logger
from tollan.utils.fmt import pformat_yaml
from copy import deepcopy
from tollan.db import SqlaDB

from typing import TYPE_CHECKING

__all__ = [
    "db",
    "get_sqla_db",
]


if TYPE_CHECKING:
    db: ObjectProxy | SQLAlchemy = None
else:
    db = ObjectProxy(SQLAlchemy)
    """A proxy to the `~flask_sqlalchemy.SQLAlchemy` instance."""


def init_ext(_config):
    return db.proxy_init()


_sqla_dbs: dict[str, SqlaDB] = {}


def init_app(server, config):
    """Initialize database for `server`."""
    server.config.update(SQLALCHEMY_TRACK_MODIFICATIONS=False)

    # extract all upper case entries and update with the new settings
    # make a copy because we will modify it.
    flask_config = {k: deepcopy(v) for k, v in config.items() if k.isupper()}

    logger.debug(f"{config}")
    binds_by_name = {b["name"]: b for b in config["binds"] if b["url"]}

    if "default" not in binds_by_name:
        k, v = next(iter(binds_by_name.items()))
        logger.warning(
            f"use bind {k} as default bind.",
        )
        default_bind_url = v["url"]
    else:
        default_bind_url = binds_by_name["default"]["url"]
    flask_config["SQLALCHEMY_DATABASE_URI"] = default_bind_url
    flask_config["SQLALCHEMY_BINDS"] = {
        name: b["url"] for name, b in binds_by_name.items()
    }
    logger.debug(f"update server config:\n{pformat_yaml(flask_config)}")
    server.config.update(flask_config)
    db.init_app(server)

    @server.teardown_appcontext
    def shutdown_db_session(_exception=None):
        db.session.remove()

    # create sqla dbs from the db
    for name in db.engines:
        _sqla_dbs[name] = SqlaDB.from_flask_sqla(db, bind=name)


def get_sqla_db(bind=None):
    """Return the sqla for `bind`."""
    return _sqla_dbs.get(bind, None)  # noqa: SIM910
