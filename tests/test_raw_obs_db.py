from pathlib import Path

import pytest
import sqlalchemy as sa

from tolteca_web.data_prod.raw_obs_db import ToltecRawObsDB


@pytest.fixture()
def raw_obs_db():
    toltecdb_filepath = Path(__file__).parent.joinpath("data/toltecdb/toltecdb.sqlite")
    toltecdb_url = f"sqlite:///{toltecdb_filepath}"
    return ToltecRawObsDB(toltecdb_url)


def test_db_alive(raw_obs_db):
    with raw_obs_db.session_context() as session:
        assert session.execute(sa.text("select 1;"))


def test_obs_query_grouped(raw_obs_db):
    result = raw_obs_db.obs_query_grouped(
        obsnum=17859,
        valid_key="any",
    )
    assert len(result) == 1
    assert result["obsnum"][0] == 17859
    assert result["all_valid"][0] == 0
    assert result["any_valid"][0] == 1

    result = raw_obs_db.obs_query_grouped(
        obsnum=17859,
        valid_key="all",
    )
    assert len(result) == 0
