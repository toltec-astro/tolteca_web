from typing import ClassVar

import pandas as pd
import sqlalchemy.sql.expression as se
from sqlalchemy.sql import alias as sqla_alias
from sqlalchemy.sql import func as sqla_func
from tollan.db import SqlaDB
from tollan.db.func import create_datetime
from tollan.utils.log import logger

from .conventions import make_toltec_raw_obs_sweep_obs_uid, make_toltec_raw_obs_uid


class ToltecRawObsDB:
    """A class to query toltec raw observation database."""

    _db: SqlaDB

    def __init__(self, db: str | SqlaDB):
        if isinstance(db, str):
            db = SqlaDB.from_url(db)
        self._db = db
        self._db.reflect_tables()

    @property
    def db(self):
        """The underlying db connection."""
        return self._db

    def session_context(self, *args, **kwargs):
        """Return the session context to the underlying db."""
        return self._db.session_context(*args, **kwargs)

    _parse_dates_keys: ClassVar[list[str]] = ["Date", "DateTime"]
    _dp_raw_obs_group_keys: ClassVar[list[str]] = [
        "master",
        "obsnum",
        "subobsnum",
        "scannum",
        "repeat",
    ]

    def query_obs_latest(
        self,
        master=None,
        obs_type=None,
        table_name="toltec",
        valid_only=True,
    ):
        """Return the latest obs."""
        tname = table_name
        t = self.db.tables
        where = [True]
        if valid_only:
            where.append(t[tname].c.Valid > 0)
        if master is not None:
            where.append(t["master"].c.label == master.upper())
        if obs_type is not None:
            where.append(t["obstype"].c.label == obs_type)
        where_clause = se.and_(*where)
        cols_map = {
            "master": t["master"].c.label,
            "obsnum": t[tname].c.ObsNum,
            "subobsnum": t[tname].c.SubObsNum,
            "scannum": t[tname].c.ScanNum,
            "obs_type": t["obstype"].c.label,
            "id": t[tname].c.id,
        }
        stmt = (
            se.select(*cols_map.values())
            .select_from(
                t[tname]
                .join(t["obstype"], onclause=(t[tname].c.ObsType == t["obstype"].c.id))
                .join(t["master"], onclause=(t[tname].c.Master == t["master"].c.id)),
            )
            .where(where_clause)
            .order_by(se.desc(t[tname].c.id))
            .limit(1)
        )
        with self.db.session_context() as session:
            r = dict(
                zip(
                    cols_map.keys(),
                    session.execute(stmt).fetchone(),
                    strict=True,
                ),
            )
        logger.debug(
            f'latest obs: uid={make_toltec_raw_obs_uid(r)} obs_type={r["obs_type"]}',
        )
        return r

    def query_obs_group_latest(
        self,
        master=None,
        obs_type=None,
        table_name="toltec",
    ):
        """Return the latest obs group."""
        r = self.query_obs_latest(
            master=master,
            obs_type=obs_type,
            table_name=table_name,
            valid_only=False,
        )
        id_start, id_end = self.query_group_id_range_for_id(
            r.pop("id"),
            table_name=table_name,
        )
        r["id_start"] = id_start
        r["id_end"] = id_end
        return r

    def query_id_range_from_time_range(
        self,
        time_start=None,
        time_end=None,
        table_name="toltec",
        valid_only=True,
    ):
        """Return the id range for time range."""
        tname = table_name
        t = self.db.tables
        where = [True]
        if valid_only:
            where.append(t[tname].c.Valid > 0)

        if time_start is not None:
            where.append(
                create_datetime(t[tname].c.Date, t[tname].c.Time) >= time_start,
            )
        if time_end is not None:
            where.append(
                create_datetime(t[tname].c.Date, t[tname].c.Time) <= time_end,
            )
        where_clause = se.and_(*where)
        cols_map = {
            "id_min": sqla_func.min(t[tname].c.id),
            "id_max": sqla_func.max(t[tname].c.id),
        }
        stmt = (
            se.select(*cols_map.values())
            .select_from(t[tname])
            .where(where_clause)
            .order_by(se.desc(t[tname].c.id))
            .limit(1)
        )
        with self.db.session_context() as session:
            r = dict(
                zip(
                    cols_map.keys(),
                    session.execute(stmt).fetchone(),
                    strict=True,
                ),
            )
        id_start, id_end = r["id_min"], r["id_max"] + 1
        logger.debug(
            f"query id time_start={time_start} time_end={time_end}: "
            f"id_start={id_start} id_end={id_end}",
        )
        return id_start, id_end

    def query_group_id_range_from_time_range(
        self,
        time_start=None,
        time_end=None,
        table_name="toltec",
    ):
        """Return group id range for time range."""
        id_start, id_end = self.query_id_range_from_time_range(
            time_start=time_start,
            time_end=time_end,
            table_name=table_name,
            valid_only=False,
        )
        grouped_id_start = self.query_group_id_range_for_id(
            id_start,
            table_name=table_name,
        )[0]
        grouped_id_end = self.query_group_id_range_for_id(
            id_end - 1,
            table_name=table_name,
        )[1]
        logger.debug(
            f"query grouped id: id_start={id_start} -> {grouped_id_start} "
            f"id_end={id_end} -> {grouped_id_end}",
        )
        return grouped_id_start, grouped_id_end

    def query_group_id_range_for_id(self, id, table_name="toltec"):
        """Return the group id range for given id."""
        # this is useful to include all items of a group.
        tname = table_name
        t = self.db.tables
        # query to get obsnum, subobsnum, scannum and master
        cols_map = {
            "master_id": t[tname].c.Master,
            "obsnum": t[tname].c.ObsNum,
            "subobsnum": t[tname].c.SubObsNum,
            "scannum": t[tname].c.ScanNum,
        }
        stmt = (
            se.select(*cols_map.values())
            .select_from(t[tname])
            .where(t[tname].c.id == id)
            .limit(1)
        )
        with self.db.session_context() as session:
            r = dict(
                zip(
                    cols_map.keys(),
                    session.execute(stmt).fetchone(),
                    strict=True,
                ),
            )
        # do query to get id range
        where = [
            t[tname].c.Master == r["master_id"],
            t[tname].c.ObsNum == r["obsnum"],
            t[tname].c.SubObsNum == r["subobsnum"],
            t[tname].c.ScanNum == r["scannum"],
        ]
        where_clause = se.and_(*where)
        cols_map = {
            "id_min": sqla_func.min(t[tname].c.id),
            "id_max": sqla_func.max(t[tname].c.id),
        }
        stmt = se.select(*cols_map.values()).select_from(t[tname]).where(where_clause)
        with self.db.session_context() as session:
            r = dict(
                zip(cols_map.keys(), session.execute(stmt).fetchone(), strict=True),
            )
        id_start, id_end = r["id_min"], r["id_max"] + 1
        logger.debug(
            f"query group id range for {id=}: id_start={id_start} id_end={id_end}",
        )
        return id_start, id_end

    def _resovle_id_range(  # noqa: PLR0913
        self,
        id,
        master=None,
        obs_type=None,
        table_name="toltec",
        valid_only=True,
    ):
        """Return a range of id for various input formats."""
        # resolve id
        if isinstance(id, int):
            id = slice(id, id + 1)
        elif isinstance(id, slice):
            if id.step is not None and id.step != 1:
                raise ValueError(
                    "id slice has to be contiguous and incrementing",
                )
        elif id is None:
            id = slice(-1, None)
        else:
            raise ValueError("id has to be None, int or slice")
        if id.stop is None:
            # query for latest id
            obs_latest = self.query_obs_latest(
                master=master,
                obs_type=obs_type,
                table_name=table_name,
                valid_only=valid_only,
            )
            id_latest = obs_latest["id"]
        else:
            id_latest = id.stop
        # resolve the id slice to id range
        return range(*id.indices(id_latest + 1))

    def id_query_grouped(  # noqa: PLR0913
        self,
        id=None,
        table_name="toltec",
        master=None,
        obs_type=None,
        valid_only=True,
        valid_key="any",
        n_items=None,
    ):
        """Return the group info for id."""
        id_range = self._resovle_id_range(
            id,
            master=master,
            obs_type=obs_type,
            table_name=table_name,
            valid_only=False,
        )
        tname = table_name
        t = self.db.tables

        where = [
            t[tname].c.id >= id_range.start,
            t[tname].c.id < id_range.stop,
        ]
        # herethe validstate is checked on the group valid state
        having = [True]
        all_valid = sqla_func.bit_and(t[tname].c.Valid).label("all_valid")
        any_valid = sqla_func.bit_or(t[tname].c.Valid).label("any_valid")
        if valid_only:
            if valid_key == "all":
                having.append(all_valid > 0)
            elif valid_key == "any":
                having.append(any_valid > 0)
            else:
                raise ValueError("invalid valid_key.")
        if master is not None:
            where.append(t["master"].c.label == master.upper())
        if obs_type is not None:
            where.append(t["obstype"].c.label == obs_type)
        select_cols = [
            sqla_func.min(t[tname].c.id).label("id_min"),
            sqla_func.max(t[tname].c.id).label("id_max"),
            sqla_func.max(create_datetime(t[tname].c.Date, t[tname].c.Time)).label(
                "time_obs",
            ),
            sqla_func.max(t[tname].c.ObsNum).label("obsnum"),
            sqla_func.max(t[tname].c.SubObsNum).label("subobsnum"),
            sqla_func.max(t[tname].c.ScanNum).label("scannum"),
            sqla_func.max(t[tname].c.RepeatLevel).label("repeat"),
            sqla_func.max(t["obstype"].c.label).label("obs_type"),
            sqla_func.max(t["master"].c.label).label("master"),
            # sqla_func.array_agg(t[tname].c.RoachIndex).label("roaches"),
            sqla_func.count(t[tname].c.id).label("n_data_items"),
            all_valid,
            any_valid,
        ]
        select_tbl = (
            t[tname]
            .join(t["obstype"], onclause=(t[tname].c.ObsType == t["obstype"].c.id))
            .join(t["master"], onclause=(t[tname].c.Master == t["master"].c.id))
        )

        stmt = (
            se.select(*select_cols)
            .select_from(select_tbl)
            .where(se.and_(*where))
            .group_by(
                t[tname].c.Master.label("master_id"),
                t[tname].c.ObsNum.label("obsnum"),
                t[tname].c.SubObsNum.label("subobsnum"),
                t[tname].c.ScanNum.label("scannum"),
            )
            .order_by(
                sqla_func.min(t[tname].c.id).desc(),
            )
            .having(se.and_(*having))
        )
        if n_items is not None:
            stmt = stmt.limit(n_items)
        with self.db.session_context() as session:
            result = pd.read_sql_query(
                stmt,
                con=session.bind,
                parse_dates=self._parse_dates_keys,
            )
        logger.debug(f"query group result: \n{result}")
        return result

    def id_query(  # noqa: PLR0913
        self,
        id=None,
        table_name="toltec",
        with_cal_info=True,
        master=None,
        obs_type=None,
        valid_only=True,
    ):
        """Return the items for id."""
        id_range = self._resovle_id_range(
            id,
            master=master,
            obs_type=obs_type,
            table_name=table_name,
            valid_only=valid_only,
        )
        logger.debug(f"query toltecdb for id [{id_range.start}:{id_range.stop}]")

        tname = table_name
        t = self.db.tables
        # add constaint for subobsnums and scannum
        where = [
            t[tname].c.id >= id_range.start,
            t[tname].c.id < id_range.stop,
        ]
        if valid_only:
            where.append(t[tname].c.Valid > 0)
        if master is not None:
            where.append(t["master"].c.label == master.upper())
        if obs_type is not None:
            where.append(t["obstype"].c.label == obs_type)
        select_cols = [
            create_datetime(t[tname].c.Date, t[tname].c.Time).label("time_obs"),
            t[tname].c.ObsNum.label("obsnum"),
            t[tname].c.SubObsNum.label("subobsnum"),
            t[tname].c.ScanNum.label("scannum"),
            t[tname].c.RoachIndex.label("roach"),
            t[tname].c.RepeatLevel.label("repeat"),
            t[tname].c.TargSweepObsNum.label("cal_obsnum"),
            t[tname].c.TargSweepSubObsNum.label("cal_subobsnum"),
            t[tname].c.TargSweepScanNum.label("cal_scannum"),
            t["obstype"].c.label.label("obs_type"),
            t["master"].c.label.label("master"),
            t[tname].c.FileName.label("source"),
        ]
        select_tbl = (
            t[tname]
            .join(t["obstype"], onclause=(t[tname].c.ObsType == t["obstype"].c.id))
            .join(t["master"], onclause=(t[tname].c.Master == t["master"].c.id))
        )
        if with_cal_info:
            # build stmt to join with cal table
            t_cal = sqla_alias(t[tname])
            t_cal_master = sqla_alias(t["master"])
            select_cols.extend(
                [
                    t_cal_master.c.label.label("cal_master"),
                    t_cal.c.FileName.label("cal_source"),
                ],
            )
            select_tbl = select_tbl.join(
                t_cal,
                onclause=(
                    se.and_(
                        t[tname].c.TargSweepObsNum == t_cal.c.ObsNum,
                        t[tname].c.TargSweepSubObsNum == t_cal.c.SubObsNum,
                        t[tname].c.TargSweepScanNum == t_cal.c.ScanNum,
                        t[tname].c.RoachIndex == t_cal.c.RoachIndex,
                        # t[tname].c.Master == t_cal.c.Master,
                    )
                ),
                isouter=True,
            ).join(
                t_cal_master,
                onclause=(t_cal.c.Master == t_cal_master.c.id),
                isouter=True,
            )
        stmt = se.select(*select_cols).select_from(select_tbl).where(se.and_(*where))
        logger.debug(f"stmt={stmt}")
        with self.db.session_context() as session:
            df_raw_obs = pd.read_sql_query(
                stmt,
                con=session.bind,
                parse_dates=self._parse_dates_keys,
            )
            logger.debug(f"{df_raw_obs}")
        return df_raw_obs

    def obs_query(  # noqa: PLR0913, C901
        self,
        master,
        obsnum=None,
        subobsnum=None,
        scannum=None,
        obs_type=None,
        table_name="toltec",
        with_cal_info=True,
        valid_only=True,
    ):
        """Return the raw obs."""

        # handle obsnum, subobsnu, and scannums.
        # when they are not present, the latest is returned.
        def _validate_num_arg(name, value, allow_negative=False):
            if value is None:
                return slice(-1, None)
            if isinstance(value, int):
                return slice(value, value + 1)
            if isinstance(value, slice):
                if value.step is not None and value.step != 1:
                    raise ValueError(
                        f"{name} slice has to be contiguous and incrementing",
                    )
                if allow_negative:
                    return value
                # check if start and stop are negative
                if value.start < 0 or value.stop < 0:
                    raise ValueError(f"{name} slice start/stop has to be non-negative")
            raise ValueError(f"{name} has to be None, int or slice")

        obsnum = _validate_num_arg("obsnum", obsnum, allow_negative=True)
        subobsnum = _validate_num_arg("subobsnum", subobsnum)
        scannum = _validate_num_arg("scannum", scannum)
        logger.debug(
            f"query obs {obsnum=} {subobsnum=} {scannum=} {master=} {obs_type=}",
        )
        if obsnum.stop is None:
            # query for latest obsnum
            obs_latest = self.query_obs_latest(
                master=master,
                obs_type=obs_type,
                valid_only=valid_only,
            )
            logger.debug(f"latest obs: {obs_latest}")
            obsnum_stop = obs_latest["obsnum"] + 1
        else:
            obsnum_stop = obsnum.stop
        # resolve the obsnum slice
        obsnum_range = range(*obsnum.indices(obsnum_stop))
        logger.debug(
            f"query toltecdb for obsnum [{obsnum_range.start}:{obsnum_range.stop}] "
            f"to find id range",
        )

        # run a query to figure out actual id for obsnum_since to obsnum_latest
        tname = table_name
        t = self.db.tables
        where = [
            t[tname].c.ObsNum >= obsnum_range.start,
            t[tname].c.ObsNum < obsnum_range.stop,
            t["master"].c.label == master.upper(),
        ]
        if subobsnum.start is not None:
            where.append(t[tname].c.SubObsNum >= subobsnum.start)
        if subobsnum.stop is not None:
            where.append(t[tname].c.SubObsNum < subobsnum.stop)
        if scannum.start is not None:
            where.append(t[tname].c.ScanNum >= scannum.start)
        if scannum.stop is not None:
            where.append(t[tname].c.ScanNum < scannum.stop)
        if valid_only:
            where.append(t[tname].c.Valid > 0)
        if obs_type is not None:
            where.append(t["obstype"].c.label == obs_type)
        stmt = (
            se.select(
                sqla_func.min(t[tname].c.id).label("id_min"),
                sqla_func.max(t[tname].c.id).label("id_max"),
                sqla_func.max(t[tname].c.ObsNum).label("obsnum"),
                sqla_func.max(t[tname].c.SubObsNum).label("subobsnum"),
                sqla_func.max(t[tname].c.ScanNum).label("scannum"),
            )
            .select_from(
                t[tname]
                .join(t["obstype"], onclause=(t[tname].c.ObsType == t["obstype"].c.id))
                .join(t["master"], onclause=(t[tname].c.Master == t["master"].c.id)),
            )
            .where(se.and_(*where))
            .group_by(
                t[tname].c.Master.label("master_id"),
                t[tname].c.ObsNum.label("obsnum"),
                t[tname].c.SubObsNum.label("subobsnum"),
                t[tname].c.ScanNum.label("scannum"),
            )
        )
        # .order_by(
        #         se.desc(t[tname].c.id)
        # )
        with self.db.session_context() as session:
            df_group_ids = pd.read_sql_query(
                stmt,
                con=session.bind,
                parse_dates=self._parse_dates_keys,
            )
        if len(df_group_ids) == 0:
            # no data found
            return None
        id_min = df_group_ids["id_min"].min()
        id_max = df_group_ids["id_max"].max()
        logger.debug(
            f"id range: [{id_min}, {id_max + 1}] "
            f"obsnum range: "
            f"[{df_group_ids['obsnum'].min()}, {df_group_ids['obsnum'].max() + 1}]"
            f"subobsnum range: [{df_group_ids['subobsnum'].min()}, "
            f"{df_group_ids['subobsnum'].max() + 1}]"
            f"scannum range: [{df_group_ids['scannum'].min()}, "
            f"{df_group_ids['scannum'].max() + 1}]",
        )
        df_raw_obs = self.id_query(
            id=slice(id_min, id_max + 1),
            master=master,
            obs_type=obs_type,
            table_name=table_name,
            with_cal_info=with_cal_info,
            valid_only=valid_only,
        )
        df_raw_obs.sort_values(by=self._dp_raw_obs_group_keys)
        return df_raw_obs

    def obs_query_grouped(  # noqa: PLR0915, PLR0913, C901, PLR0912
        self,
        obsnum=None,
        subobsnum=None,
        scannum=None,
        table_name="toltec",
        master=None,
        obs_type=None,
        valid_only=True,
        valid_key="any",
        n_items=None,
    ):
        """Return raw obs grouped."""

        # handle obsnum, subobsnu, and scannums.
        # when they are not present, the latest is returned.
        def _validate_num_arg(name, value, allow_negative=False):
            if value is None:
                return slice(-1, None)
            if isinstance(value, int):
                return slice(value, value + 1)
            if isinstance(value, slice):
                if value.step is not None and value.step != 1:
                    raise ValueError(
                        f"{name} slice has to be contiguous and incrementing",
                    )
                if allow_negative:
                    return value
                # check if start and stop are negative
                if value.start < 0 or value.stop < 0:
                    raise ValueError(f"{name} slice start/stop has to be non-negative")
            raise ValueError(f"{name} has to be None, int or slice")

        obsnum = _validate_num_arg("obsnum", obsnum, allow_negative=True)
        subobsnum = _validate_num_arg("subobsnum", subobsnum)
        scannum = _validate_num_arg("scannum", scannum)
        logger.debug(
            f"query obs group {obsnum=} {subobsnum=} {scannum=} {master=} {obs_type=}",
        )
        if obsnum.stop is None:
            # query for latest obsnum
            obs_latest = self.query_obs_latest(
                master=master,
                obs_type=obs_type,
                valid_only=valid_only,
            )
            logger.debug(f"latest obs: {obs_latest}")
            obsnum_stop = obs_latest["obsnum"] + 1
        else:
            obsnum_stop = obsnum.stop
        # resolve the obsnum slice
        obsnum_range = range(*obsnum.indices(obsnum_stop))
        logger.debug(
            f"query toltecdb for groups of obsnum "
            f"[{obsnum_range.start}:{obsnum_range.stop}]",
        )

        tname = table_name
        t = self.db.tables

        where = [
            t[tname].c.ObsNum >= obsnum_range.start,
            t[tname].c.ObsNum < obsnum_range.stop,
        ]
        if subobsnum.start is not None:
            where.append(t[tname].c.SubObsNum >= subobsnum.start)
        if subobsnum.stop is not None:
            where.append(t[tname].c.SubObsNum < subobsnum.stop)
        if scannum.start is not None:
            where.append(t[tname].c.ScanNum >= scannum.start)
        if scannum.stop is not None:
            where.append(t[tname].c.ScanNum < scannum.stop)
        if obs_type is not None:
            where.append(t["obstype"].c.label == obs_type)

        # herethe valid state is checked on the group valid state
        having = [True]
        all_valid = sqla_func.bit_and(t[tname].c.Valid).label("all_valid")
        any_valid = sqla_func.bit_or(t[tname].c.Valid).label("any_valid")
        if valid_only:
            if valid_key == "all":
                having.append(all_valid > 0)
            elif valid_key == "any":
                having.append(any_valid > 0)
            else:
                raise ValueError("invalid valid_key.")

        if master is not None:
            where.append(t["master"].c.label == master.upper())
        if obs_type is not None:
            where.append(t["obstype"].c.label == obs_type)

        select_cols = [
            sqla_func.min(t[tname].c.id).label("id_min"),
            sqla_func.max(t[tname].c.id).label("id_max"),
            sqla_func.max(create_datetime(t[tname].c.Date, t[tname].c.Time)).label(
                "time_obs",
            ),
            sqla_func.max(t[tname].c.ObsNum).label("obsnum"),
            sqla_func.max(t[tname].c.SubObsNum).label("subobsnum"),
            sqla_func.max(t[tname].c.ScanNum).label("scannum"),
            sqla_func.max(t[tname].c.RepeatLevel).label("repeat"),
            sqla_func.max(t["obstype"].c.label).label("obs_type"),
            sqla_func.max(t["master"].c.label).label("master"),
            # sqla_func.array_agg(t[tname].c.RoachIndex).label("roaches"),
            sqla_func.count(t[tname].c.id).label("n_data_items"),
            all_valid,
            any_valid,
        ]
        select_tbl = (
            t[tname]
            .join(t["obstype"], onclause=(t[tname].c.ObsType == t["obstype"].c.id))
            .join(t["master"], onclause=(t[tname].c.Master == t["master"].c.id))
        )

        stmt = (
            se.select(*select_cols)
            .select_from(select_tbl)
            .where(se.and_(*where))
            .group_by(
                t[tname].c.Master.label("master_id"),
                t[tname].c.ObsNum.label("obsnum"),
                t[tname].c.SubObsNum.label("subobsnum"),
                t[tname].c.ScanNum.label("scannum"),
            )
            .order_by(
                sqla_func.min(t[tname].c.id).desc(),
            )
            .having(se.and_(*having))
        )
        if n_items is not None:
            stmt = stmt.limit(n_items)

        with self.db.session_context() as session:
            result = pd.read_sql_query(
                stmt,
                con=session.bind,
                parse_dates=self._parse_dates_keys,
            )
        logger.debug(f"query obs group result: \n{result}")
        return result

    @staticmethod
    def _normalize_meta(meta):
        def _conv(v):
            # fix time_obs type for json serialization
            if isinstance(v, pd.Timestamp):
                return v.to_pydatetime().isoformat()
            return v

        return {k: _conv(v) for k, v in meta.items()}

    def _make_data_item_from_entry(self, entry):
        # TODO: handle source/filepath resolving
        meta = self._normalize_meta(entry.copy())
        interface = meta["interface"] = f'toltec{meta["roach"]}'
        meta["name"] = f"{interface}-{make_toltec_raw_obs_uid(meta)}"
        if meta.get("cal_source", None) is not None:
            meta["cal_filepath"] = meta["cal_source"]
            meta["cal_name"] = f"{interface}-{make_toltec_raw_obs_sweep_obs_uid(meta)}"
        else:
            meta["cal_filepath"] = meta["cal_name"] = None
        return {"meta": meta, "filepath": meta["source"]}

    def _make_dp_index_for_raw_obs(self, df):
        # create dp index for single raw obs
        df.sort_values(by=["roach"])
        entries = df.to_dict(orient="records")
        meta = {
            "data_prod_type": "dp_raw_obs",
            "name": make_toltec_raw_obs_uid(entries[0]),
            **{k: entries[0][k] for k in self._dp_raw_obs_group_keys},
        }
        data_items = [self._make_data_item_from_entry(entry) for entry in entries]
        return {"meta": meta, "data_items": data_items}

    def make_dp_index_from_query_result(self, df_raw_obs):
        """Return data prod index."""
        master = df_raw_obs["master"][0]
        obsnum_min = df_raw_obs["obsnum"].min()
        obsnum_max = df_raw_obs["obsnum"].max()
        n_entries = len(df_raw_obs)
        meta = {
            "data_prod_type": "dp_named_group",
            "name": f"g_{master}_{obsnum_min}_{obsnum_max}",
        }
        data_items = []
        logger.debug(
            f"collect {n_entries} entries from toltec files db "
            f"obsnum range [{obsnum_min}:{obsnum_max}",
        )
        grouped = df_raw_obs.groupby(by=self._dp_raw_obs_group_keys)
        for _, df in grouped:
            data_items.append(self._make_dp_index_for_raw_obs(df))
        return {"meta": meta, "data_items": data_items}

    @staticmethod
    def _squeeze_index(index, squeeze):
        if squeeze and len(index["data_items"]) == 1:
            return index["data_items"][0]
        return index

    def get_dp_index_for_obs(self, squeeze=False, **kwargs):
        """Return data prod index of given obs."""
        index = self.make_dp_index_from_query_result(self.obs_query(**kwargs))
        return self._squeeze_index(index, squeeze)

    def get_dp_index_for_id(self, squeeze=False, **kwargs):
        """Return data prod index of given id."""
        index = self.make_dp_index_from_query_result(self.id_query(**kwargs))
        return self._squeeze_index(index, squeeze)