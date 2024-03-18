import collections
import functools
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from pathlib import Path
from typing import ClassVar, Protocol

import pandas as pd
import sqlalchemy.exc as sae
from astropy.utils.console import human_time
from tollan.db import SqlaDB
from tollan.utils.fileloc import FileLoc
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger, timeit
from tolteca_datamodels.toltec.file import guess_meta_from_source
from tolteca_datamodels.toltec.types import ToltecDataKind

from .cache import YamlFileIndex
from .raw_obs_db import ToltecRawObsDB

__all__ = [
    "DataProdCollectorProtocol",
    "QLDataProdCollector",
]


class DataProdCollectorProtocol(Protocol):
    """A protocol class for data prod collectors."""

    def collect():
        """Collect data products."""

    @property
    def data_prod_index_store() -> YamlFileIndex:
        """The data prodroct index store."""


@dataclass
class DataProdCollectorInfo:
    is_active: bool
    message: None | str = None
    query_cursor: None | dict = None


@functools.lru_cache
def _guess_meta_from_source_cached(source):
    # TODO: make the data prod index name more
    # standard.
    file_loc = FileLoc(source)
    filepath = file_loc.path
    if filepath.suffix == ".yaml" and filepath.name.startswith("dp_toltec"):
        return {"source": source, "file_loc": file_loc, "file_ext": "yaml"}
    return guess_meta_from_source(source)


class DataProdType(Enum):
    dp_cal_group = auto()
    dp_drivefit = auto()
    dp_focus_group = auto()


class DataProdAssocType(Enum):
    dpa_cal_group_obs = auto()
    dpa_drivefit_obs = auto()
    dpa_focus_group_obs = auto()


_collator_cls_registry = set()


class GroupFlag(Flag):
    implicit = auto()
    explicit_start = auto()
    explicit_end = auto()
    explicit = explicit_start | explicit_end
    implicit_start = implicit | explicit_end
    implicit_end = implicit | explicit_start


@dataclass
class Group:
    flag: GroupFlag = GroupFlag.implicit
    items: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def append(self, item, add_flag=None):
        if add_flag is not None:
            self.flag |= flag
        self.items.append(item)


class CollatorBase:

    def __init_subclass__(cls, *args, **kwargs):
        if not cls.__dict__.get("__abstract__", False):
            _collator_cls_registry.add(cls)
        return super().__init_subclass__(*args, **kwargs)

    data_prod_type: ClassVar[DataProdType]
    data_prod_assoc_type: ClassVar[DataProdAssocType]

    def _make_item_groups(self, data_prods):
        return NotImplemented

    def _make_assocs(self, group):
        return [
            {"data_prod_assoc_type": self.data_prod_assoc_type.name, "index": item}
            for item in group.items
        ]

    def _make_data_items(self, group):
        return group.items

    def _make_name(self, group):
        return NotImplemented

    def _make_meta(self, group):
        return {
            "data_prod_type": self.data_prod_type.name,
            "collator_group_flag": group.flag.name,
            "name": self._make_name(group),
        }

    def make_data_prod_groups(self, data_prods):
        groups = self._make_item_groups(data_prods)
        logger.debug(f"created {len(groups)} data prod groups.")
        return [
            {
                "meta": self._make_meta(g),
                "data_items": self._make_data_items(g),
                "assocs": self._make_assocs(g),
            }
            for g in groups
        ]


class CollateByPosition(CollatorBase):
    __abstract__ = True

    class Position(Flag):
        start = auto()
        end = auto()
        inner = auto()
        not_start = inner | end
        not_end = start | inner
        outer = start | end
        anywhere = inner | outer

    def _get_item_position(self, dp_index):
        return NotImplemented

    def _make_item_subgroups(self, dps):
        return NotImplemented

    def _make_item_groups(self, dps):
        T = DataProdType
        AT = DataProdAssocType
        P = self.Position
        GF = GroupFlag

        dps_main, get_subdps = self._make_item_subgroups(dps)
        pos_dp = []
        for dp in dps_main:
            p = self._get_item_position(dp)
            if p is None:
                continue
            pos_dp.append((p, dp))
        logger.debug(f"pos_items:\n{pd.DataFrame(pos_dp)}")
        if not pos_dp:
            return []

        groups = [Group()]
        for p, dp in pos_dp:
            if p == P.start:
                groups.append(Group(flag=GF.explicit_start, items=[dp]))
            elif p == P.end:
                groups[-1].append(dp, add_flag=GF.explicit_end)
                groups.append(Group())
            else:
                if groups[-1].flag & GF.explicit_end:
                    # group has end, new group not open yet
                    # skip
                    continue
                # current group open, add to it
                groups[-1].append(dp)
        # clean up empty and implicit groups
        for g in groups:
            if g.items and g.flag & GF.explicit:
                # add sub dps back to items
                items = []
                for dp in g.items:
                    items.append(dp)
                    items.extend(get_subdps(dp))
                items.sort(key=lambda d: d["meta"]["sort_key"])
                g.items = items
            else:
                continue
        return groups


_CollateKeysT = tuple[str, ...]
_CollateValuesAllowedT = tuple[None | tuple, ...]


class CollateByMetadata(CollatorBase):
    __abstract__ = True

    collate_by_meta_keys: ClassVar[_CollateKeysT]
    collate_by_meta_values_allowed: ClassVar[_CollateValuesAllowedT]

    def _filter_df_items(self, df_items):
        return df_items

    def _make_item_groups(self, dps):
        T = DataProdType
        AT = DataProdAssocType
        GF = GroupFlag

        collate_keys = self.collate_by_meta_keys
        collate_values_allowed = self.collate_by_meta_values_allowed
        records = []

        _missing_value = object()
        for dp in dps:
            record = {"data_prod": dp}
            for k in collate_keys:
                record[k] = dp["meta"].get(k, _missing_value)
            if _missing_value in record.values():
                continue
            records.append(record)
        if not records:
            return []
        df_items = self._filter_df_items(pd.DataFrame.from_records(records))
        if len(df_items) == 0:
            return []
        # filter by value
        mask_value = None
        for k, values_allowed in zip(collate_keys, collate_values_allowed, strict=True):
            if values_allowed is not None:
                m = df_items[k].isin(values_allowed)
                if mask_value is None:
                    mask_value = m
                else:
                    mask_value = mask_value & m
        if mask_value is not None:
            df_items = df_items[mask_value]
        logger.debug(f"df_items:\n{df_items}\ngrouby={collate_keys}")
        df_groups = df_items.groupby(list(collate_keys), as_index=True, sort=False)
        groups = []
        for gk, df in df_groups:
            g = Group(flag=GF.explicit)
            g.items = df["data_prod"].tolist()
            g.meta.update(dict(zip(collate_keys, gk, strict=True)))
            groups.append(g)
        return groups


def _get_toltec_data_kind_union(dp_index):
    data_items = dp_index["data_items"]

    def _get_toltec_data_kind(meta):
        dk = meta.get("data_kind", None)
        if dk is None:
            return None
        k, v = dk.rsplit(".", 1)
        if k != "ToltecDataKind":
            return None
        return ToltecDataKind[v]

    def _make_union(dks):
        if not dks:
            return None
        dk0 = dks[0]
        for dk in dks[1:]:
            dk0 = dk0 | dk
        return dk0

    return _make_union(
        list(
            filter(
                lambda d: d is not None,
                [_get_toltec_data_kind(d["meta"]) for d in data_items],
            ),
        ),
    )


def _make_data_prod_group_name(group, suffix, count_func=None):
    items = group.items
    obsnum0 = items[0]["meta"]["obsnum"]
    master0 = items[0]["meta"]["master"].lower()
    if count_func is not None:
        n_items = count_func(group)
    else:
        n_items = len(items)
    return f"{master0}-{obsnum0}-g{n_items}-{suffix}"


def _get_raw_obs_count(group):
    return sum(item["meta"]["data_prod_type"] == "dp_raw_obs" for item in group.items)


class CalGroupCollator(CollateByPosition):

    data_prod_type: ClassVar[DataProdType] = DataProdType.dp_cal_group
    data_prod_assoc_type: ClassVar[DataProdAssocType] = (
        DataProdAssocType.dpa_cal_group_obs
    )

    def _make_item_subgroups(self, dps):

        def _key(dp):
            return dp["meta"]["name"]

        def _type(dp):
            return dp["meta"]["data_prod_type"]

        dp_by_key = {_key(dp): dp for dp in dps}
        key_maps = {}
        dp_main = []
        for dp in dps:
            dp_type = _type(dp)
            if dp_type == "dp_raw_obs":
                dp_main.append(dp)
                key = _key(dp)
                key_reduced = f"{key}-reduced"
                mapped = []
                if key_reduced in dp_by_key:
                    mapped.append(key_reduced)
                key_maps[key] = mapped

        logger.debug(f"subgroup key maps:\n{pformat_yaml(key_maps)}")

        def _get_subdps(dp):
            return [dp_by_key[k] for k in key_maps[_key(dp)]]

        return dp_main, _get_subdps

    def _get_item_position(self, dp_index):
        if dp_index["meta"]["data_prod_type"] != "dp_raw_obs":
            return None
        P = self.Position
        data_kind_union = _get_toltec_data_kind_union(dp_index)
        if data_kind_union is None:
            return None
        if ToltecDataKind.RawSweep & data_kind_union:
            if ToltecDataKind.VnaSweep & data_kind_union:
                return P.start
            return P.not_start
        return None

    def _make_item_groups(self, dps):
        groups = super()._make_item_groups(dps)
        return [g for g in groups if _get_raw_obs_count(g) > 1]

    _make_name = staticmethod(
        functools.partial(
            _make_data_prod_group_name,
            suffix="cal",
            count_func=_get_raw_obs_count,
        ),
    )


class DriveFitCollator(CollateByMetadata):

    data_prod_type: ClassVar[DataProdType] = DataProdType.dp_drivefit
    data_prod_assoc_type: ClassVar[DataProdAssocType] = (
        DataProdAssocType.dpa_drivefit_obs
    )
    collate_by_meta_keys: ClassVar[_CollateKeysT] = (
        "obsnum",
        "master",
    )
    collate_by_meta_values_allowed: ClassVar[_CollateValuesAllowedT] = (
        None,
        None,
    )

    def _filter_df_items(self, df_items):
        def _f(dp_index):
            dk = _get_toltec_data_kind_union(dp_index)
            return bool(dk & ToltecDataKind.TargetSweep)

        return df_items[df_items["data_prod"].apply(_f)]

    def _make_item_groups(self, *args, **kwargs):
        groups = super()._make_item_groups(*args, **kwargs)
        return [g for g in groups if _get_raw_obs_count(g) > 1]

    _make_name = staticmethod(
        functools.partial(
            _make_data_prod_group_name,
            suffix="drivefit",
            count_func=_get_raw_obs_count,
        ),
    )


class FocusGroupCollator(CollateByMetadata):

    data_prod_type: ClassVar[DataProdType] = DataProdType.dp_focus_group
    data_prod_assoc_type: ClassVar[DataProdAssocType] = (
        DataProdAssocType.dpa_focus_group_obs
    )
    collate_by_meta_keys: ClassVar[_CollateKeysT] = ("obs_goal",)
    collate_by_meta_values_allowed: ClassVar[_CollateValuesAllowedT] = (("focus",),)

    def _filter_df_items(self, df_items):
        def _f(dp_index):
            return False

        return df_items[df_items["data_prod"].apply(_f)]

    def _make_item_groups(self, *args, **kwargs):
        groups = super()._make_item_groups(*args, **kwargs)
        return [g for g in groups if len(g.items) > 1]

    def _make_meta(self, group):
        meta = super()._make_meta(group)
        items = group.items
        obsnum0 = items[0]["meta"]["obsnum"]
        master0 = items[0]["meta"]["master"]
        n_items = len(items)
        meta["name"] = f"{master0}-{obsnum0}-g{n_items}-focus"
        return meta

    _make_name = staticmethod(
        functools.partial(
            _make_data_prod_group_name,
            suffix="focus",
        ),
    )


_sort_key_value_delta = 1e-3


@dataclass
class QLDataProdCollector:
    """A class to collect data prod from raw obs db."""

    db: SqlaDB
    data_lmt_rootpath: Path = Path("/data_lmt")
    data_prod_output_path: Path = Path("dataprod_toltec")
    data_prod_index_filename_prefix: str = "dp_toltec_"

    def __post_init__(self):
        self._dp_index_store = YamlFileIndex(
            self.data_prod_output_path,
            sort_key=lambda d: float(d["meta"]["sort_key"]),
        )
        self._dp_item_rootpath = Path(
            os.path.relpath(
                self.data_lmt_rootpath,
                self.data_prod_output_path,
            ),
        )
        self._last_collected = None
        self._query_cursor = None
        collators = []
        for collator_cls in _collator_cls_registry:
            collators.append(collator_cls())
        self._collators = collators

    def _conect_or_get_raw_obs_db(self):
        try:
            return ToltecRawObsDB(self.db)
        except sae.OperationalError as e:
            logger.error(f"failed to connect to {self.db}: {e}")
            return None

    # @cachetools.func.ttl_cache(ttl=1)
    def collect(
        self,
        n_items=10,
        n_updates=None,
        n_items_for_groups=100,
    ) -> DataProdCollectorInfo:
        """Collect data prod and return the list of index file paths."""
        id_per_item = 13
        with timeit("check new files in toltec db"):
            rodb = self._conect_or_get_raw_obs_db()
            if rodb is None:
                return DataProdCollectorInfo(
                    is_active=False,
                    message=(
                        f"Failed to connect to raw obs database. Last updated: "
                        f"{self._report_last_collected_time(reset=False)}",
                    ),
                )
            cursor = self._query_cursor
            if self._query_cursor is None:
                # do a initial query of n_items
                kw = {
                    "n_items": n_items,
                    "id": slice(-n_items * id_per_item, None),
                }
            else:
                # this will also query the previous n_updates items.
                if n_updates is None or n_updates > n_items:
                    n_updates = n_items
                kw = {
                    "n_items": n_updates,
                    "id": slice(cursor["id_max"] - n_updates * id_per_item, None),
                }
            r_grouped = rodb.id_query_grouped(valid_key="any", **kw)
            # update cursor
            self._query_cursor = r_grouped.iloc[0].to_dict()
            logger.debug(f"current query cursor: {self._query_cursor}")

            # now we get the data prod index from re-query by id
            id_min = r_grouped["id_min"].min()
            id_max = r_grouped["id_max"].max()
            dp_group_index = rodb.get_dp_index_for_id(id=slice(id_min, id_max + 1))
        # print(pformat_yaml(dp_group_index))
        # for each raw obs in the group, write the data products.
        with timeit("generate obs dp index"):
            for dp_index in dp_group_index["data_items"]:
                self._save_raw_obs_index_file(dp_index)
                self._save_basic_reduced_obs_index_file(dp_index)
        with timeit("generate obs group dp index"):
            for dpg_index in self.collect_groups(n_items_for_groups):
                self._save_dp_index(dpg_index)
        # generate a human readable status message
        return DataProdCollectorInfo(
            is_active=True,
            message=f"Last updated: {self._report_last_collected_time(reset=True)}",
        )

    def collect_groups(self, n_items):
        store = self.data_prod_index_store
        i = 0
        dps = collections.deque()
        iter_filenames = store.iter_filenames(reverse=True)
        while i < n_items:
            try:
                filename = next(iter_filenames)
            except StopIteration:
                break
            else:
                # print(filename)
                if re.search(r"-\d+-g\d+", filename):
                    # skip group
                    logger.debug(f"delete group file {filename} for regen")
                    del store[filename]
                    continue
                dps.appendleft(store[filename])
                i += 1
        dpgs = []
        for collator in self._collators:
            logger.debug(f"collate {len(dps)} data prods with {collator=}")
            dpgs.extend(collator.make_data_prod_groups(dps))
        logger.debug(f"collected {len(dpgs)} data prod groups.")
        # replace the data product index with the actual filenames in cache

        def _get_dp_filepath(dp):
            filename = self._make_dp_index_filename(dp)
            return store.get_filepath(filename)

        for dpg in dpgs:
            dpg["meta"]["sort_key"] = (
                max(dp["meta"]["sort_key"] for dp in dpg["data_items"])
                + _sort_key_value_delta
            )
            dpg["data_items"] = [
                {
                    "meta": dp["meta"],
                    "filepath": _get_dp_filepath(dp),
                }
                for dp in dpg["data_items"]
            ]
            for a in dpg["assocs"]:
                dp = a.pop("index")
                a["filepath"] = _get_dp_filepath(dp)
        return dpgs

    def _report_last_collected_time(self, reset):
        now = time.time()
        if reset:
            self._last_collected = now
            return "Just now"
        _last_collected = self._last_collected
        if _last_collected is None:
            return "Never"
        time_elapsed = now - _last_collected
        return f"{human_time(time_elapsed)} ago"

    @property
    def data_prod_index_store(self):
        """The index to access collected data prod index."""
        return self._dp_index_store

    def _make_dp_index_filename(self, dp_index):
        return f"{self.data_prod_index_filename_prefix}{dp_index['meta']['name']}.yaml"

    def _get_dp_index_filepath(self, dp_index):
        filename = self._make_dp_index_filename(dp_index)
        return self._dp_index_store.get_filepath(filename)

    def _get_dp_index_relpath(self, filepath):
        """Return the relative path to write in the generated dp index."""
        filepath = Path(filepath)

        def _get_subpath(path, parent_name):
            for p in filepath.parents:
                if p.name == parent_name:
                    return filepath.relative_to(p)
            return None

        subfilepath = _get_subpath(filepath, "data_lmt")
        if subfilepath is not None:
            # remake the root.
            return self._dp_item_rootpath.joinpath(subfilepath)
        dpdir = self.data_prod_output_path
        subfilepath = _get_subpath(filepath, dpdir.name)
        if subfilepath is not None:
            # already a valid path
            return filepath
        raise ValueError("not a valid data file path")

    def _update_data_item_filepaths(self, dp_index):
        for d in dp_index["data_items"]:
            # replace file paths
            d["filepath"] = self._get_dp_index_relpath(d["filepath"])
        return dp_index

    def _update_data_item_meta(self, dp_index):
        for d in dp_index["data_items"]:
            d["meta"].update(_guess_meta_from_source_cached(d["filepath"]))
        return dp_index

    def _save_dp_index(self, dp_index):
        filename = self._make_dp_index_filename(dp_index)
        self._update_data_item_meta(dp_index)
        self._update_data_item_filepaths(dp_index)
        # save
        store = self._dp_index_store
        store[filename] = dp_index
        return store.get_filepath(filename)

    def _save_raw_obs_index_file(
        self,
        raw_obs_dp_index,
    ):
        dp_index = raw_obs_dp_index
        dp_index["meta"] |= {
            "data_prod_type": "dp_raw_obs",
            "sort_key": dp_index["meta"]["id_min"],
        }
        dp_index["assocs"] = []
        return self._save_dp_index(dp_index)

    def _save_basic_reduced_obs_index_file(
        self,
        raw_obs_dp_index,
    ):
        reduced_path = self.data_lmt_rootpath / "toltec/reduced"

        def _get_basic_reduced_items(data_item):
            d = data_item
            m = d["meta"]
            # print(pformat_yaml(d))
            files = reduced_path.glob(
                f"{m['interface']}_{m['obsnum']:06d}_{m['subobsnum']:03d}_{m['scannum']:04d}_*",
            )
            data_items = [d]
            for f in files:
                data_items.append(  # noqa: PERF401
                    {
                        "filepath": f,
                        "meta": {},
                    },
                )
            return data_items

        data_items = []
        for d in raw_obs_dp_index["data_items"]:
            data_items.extend(_get_basic_reduced_items(d))
        assocs = [
            {
                "data_prod_assoc_type": "dpa_basic_reduced_obs_raw_obs",
                "filepath": self._make_dp_index_filename(raw_obs_dp_index),
            },
        ] + raw_obs_dp_index["assocs"]
        dp_index = {
            "meta": raw_obs_dp_index["meta"]
            | {
                "data_prod_type": "dp_basic_reduced_obs",
                "name": f"{raw_obs_dp_index['meta']['name']}-reduced",
                "sort_key": raw_obs_dp_index["meta"]["id_max"] + _sort_key_value_delta,
            },
            "data_items": data_items,
            "assocs": assocs,
        }
        return self._save_dp_index(dp_index)
