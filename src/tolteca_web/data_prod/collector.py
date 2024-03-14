import functools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import sqlalchemy.exc as sae
from astropy.utils.console import human_time
from tollan.db import SqlaDB
from tollan.utils.log import logger, timeit
from tolteca_datamodels.toltec.file import guess_meta_from_source

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
    return guess_meta_from_source(source)


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
        )
        self._dp_item_rootpath = Path(
            os.path.relpath(
                self.data_lmt_rootpath,
                self.data_prod_output_path,
            ),
        )
        self._last_collected = None
        self._query_cursor = None

    def _conect_or_get_raw_obs_db(self):
        try:
            return ToltecRawObsDB(self.db)
        except sae.OperationalError as e:
            logger.error(f"failed to connect to {self.db}: {e}")
            return None

    # @cachetools.func.ttl_cache(ttl=1)
    def collect(self, n_items=10, n_updates=None) -> DataProdCollectorInfo:
        """Collect data prod and return the list of index file paths."""
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
                    "obsnum": slice(-n_items, None),
                }
            else:
                # this will also query the previous n_updates items.
                if n_updates is None or n_updates > n_items:
                    n_updates = n_items
                kw = {
                    "n_items": n_updates,
                    "obsnum": slice(cursor["obsnum"] - n_updates, None),
                }
            r_grouped = rodb.obs_query_grouped(valid_key="any", **kw)
            # update cursor
            self._query_cursor = r_grouped.iloc[0].to_dict()
            logger.debug(f"current query cursor: {self._query_cursor}")

            # now we get the data prod index from re-query by id
            id_min = r_grouped["id_min"].min()
            id_max = r_grouped["id_max"].max()
            dp_group_index = rodb.get_dp_index_for_id(id=slice(id_min, id_max + 1))
        # print(pformat_yaml(dp_group_index))
        # for each raw obs in the group, write the data products.
        with timeit("generate dp index"):
            for dp_index in dp_group_index["data_items"]:
                self._save_raw_obs_index_file(dp_index)
                self._save_basic_reduced_obs_index_file(dp_index)
            # generate a human readable status message
            return DataProdCollectorInfo(
                is_active=True,
                message=f"Last updated: {self._report_last_collected_time(reset=True)}",
            )

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
        for p in filepath.parents:
            if p.name == "data_lmt":
                subfilepath = filepath.relative_to(p)
                break
        else:
            raise ValueError("not a valid data file path")
        return self._dp_item_rootpath.joinpath(subfilepath)

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
            "meta": {
                "data_prod_type": "dp_basic_reduced_obs",
                "name": f"{raw_obs_dp_index['meta']['name']}-reduced",
            },
            "data_items": data_items,
            "assocs": assocs,
        }
        return self._save_dp_index(dp_index)
