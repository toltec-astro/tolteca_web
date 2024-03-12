import os
from tollan.utils.log import logger
from pathlib import Path
from dataclasses import dataclass

from tollan.utils.yaml import yaml_dump, yaml_load
from tollan.utils.log import logit

from .raw_obs_db import ToltecRawObsDB
from tolteca_datamodels.toltec.file import guess_meta_from_source
from contextlib import nullcontext


from tollan.db import SqlaDB
from multiprocessing.synchronize import Lock as LockType
from typing import Protocol


__all__ = [
    "DataProdCollectorProtocol",
    "QLDataProdCollector",
]


class DataProdCollectorProtocol(Protocol):
    """A protocol class for data prod collectors."""

    def collect():
        """Collect data products."""

    def get_collected():
        """Return collected data products."""


@dataclass
class QLDataProdCollector:
    """A class to collect data prod from raw obs db."""

    db: SqlaDB
    data_lmt_rootpath: Path = Path("/data_lmt")
    data_prod_output_path: Path = Path("dataprod_toltec")
    data_prod_index_filename_prefix: str = "dp_toltec_"
    data_prod_output_lock: None | LockType = None

    def __post_init__(self):
        self._raw_obs_db = ToltecRawObsDB(self.db)
        self._dp_item_rootpath = Path(
            os.path.relpath(
                self.data_lmt_rootpath,
                self.data_prod_output_path,
            ),
        )

    # @cachetools.func.ttl_cache(maxsize=256, ttl=1)
    def collect(self, n_items=10):
        """Collect data prod and return the list of index file paths."""
        rodb = self._raw_obs_db
        r_grouped = rodb.obs_query_grouped(obsnum=slice(-n_items, None))
        id_min = r_grouped["id_min"].min()
        id_max = r_grouped["id_min"].max()
        dp_group_index = rodb.get_dp_index_for_id(id=slice(id_min, id_max + 1))
        # print(pformat_yaml(dp_group_index))
        for dp_index in dp_group_index["data_items"]:
            self._save_raw_obs_index_file(dp_index)
            self._save_basic_reduced_obs_index_file(dp_index)
        return self.data_prod_output_path

    def get_collected(self):
        """Return list of data prod index files collected."""
        glob_pattern = f"{self.data_prod_index_filename_prefix}*.yaml"
        return list(self.data_prod_output_path.glob(glob_pattern))

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

    def _get_dp_index_filename(self, dp_index):
        return f"{self.data_prod_index_filename_prefix}{dp_index['meta']['name']}.yaml"

    def _get_dp_index_filepath(self, dp_index):
        return self.data_prod_output_path.joinpath(
            self._get_dp_index_filename(dp_index),
        )

    def _update_filepaths(self, dp_index):
        for d in dp_index["data_items"]:
            # replace file paths
            d["filepath"] = self._get_dp_index_relpath(d["filepath"])
        return dp_index

    def _save_dp_index(self, dp_index):
        dp_index_filepath = self._get_dp_index_filepath(dp_index)
        if dp_index_filepath.exists():
            dp = yaml_load(dp_index_filepath)
            if isinstance(dp, dict) and set(dp.keys()).intersection(
                ["data_items", "meta"],
            ):
                return dp_index_filepath
        # collect meta
        for d in dp_index["data_items"]:
            d["meta"].update(guess_meta_from_source(d["filepath"]))
        dp_index = self._update_filepaths(dp_index)
        with (
            self.data_prod_output_lock or nullcontext(),
            logit(logger.debug, f"generate dp index file {dp_index_filepath}"),
            dp_index_filepath.open("w") as fo,
        ):
            yaml_dump(dp_index, fo)
        return dp_index_filepath

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
                "filepath": self._get_dp_index_filename(raw_obs_dp_index),
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
