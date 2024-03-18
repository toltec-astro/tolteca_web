import os
from pathlib import Path

import diskcache
from tollan.utils.yaml import yaml_dump, yaml_loads

# from tollan.utils.log import logger
# from tollan.utils.fmt import pformat_yaml


class _YamlFileDisk(diskcache.Disk):
    """A cache disk to store value in yaml format."""

    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)

    def store(self, value, read, key=diskcache.UNKNOWN):
        """Store value in cache."""
        if not read:
            # logger.debug(f"dump yaml:\n{pformat_yaml(value)}")
            value = yaml_dump(value)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        """Fetch value from cache."""
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = yaml_loads(data)
        return data

    def filename(self, key=diskcache.UNKNOWN, value=diskcache.UNKNOWN):
        """Return the filename."""
        if key is not diskcache.UNKNOWN:
            filename = str(key)
            return filename, os.path.join(self._directory, filename)  # noqa: PTH118
        return super().filename(key=key, value=value)

    def _write(self, full_path, iterator, _mode, encoding=None):
        return super()._write(
            full_path,
            iterator,
            _mode.replace("x", "w"),
            encoding=encoding,
        )

    def remove(self, _file_name):
        # do not remove file.
        return


class YamlFileIndex(diskcache.Index):
    """A cache index to store yaml files."""

    def __init__(self, directory, sort_key=None):
        cache = diskcache.Cache(
            directory=directory,
            disk=_YamlFileDisk,
            eviction_policy="none",
            disk_min_file_size=0,
        )
        self._cache = cache
        # this is to store the sort key for iter
        self._cache_sort_key = diskcache.Cache(
            directory=Path(directory).joinpath("cache_sort_key"),
            eviction_policy="none",
        )
        self._sort_key = sort_key

    def __setitem__(self, key, value):
        # update sort_key index
        if self._sort_key is not None:
            sort_key = self._sort_key(value)
        else:
            sort_key = key
        v = self._cache_sort_key.get(sort_key, set())
        v.add(key)
        self._cache_sort_key[sort_key] = v
        super().__setitem__(key, value)

    def iter_filenames(self, reverse=False):
        sort_keys = self._cache_sort_key.iterkeys(reverse=reverse)
        for sort_key in sort_keys:
            # logger.debug(f"iter sort_key={sort_key}")
            filenames = self._cache_sort_key[sort_key]
            for filename in filenames:
                if filename in self:
                    yield filename

    def get_filepath(self, key):
        """Return the filepath for key."""
        return self.cache.disk.filename(key)[1]

    def get_filepaths(self):
        """Return the filepaths."""
        return [self.get_filepath(key) for key in self]
