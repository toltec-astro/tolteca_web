import os

import diskcache
from tollan.utils.yaml import yaml_dump, yaml_loads


class _YamlFileDisk(diskcache.Disk):
    """A cache disk to store value in yaml format."""

    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)

    def store(self, value, read, key=diskcache.UNKNOWN):
        """Store value in cache."""
        if not read:
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

    def __init__(self, directory):
        cache = diskcache.Cache(
            directory=directory,
            disk=_YamlFileDisk,
            eviction_policy="none",
            disk_min_file_size=0,
        )
        self._cache = cache

    def get_filepath(self, key):
        """Return the filepath for key."""
        return self.cache.disk.filename(key)[1]

    def get_filepaths(self):
        """Return the filepaths."""
        return [self.get_filepath(key) for key in self]
