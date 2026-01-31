"""File resolver with remote caching support using HTTP.

This module provides transparent file access with automatic caching for remote files.
When configured with a remote base URL, files that don't exist locally are fetched
via HTTP and cached locally for subsequent access.

Configuration (environment variables):
--------------------------------------
TOLTECA_WEB_DATA_LMT_ROOTPATH : str
    Local data_lmt root path (required)
TOLTECA_WEB_DATA_LMT_REMOTE_BASE_URL : str, optional
    HTTP base URL for remote files (e.g., "http://localhost:60080/data_lmt")
    If not set, remote caching is disabled.
TOLTECA_WEB_FILE_CACHE_DIR : str, optional
    Local directory for cached remote files.
    Defaults to TOLTECA_WEB_DATA_LMT_ROOTPATH if not set.

Remote Server Setup:
--------------------
The remote server should have a web server (nginx, apache, etc.) serving the
data_lmt directory. Port forward the HTTP port via SSH:

    ssh -N -L 60080:localhost:80 taco-mx

Then configure:
    TOLTECA_WEB_DATA_LMT_REMOTE_BASE_URL=http://localhost:60080/data_lmt

Usage:
------
    from tolteca_web.data_prod.file_resolver import get_file_resolver, resolve_file

    # Get the singleton resolver
    resolver = get_file_resolver()

    # Resolve a filepath - returns local path (cached if needed)
    local_path = resolver.resolve(filepath)

    # Or use the convenience function
    local_path = resolve_file(filepath)
"""

from __future__ import annotations

import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tollan.utils.log import logger

if TYPE_CHECKING:
    import diskcache


class CacheProgressStore:
    """Thread/process-safe store for download progress using diskcache."""

    def __init__(self, cache_dir: Path):
        import diskcache
        self._cache_dir = cache_dir / ".progress_cache"
        self._cache: diskcache.Cache | None = None

    def _get_cache(self) -> diskcache.Cache:
        """Lazy-init cache to avoid issues with multiprocessing."""
        if self._cache is None:
            import diskcache
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(self._cache_dir))
        return self._cache

    def set_download(
        self,
        file_id: str,
        total_bytes: int,
        current_bytes: int,
        status: str,
        filename: str | None = None,
    ) -> None:
        """Set download progress (atomic write).

        Parameters
        ----------
        file_id : str
            Unique identifier for the download (relative path)
        total_bytes : int
            Total file size in bytes
        current_bytes : int
            Current downloaded bytes
        status : str
            Download status: 'downloading', 'complete', 'error'
        filename : str, optional
            Display filename
        """
        cache = self._get_cache()
        expire = 300 if status == "downloading" else 10
        cache.set(
            f"download:{file_id}",
            {
                "total_bytes": total_bytes,
                "current_bytes": current_bytes,
                "status": status,
                "filename": filename or file_id,
                "updated_at": time.time(),
                "start_time": self._get_start_time(file_id),
            },
            expire=expire,
        )

    def _get_start_time(self, file_id: str) -> float:
        """Get start time for a download, preserving original if exists."""
        cache = self._get_cache()
        existing = cache.get(f"download:{file_id}")
        if existing and "start_time" in existing:
            return existing["start_time"]
        return time.time()

    def get_active_downloads(self) -> dict[str, dict]:
        """Get all active downloads (atomic read).

        Returns
        -------
        dict[str, dict]
            Mapping of file_id to progress info
        """
        cache = self._get_cache()
        result = {}
        for key in list(cache):
            if isinstance(key, str) and key.startswith("download:"):
                value = cache.get(key)
                if value and value.get("status") == "downloading":
                    result[key[9:]] = value
        return result

    def get_cache_stats(self) -> dict:
        """Get overall cache statistics.

        Returns
        -------
        dict
            Cache statistics including file count and total size
        """
        cache = self._get_cache()
        # Count completed downloads and active
        active = 0
        completed = 0
        for key in list(cache):
            if isinstance(key, str) and key.startswith("download:"):
                value = cache.get(key)
                if value:
                    if value.get("status") == "downloading":
                        active += 1
                    elif value.get("status") == "complete":
                        completed += 1
        return {
            "active_downloads": active,
            "recent_completed": completed,
        }

    def complete_download(self, file_id: str, total_bytes: int) -> None:
        """Mark download as complete."""
        self.set_download(file_id, total_bytes, total_bytes, "complete")

    def error_download(self, file_id: str, error_msg: str) -> None:
        """Mark download as errored."""
        cache = self._get_cache()
        cache.set(
            f"download:{file_id}",
            {
                "status": "error",
                "error": error_msg,
                "updated_at": time.time(),
            },
            expire=30,
        )

    def clear(self) -> None:
        """Clear all progress entries."""
        cache = self._get_cache()
        for key in list(cache):
            if isinstance(key, str) and key.startswith("download:"):
                cache.delete(key)


@dataclass
class FileResolverConfig:
    """Configuration for file resolver.

    Attributes
    ----------
    local_data_root : Path
        Local data_lmt root path
    remote_base_url : str | None
        HTTP base URL for remote files (e.g., "http://localhost:60080/data_lmt")
    cache_dir : Path | None
        Local cache directory for remote files
    """

    local_data_root: Path
    remote_base_url: str | None = None
    cache_dir: Path | None = None

    @classmethod
    def from_env(cls) -> FileResolverConfig:
        """Create configuration from environment variables."""
        local_root_str = os.environ.get("TOLTECA_WEB_DATA_LMT_ROOTPATH", "/data_lmt")
        local_data_root = Path(local_root_str).expanduser().absolute()

        remote_base_url = os.environ.get("TOLTECA_WEB_DATA_LMT_REMOTE_BASE_URL")
        if remote_base_url:
            remote_base_url = remote_base_url.rstrip("/")

        cache_dir_str = os.environ.get("TOLTECA_WEB_FILE_CACHE_DIR")
        cache_dir = Path(cache_dir_str).expanduser().absolute() if cache_dir_str else None

        return cls(
            local_data_root=local_data_root,
            remote_base_url=remote_base_url,
            cache_dir=cache_dir,
        )

    @property
    def remote_enabled(self) -> bool:
        """Check if remote file access is enabled."""
        return self.remote_base_url is not None

    @property
    def effective_cache_dir(self) -> Path:
        """Get effective cache directory (cache_dir or local_data_root)."""
        return self.cache_dir or self.local_data_root


@dataclass
class FileResolver:
    """File resolver with remote caching support.

    Resolves file paths, fetching from remote via HTTP and caching locally when needed.
    """

    config: FileResolverConfig
    _progress_store: CacheProgressStore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the resolver."""
        if self.config.remote_enabled:
            self._progress_store = CacheProgressStore(self.config.effective_cache_dir)
            logger.info(
                f"File resolver initialized with remote caching: "
                f"{self.config.remote_base_url} -> {self.config.effective_cache_dir}"
            )
        else:
            logger.info(
                f"File resolver initialized (local only): {self.config.local_data_root}"
            )

    @property
    def progress_store(self) -> CacheProgressStore | None:
        """Get the progress store for monitoring downloads."""
        return self._progress_store

    def _compute_relative_path(self, filepath: str | Path) -> str | None:
        """Compute relative path from local data root.

        Returns None if the filepath is not under the local data root.
        """
        filepath = Path(filepath).absolute()

        try:
            return str(filepath.relative_to(self.config.local_data_root))
        except ValueError:
            return None

    def _compute_remote_url(self, relative_path: str) -> str:
        """Compute full HTTP URL from relative path."""
        if not self.config.remote_base_url:
            raise RuntimeError("Remote base URL not configured")
        from urllib.parse import quote
        encoded_path = quote(relative_path, safe="/")
        return f"{self.config.remote_base_url}/{encoded_path}"

    def _compute_cache_path(self, relative_path: str) -> Path:
        """Compute local cache path from relative path."""
        return self.config.effective_cache_dir / relative_path

    def resolve(self, filepath: str | Path) -> Path | None:
        """Resolve a filepath, fetching from remote if needed.

        Parameters
        ----------
        filepath : str | Path
            The filepath to resolve (expected to be an absolute path
            under the local data root)

        Returns
        -------
        Path | None
            The resolved local path (original or cached), or None if
            the file cannot be resolved (doesn't exist locally and
            remote fetch failed)
        """
        filepath = Path(filepath)

        if filepath.exists():
            return filepath

        if not self.config.remote_enabled:
            logger.debug(f"File not found (remote disabled): {filepath}")
            return None

        relative_path = self._compute_relative_path(filepath)
        if relative_path is None:
            logger.warning(
                f"File not under local data root, cannot resolve remotely: {filepath}"
            )
            return None

        cache_path = self._compute_cache_path(relative_path)

        if cache_path.exists():
            logger.debug(f"Using cached file: {cache_path}")
            return cache_path

        remote_url = self._compute_remote_url(relative_path)
        return self._fetch_and_cache(remote_url, cache_path, relative_path)

    def _fetch_and_cache(self, remote_url: str, cache_path: Path, relative_path: str) -> Path | None:
        """Fetch a file from remote via HTTP and cache locally.

        Parameters
        ----------
        remote_url : str
            Full HTTP URL to fetch
        cache_path : Path
            Local path to cache the file
        relative_path : str
            Relative path for progress tracking

        Returns
        -------
        Path | None
            The cache path if successful, None otherwise
        """
        try:
            logger.info(f"Fetching remote file: {remote_url} -> {cache_path}")

            cache_path.parent.mkdir(parents=True, exist_ok=True)

            request = urllib.request.Request(remote_url)
            with urllib.request.urlopen(request, timeout=300) as response:
                total_bytes = int(response.headers.get("Content-Length", 0))
                current_bytes = 0
                filename = cache_path.name

                if self._progress_store and total_bytes > 0:
                    self._progress_store.set_download(
                        relative_path, total_bytes, 0, "downloading", filename
                    )

                with open(cache_path, "wb") as f:
                    while chunk := response.read(8192):
                        f.write(chunk)
                        current_bytes += len(chunk)

                        if self._progress_store and total_bytes > 0:
                            self._progress_store.set_download(
                                relative_path, total_bytes, current_bytes, "downloading", filename
                            )

            if cache_path.exists():
                final_size = cache_path.stat().st_size
                logger.info(f"Cached: {cache_path} ({final_size} bytes)")

                if self._progress_store:
                    self._progress_store.complete_download(relative_path, final_size)

                return cache_path

            logger.error(f"Cache file not created: {cache_path}")
            if self._progress_store:
                self._progress_store.error_download(relative_path, "File not created")
            return None

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"Remote file not found: {remote_url}")
            else:
                logger.warning(f"HTTP error fetching {remote_url}: {e.code} {e.reason}")
            if self._progress_store:
                self._progress_store.error_download(relative_path, f"HTTP {e.code}")
            return None
        except urllib.error.URLError as e:
            logger.warning(f"URL error fetching {remote_url}: {e.reason}")
            if self._progress_store:
                self._progress_store.error_download(relative_path, str(e.reason))
            return None
        except Exception as e:
            logger.exception(f"Failed to fetch remote file: {remote_url}: {e}")
            if self._progress_store:
                self._progress_store.error_download(relative_path, str(e))
            return None

    @property
    def progress_store(self) -> CacheProgressStore | None:
        """Access the progress store for monitoring downloads."""
        return self._progress_store

    def get_download_status(self) -> dict:
        """Get current download status for UI monitoring.

        Returns
        -------
        dict
            Dictionary with 'active_downloads', 'cache_stats', and 'remote_enabled' keys
        """
        if not self._progress_store:
            return {"active_downloads": {}, "cache_stats": {}, "remote_enabled": False}
        return {
            "active_downloads": self._progress_store.get_active_downloads(),
            "cache_stats": self._progress_store.get_cache_stats(),
            "remote_enabled": True,
        }

    def prefetch(self, filepaths: list[str | Path]) -> dict[str, Path | None]:
        """Prefetch multiple files from remote.

        Parameters
        ----------
        filepaths : list[str | Path]
            List of filepaths to prefetch

        Returns
        -------
        dict[str, Path | None]
            Mapping of original filepath to resolved path
        """
        results = {}
        for fp in filepaths:
            results[str(fp)] = self.resolve(fp)
        return results

    def is_resolvable(self, filepath: str | Path) -> bool:
        """Check if a file is potentially resolvable without fetching.

        Returns True if the file exists locally, in cache, or if remote
        access is enabled (meaning it could potentially be fetched).
        This is useful for filtering file lists without actually fetching.

        Parameters
        ----------
        filepath : str | Path
            The filepath to check

        Returns
        -------
        bool
            True if file is resolvable, False otherwise
        """
        filepath = Path(filepath)

        if filepath.exists():
            return True

        if not self.config.remote_enabled:
            return False

        relative_path = self._compute_relative_path(filepath)
        if relative_path is None:
            return False

        cache_path = self._compute_cache_path(relative_path)
        if cache_path.exists():
            return True

        return True

    def list_remote(self, prefix: str = "") -> list[str]:
        """List files in remote storage.

        Note: HTTP directory listing requires server support (autoindex).
        This method is not guaranteed to work with all HTTP servers.

        Parameters
        ----------
        prefix : str
            Path prefix to filter results

        Returns
        -------
        list[str]
            List of file paths in remote storage (empty if not supported)
        """
        if not self.config.remote_enabled:
            return []

        logger.warning(
            "list_remote() is not fully supported for HTTP. "
            "Use database queries to discover available files."
        )
        return []


_resolver: FileResolver | None = None


def get_file_resolver() -> FileResolver:
    """Get the singleton file resolver instance."""
    global _resolver
    if _resolver is None:
        config = FileResolverConfig.from_env()
        _resolver = FileResolver(config=config)
    return _resolver


def resolve_file(filepath: str | Path) -> Path | None:
    """Resolve a filepath, fetching from remote if needed.

    Convenience function that uses the singleton file resolver.

    Parameters
    ----------
    filepath : str | Path
        The filepath to resolve

    Returns
    -------
    Path | None
        The resolved local path, or None if not resolvable
    """
    return get_file_resolver().resolve(filepath)


def is_file_resolvable(filepath: str | Path) -> bool:
    """Check if a file is potentially resolvable without fetching.

    Convenience function that uses the singleton file resolver.

    Parameters
    ----------
    filepath : str | Path
        The filepath to check

    Returns
    -------
    bool
        True if file is resolvable, False otherwise
    """
    return get_file_resolver().is_resolvable(filepath)


def reset_resolver() -> None:
    """Reset the singleton file resolver (for testing)."""
    global _resolver
    _resolver = None
