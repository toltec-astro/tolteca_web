"""ToltecaDB-based data product collector.

Production Architecture Overview:
==================================

Service 1: Dagster (Continuous Ingestion)
------------------------------------------
1. quartet_sensor polls toltec_db every 10 seconds
2. Detects complete quartets (timeout-based validation)
3. Triggers process_quartet asset per quartet
4. Ingests quartet from toltecdb → tolteca_db:
   - Creates ONE DataProd per observation (quartet)
   - Creates MULTIPLE DataProdSource (one per interface file)
   - Generates associations (CalGroup, DriveFit, FocusGroup)
5. Writes to SQLite database (multi-process concurrent writes)

Service 2: Webapp with Polling (THIS MODULE)
---------------------------------------------
1. LiveUpdateSection timer triggers every 5-15 seconds (user configurable)
2. ToltecaDBCollector.collect() queries tolteca_db (read-only)
3. Gets structured data from DataProd + DataProdAssoc tables
4. No YAML file I/O needed
5. Displays results with automatic refresh
6. User clicks → ToltecaDBAdapter.get_associations() returns groups

Database Tables:
----------------
- DataProd: Observations (ONE per quartet)
- DataProdSource: Interface files (MULTIPLE per DataProd)
- DataProdAssoc: Associations/groups created by Dagster
  - CalGroup (calibration sequences)
  - DriveFit (drive characterization) 
  - FocusGroup (focus measurements)
- Location: Storage sites (LMT, UMass, etc.)

This collector replaces the legacy QLDataProdCollector by querying
tolteca_db's DataProd table instead of toltecdb directly, and eliminates
YAML file caching since associations are already in the database.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from tollan.utils.log import logger
from tolteca_db.constants import ToltecDataKind

from .collector import DataProdCollectorInfo
from .tolteca_db_adapter import get_tolteca_db_adapter


def _convert_data_kind_to_legacy(data_kind):
    """Convert data_kind integer to legacy string format.
    
    Parameters
    ----------
    data_kind : int or str
        Data kind value from database (bitwise flag)
        
    Returns
    -------
    str
        Legacy format like "ToltecDataKind.VnaSweep"
    """
    if isinstance(data_kind, str):
        return data_kind
    
    # Convert integer to ToltecDataKind flag and get name
    try:
        kind_flag = ToltecDataKind(data_kind)
        return f"ToltecDataKind.{kind_flag.name}"
    except ValueError:
        return f"ToltecDataKind.Unknown({data_kind})"


class ToltecaDBIndexStore:
    """Minimal store interface for ToltecaDBDataProdCollector compatibility.
    
    This store can load data products on-demand from the database when they
    are not in the cache. This is essential for loading associated data products
    that may not have been included in the initial collection.
    """
    
    def __init__(self, collector=None):
        """Initialize the store.
        
        Parameters
        ----------
        collector : ToltecaDBDataProdCollector, optional
            Collector reference for on-demand loading
        """
        self._items = []  # List of UIDs
        self._index_cache = {}  # uid -> index dict mapping
        self._collector = collector
    
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        return iter(self._items)
    
    def iter_filenames(self, reverse=False):
        """Iterate over data product UIDs (acting as filenames)."""
        items = reversed(self._items) if reverse else self._items
        return iter(items)
    
    def get_filepath(self, uid_or_uri):
        """Return filepath for a UID, tolteca_db:// URI, or source file path.
        
        Parameters
        ----------
        uid_or_uri : str
            Can be:
            - Plain UID (e.g., "1") → returns "tolteca_db://1"
            - tolteca_db:// URI (e.g., "tolteca_db://1") → returns as-is
            - Source file path (e.g., "toltec/tcs/toltec0/file.nc") → returns as-is
            
        Returns
        -------
        str
            Filepath or tolteca_db:// URI
        """
        # If already a tolteca_db:// URI, return as-is
        if uid_or_uri.startswith("tolteca_db://"):
            return uid_or_uri
        
        # If it looks like a file path (contains "/" or file extension), return as-is
        if "/" in uid_or_uri or uid_or_uri.endswith((".nc", ".fits", ".ecsv", ".parquet")):
            return uid_or_uri
        
        # Otherwise, treat as UID and add prefix
        return f"tolteca_db://{uid_or_uri}"
    
    def __getitem__(self, uid_or_uri):
        """Get data product by UID or URI (returns cached index dict).
        
        If the data product is not in cache and a collector is available,
        attempts to load it from the database.
        
        Parameters
        ----------
        uid_or_uri : str
            Either a plain UID (e.g., "1") or a full URI (e.g., "tolteca_db://1")
            
        Returns
        -------
        dict
            Cached index data for the data product
        """
        # Extract UID if given a URI
        if uid_or_uri.startswith("tolteca_db://"):
            uid = uid_or_uri.replace("tolteca_db://", "")
        else:
            uid = uid_or_uri
        
        # Check cache first
        if uid in self._index_cache:
            return self._index_cache[uid]
        
        # If not in cache and we have a collector, try to load from database
        if self._collector is not None:
            try:
                index_dict = self._collector.load_data_product_by_uid(uid)
                if index_dict:
                    # Cache it for future use
                    self._index_cache[uid] = index_dict
                    logger.debug(f"Loaded data product {uid} from database on-demand")
                    return index_dict
            except Exception as e:
                logger.warning(f"Failed to load data product {uid} from database: {e}")
        
        # Fallback: return minimal dict
        return {"uid": uid}
    
    def update(self, index_dicts):
        """Update the store with new data products.
        
        Parameters
        ----------
        index_dicts : list[dict]
            List of index dictionaries with 'uid' key
        """
        self._items = [d["uid"] for d in index_dicts]
        self._index_cache = {d["uid"]: d for d in index_dicts}


@dataclass
class ToltecaDBDataProdCollector:
    """Data product collector using tolteca_db backend.

    This collector queries the tolteca_db database for data products
    and provides them to the tolteca_web viewer interfaces. It queries
    the DataProd and DataProdAssoc tables directly instead of using
    YAML index files.
    """

    db_url: str
    """Database URL for tolteca_db."""

    _adapter: any = field(default=None, repr=False, init=False)
    _cached_data_prods: list[dict] = field(default_factory=list, repr=False, init=False)
    _index_store: ToltecaDBIndexStore = field(default_factory=ToltecaDBIndexStore, repr=False, init=False)

    @property
    def data_prod_index_store(self):
        """Return the index store for protocol compatibility."""
        return self._index_store

    def __post_init__(self):
        """Initialize collector."""
        # Initialize tolteca_db adapter
        self._adapter = get_tolteca_db_adapter(self.db_url)

        if self._adapter is None:
            logger.error("Failed to initialize tolteca_db adapter")
            return

        # Pass self to store for on-demand loading
        self._index_store._collector = self

        logger.info(f"Initialized ToltecaDBDataProdCollector with {self.db_url}")

    def collect(
        self,
        n_items: int | None = None,
        n_updates: int | None = None,
        data_prod_type: str | None = None,
        master: str | None = None,
        min_obsnum: int | None = None,
        max_obsnum: int | None = None,
        obs_date: str | None = None,
    ) -> DataProdCollectorInfo:
        """Collect data products from tolteca_db.

        Queries the DataProd table directly. Note that grouping and associations
        (CalGroup, DriveFit, FocusGroup) are handled by tolteca_db's
        AssociationGenerator, not by this collector. The associations are
        already stored in the DataProdAssoc table.

        Parameters
        ----------
        n_items : int
            Number of recent items to collect
        n_updates : int
            Number of recent items to refresh/update (not used in tolteca_db)
        data_prod_type : str, optional
            Filter by data product type label (e.g., 'dp_raw_obs', 'dp_basic_reduced_obs')
        master : str, optional
            Master to filter by ('ics' or 'tcs')
        min_obsnum : int, optional
            Minimum observation number to include
        max_obsnum : int, optional
            Maximum observation number to include
        obs_date : str, optional
            Observation date to filter by (YYYY-MM-DD format)

        Returns
        -------
        DataProdCollectorInfo
            Collection status information
        """
        if self._adapter is None:
            return DataProdCollectorInfo(
                is_active=False,
                message="tolteca_db adapter not initialized",
            )

        try:
            # Query recent data products from tolteca_db with filters
            # This queries the DataProd table directly
            # Groups and associations are already in DataProdAssoc table
            data_prods = self._adapter.query_data_products(
                data_prod_type=data_prod_type,
                master=master,
                min_obsnum=min_obsnum,
                max_obsnum=max_obsnum,
                obs_date=obs_date,
                limit=n_items,
            )

            logger.debug(f"Collected {len(data_prods)} data products from tolteca_db")

            # Cache the results for viewer access
            self._cached_data_prods = data_prods
            
            # Build index dicts compatible with DataProd viewer class
            index_dicts = []
            for dp in data_prods:
                # Get sources for this data product
                sources = self._adapter.query_sources_for_data_product(dp["uid"])
                
                # Convert meta to dict if it's not already
                meta = dp.get("meta", {})
                if hasattr(meta, "__dict__"):
                    # It's a dataclass, convert to dict
                    meta = {k: v for k, v in meta.__dict__.items()}
                elif not isinstance(meta, dict):
                    meta = {}
                
                # Get data_kind BEFORE converting enums
                data_kind_raw = meta.get("data_kind", 0)
                
                # Convert enum values to strings
                for key, value in list(meta.items()):
                    if hasattr(value, "value"):
                        meta[key] = value.value
                    elif hasattr(value, "name"):
                        meta[key] = value.name
                
                # Convert data_kind to legacy string format
                data_kind_str = _convert_data_kind_to_legacy(data_kind_raw)
                
                # Build data_items from sources (matching legacy structure)
                data_items = []
                for source in sources:
                    source_meta = source.get("meta", {})
                    if hasattr(source_meta, "__dict__"):
                        source_meta = {k: v for k, v in source_meta.__dict__.items()}
                    
                    roach = source_meta.get("roach")
                    interface = f"toltec{roach}" if roach is not None else None
                    
                    # Build item meta with only legacy-compatible fields
                    # Exclude: nw_id and other non-legacy fields
                    item_meta = {
                        "data_kind": data_kind_str,
                        "roach": roach,
                        "interface": interface,
                        "master": meta.get("master", "").upper(),
                        "obsnum": meta.get("obsnum"),
                        "subobsnum": meta.get("subobsnum"),
                        "scannum": meta.get("scannum"),
                        "name": f"{interface}-{meta.get('master', '').lower()}-{meta.get('obsnum')}-{meta.get('subobsnum')}-{meta.get('scannum')}" if interface else None,
                        "source": source.get("source_uri", ""),
                        # Add source_meta fields but exclude nw_id
                        **{k: v for k, v in source_meta.items() if k != "nw_id"},
                    }
                    # Ensure data_kind stays as string (override any source_meta value if present)
                    # For tel files, source_meta will have data_kind=16 which gets converted
                    if "data_kind" in source_meta:
                        item_meta["data_kind"] = _convert_data_kind_to_legacy(source_meta["data_kind"])
                    else:
                        item_meta["data_kind"] = data_kind_str
                    
                    data_items.append({
                        "filepath": source.get("filepath", source.get("source_uri", "")),  # Use absolute filepath
                        "meta": item_meta,
                    })
                
                # Build index dict with legacy structure
                # Ensure data_prod_type is a string, not enum
                data_prod_type = dp.get("data_prod_type", "unknown")
                if hasattr(data_prod_type, "value"):
                    data_prod_type = data_prod_type.value
                elif not isinstance(data_prod_type, str):
                    data_prod_type = str(data_prod_type)
                
                # Populate associations from database
                assocs = []
                db_assocs = self._adapter.get_associations(dp["uid"])
                for db_assoc in db_assocs:
                    # Convert database association to legacy format
                    # Format expected by viewer: {"data_prod_assoc_type": "dpa_cal_group_obs", "filepath": "..."}
                    assocs.append({
                        "data_prod_assoc_type": db_assoc.get("assoc_type", "unknown"),
                        "filepath": f"tolteca_db://{db_assoc['dst_uid']}"  # Pseudo-filepath using UID
                    })
                
                index_dict = {
                    "assocs": assocs,  # Populated from database
                    "data_items": data_items,
                    "meta": {
                        "data_prod_type": data_prod_type,
                        "name": meta.get("name", f"dp_{dp['uid']}"),
                        "master": meta.get("master", "").upper(),
                        "obsnum": meta.get("obsnum"),
                        "subobsnum": meta.get("subobsnum", 0),
                        "scannum": meta.get("scannum", 0),
                        # Exclude non-legacy fields: nw_id, description, obs_goal, source_name, tag, data_kind
                        **{k: v for k, v in meta.items() if k not in ["tag", "data_kind", "nw_id", "description", "obs_goal", "source_name"]},
                    },
                    "uid": dp["uid"],
                }
                index_dicts.append(index_dict)
            
            # Update index store with structured data
            self._index_store.update(index_dicts)

            return DataProdCollectorInfo(
                is_active=True,
                message=f"Collected {len(data_prods)} data products",
                query_cursor={"n_items": len(data_prods)},
            )

        except Exception as e:
            logger.exception(f"Error collecting data products: {e}")
            return DataProdCollectorInfo(
                is_active=False,
                message=f"Error: {e!s}",
            )

    def get_data_products(self) -> list[dict]:
        """Return the cached list of data products.

        Returns
        -------
        list[dict]
            List of data product dictionaries
        """
        return self._cached_data_prods

    def get_data_product_by_uid(self, uid: str) -> dict | None:
        """Get a specific data product by UID.

        Parameters
        ----------
        uid : str
            Data product UID

        Returns
        -------
        dict or None
            Data product dictionary or None if not found
        """
        if self._adapter is None:
            return None

        try:
            return self._adapter.get_data_prod_by_uid(uid)
        except Exception as e:
            logger.error(f"Error fetching data product {uid}: {e}")
            return None

    def load_data_product_by_uid(self, uid: str) -> dict | None:
        """Load a data product by UID and convert to index dict format.
        
        This method is used for on-demand loading of associated data products
        that may not be in the initial collection.

        Parameters
        ----------
        uid : str
            Data product UID

        Returns
        -------
        dict or None
            Index dict for the data product, or None if not found
        """
        if self._adapter is None:
            return None

        try:
            # Query for this specific data product using efficient UID lookup
            dp = self._adapter.query_data_product_by_uid(uid)
            
            if not dp:
                logger.warning(f"Data product {uid} not found in database")
                return None
            
            # Convert to index dict format (same logic as in collect())
            sources = self._adapter.query_sources_for_data_product(dp["uid"])
            
            # Convert meta to dict if it's not already
            meta = dp.get("meta", {})
            if hasattr(meta, "__dict__"):
                meta = {k: v for k, v in meta.__dict__.items()}
            elif not isinstance(meta, dict):
                meta = {}
            
            # Get data_kind BEFORE converting enums
            data_kind_raw = meta.get("data_kind", 0)
            
            # Convert enum values to strings
            for key, value in list(meta.items()):
                if hasattr(value, "value"):
                    meta[key] = value.value
                elif hasattr(value, "name"):
                    meta[key] = value.name
            
            # Convert data_kind to legacy string format
            data_kind_str = _convert_data_kind_to_legacy(data_kind_raw)
            
            # Build data_items from sources
            data_items = []
            for source in sources:
                source_meta = source.get("meta", {})
                if hasattr(source_meta, "__dict__"):
                    source_meta = {k: v for k, v in source_meta.__dict__.items()}
                
                roach = source_meta.get("roach")
                interface = f"toltec{roach}" if roach is not None else None
                
                item_meta = {
                    "data_kind": data_kind_str,
                    "roach": roach,
                    "interface": interface,
                    "master": meta.get("master", "").upper(),
                    "obsnum": meta.get("obsnum"),
                    "subobsnum": meta.get("subobsnum"),
                    "scannum": meta.get("scannum"),
                    "name": f"{interface}-{meta.get('master', '').lower()}-{meta.get('obsnum')}-{meta.get('subobsnum')}-{meta.get('scannum')}" if interface else None,
                    "source": source.get("source_uri", ""),
                    **{k: v for k, v in source_meta.items() if k != "nw_id"},
                }
                item_meta["data_kind"] = data_kind_str
                
                data_items.append({
                    "filepath": source.get("filepath", source.get("source_uri", "")),  # Use absolute filepath
                    "meta": item_meta,
                })
            
            # Ensure data_prod_type is a string
            data_prod_type = dp.get("data_prod_type", "unknown")
            if hasattr(data_prod_type, "value"):
                data_prod_type = data_prod_type.value
            elif not isinstance(data_prod_type, str):
                data_prod_type = str(data_prod_type)
            
            # Populate associations
            assocs = []
            db_assocs = self._adapter.get_associations(dp["uid"])
            for db_assoc in db_assocs:
                assocs.append({
                    "data_prod_assoc_type": db_assoc.get("assoc_type", "unknown"),
                    "filepath": f"tolteca_db://{db_assoc['dst_uid']}"
                })
            
            index_dict = {
                "assocs": assocs,
                "data_items": data_items,
                "meta": {
                    "data_prod_type": data_prod_type,
                    "name": meta.get("name", f"dp_{dp['uid']}"),
                    "master": meta.get("master", "").upper(),
                    "obsnum": meta.get("obsnum"),
                    "subobsnum": meta.get("subobsnum", 0),
                    "scannum": meta.get("scannum", 0),
                    **{k: v for k, v in meta.items() if k not in ["tag", "data_kind", "nw_id", "description", "obs_goal", "source_name"]},
                },
                "uid": dp["uid"],
            }
            
            return index_dict
            
        except Exception as e:
            logger.exception(f"Error loading data product {uid}: {e}")
            return None

    def get_associations(self, uid: str) -> list[dict]:
        """Get associations for a data product.

        Parameters
        ----------
        uid : str
            Data product UID

        Returns
        -------
        list[dict]
            List of associated data products
        """
        if self._adapter is None:
            return []

        try:
            return self._adapter.get_associations(uid)
        except Exception as e:
            logger.error(f"Error fetching associations for {uid}: {e}")
            return []


def create_tolteca_db_collector(
    db_url: str | None = None,
) -> ToltecaDBDataProdCollector | None:
    """Factory function to create ToltecaDBDataProdCollector.

    Parameters
    ----------
    db_url : str, optional
        Database URL. If not provided, tries environment variable.

    Returns
    -------
    ToltecaDBDataProdCollector or None
        Collector instance or None if db_url not available
    """
    if db_url is None:
        db_url = os.environ.get("TOLTECA_DB_URL", None)

    if db_url is None:
        logger.warning("No TOLTECA_DB_URL provided, tolteca_db collector disabled")
        return None

    try:
        collector = ToltecaDBDataProdCollector(db_url=db_url)
        logger.info("Created tolteca_db data product collector")
        return collector
    except Exception as e:
        logger.error(f"Failed to create tolteca_db collector: {e}")
        return None
