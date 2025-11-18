"""Adapter module to bridge tolteca_db and tolteca_web interfaces.

Production Architecture Overview:
==================================

Service 1: Dagster (Continuous Ingestion)
------------------------------------------
- Monitors toltecdb for new observations (quartet_sensor polls every 10s)
- Detects complete quartets with timeout-based validation
- Ingests data products into tolteca_db (WRITE mode)
- Generates associations automatically (CalGroup, DriveFit, FocusGroup)
- Uses SQLite for multi-process concurrent writes

Service 2: Webapp (Visualization - THIS MODULE)
------------------------------------------------
- Polls tolteca_db for updates (LiveUpdateSection: 5-15 second intervals)
- Displays data products with automatic refresh (READ-ONLY mode)
- Uses SQLite for multi-process concurrent reads
- No notification system needed - polling handles updates

Database Architecture:
----------------------
- toltecdb (SQLite): Real-time data acquisition database from telescope
- tolteca_db (SQLite + DuckDB): Data product database
  - SQLite for metadata (multi-process/thread safe)
  - DuckDB for Parquet queries (high-performance analytics)
- tolteca_web (this adapter): Queries tolteca_db for structured data

This adapter provides a backward-compatible dictionary-based interface
to tolteca_db's ORM models, replacing direct toltecdb queries and YAML caching.

Configuration:
--------------
    TOLTECA_DB_URL should use SQLite for production deployment:
    - sqlite:///path/to/tolteca_metadata.db (recommended)
    - Database created/updated by Dagster service
    - Webapp opens in read-only mode for concurrent access
    
Polling Mechanism:
------------------
    The webapp uses LiveUpdateSection with configurable intervals:
    - User selects: 5s, 10s, or 15s (default: 5s)
    - Timer automatically triggers ToltecaDBCollector.collect()
    - No server-side notifications needed
    - See: tolteca_web/common/liveupdatesection.py
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Integer
from tollan.utils.log import logger

from tolteca_db.db import Database, create_database


@dataclass
class ToltecaDBAdapter:
    """Adapter to make tolteca_db compatible with tolteca_web interfaces.
    
    Parameters
    ----------
    db_url : str
        Database URL. For webapp use sqlite:/// for multi-process safety.
        Example: "sqlite:///scratch/tolteca_metadata.db"
    """

    db_url: str
    _database: Database | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the database connection."""
        try:
            # Use read_only=True for webapp (safe for concurrent reads)
            self._database = create_database(
                database_url=self.db_url,
                read_only=True,  # Webapp only reads, ingestion writes
                echo=False,
            )
            # Test connection
            with self._database.session() as session:
                logger.info(f"Successfully connected to tolteca_db: {self.db_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Database: {e}")
            self._database = None

    def get_session(self):
        """Get a database session context manager."""
        if self._database is None:
            raise RuntimeError("Database not initialized")
        return self._database.session()

    def query_data_products(
        self,
        limit: int = 100,
        order_by: str = "created_at",
        data_prod_type: str | None = None,
        master: str | None = None,
        min_obsnum: int | None = None,
        max_obsnum: int | None = None,
        obs_date: str | None = None,
    ) -> list[Any]:
        """Query data products from tolteca_db with optional filters.

        Parameters
        ----------
        data_prod_type : str, optional
            Filter by data product type label
        master : str, optional
            Master to filter by ('ics' or 'tcs')
        min_obsnum : int, optional
            Minimum observation number to include
        max_obsnum : int, optional
            Maximum observation number to include
        obs_date : str, optional
            Observation date to filter by (YYYY-MM-DD format)
        limit : int
            Maximum number of results to return
        order_by : str
            Field to order results by (default: created_at)

        Returns
        -------
        list[dict]
            List of data product dictionaries with metadata and associations
        """
        if self._database is None:
            logger.error("Database not initialized")
            return []

        try:
            with self.get_session() as session:
                from tolteca_db.models import DataProd

                query = session.query(DataProd)

                # Filter by data_prod_type using relationship has()
                if data_prod_type is not None:
                    # Use has() to filter by the related DataProdType's label
                    query = query.filter(
                        DataProd.data_prod_type.has(label=data_prod_type)
                    )
                
                # Filter by master flag (meta is JSON column)
                if master is not None:
                    # Use json_extract function for SQLite compatibility
                    query = query.filter(
                        sa.func.json_extract(DataProd.meta, "$.master") == master
                    )
                
                # Filter by obsnum range (meta is JSON column)
                if min_obsnum is not None:
                    # Use JSON path extraction and cast to integer
                    query = query.filter(
                        sa.cast(DataProd.meta["obsnum"], Integer) >= min_obsnum
                    )
                
                if max_obsnum is not None:
                    # Use JSON path extraction and cast to integer
                    query = query.filter(
                        sa.cast(DataProd.meta["obsnum"], Integer) <= max_obsnum
                    )
                
                # Filter by specific date (use obs_datetime if available, fallback to created_at)
                # Use SQLite's DATE() function for date comparison
                if obs_date is not None:
                    # Try obs_datetime first (stored as ISO8601 string in JSON), fallback to created_at
                    obs_date_filter = sa.or_(
                        sa.func.date(DataProd.meta["obs_datetime"].as_string()) == obs_date,
                        sa.and_(
                            DataProd.meta["obs_datetime"].as_string() == None,
                            sa.func.date(DataProd.created_at) == obs_date
                        )
                    )
                    query = query.filter(obs_date_filter)

                # Order by specified field (default: most recent first)
                if hasattr(DataProd, order_by):
                    query = query.order_by(getattr(DataProd, order_by).desc())

                query = query.limit(limit)
                results = query.all()

                # Convert to dictionaries with metadata
                data_prods = []
                for dp in results:
                    # Get primary source URI (first source with PRIMARY role)
                    uri = None
                    if dp.sources:
                        primary_sources = [s for s in dp.sources if s.role == "PRIMARY"]
                        if primary_sources:
                            uri = primary_sources[0].source_uri
                        elif dp.sources:
                            uri = dp.sources[0].source_uri
                    
                    dp_dict = {
                        "uid": str(dp.pk),  # Use pk as uid
                        "data_prod_type": dp.data_prod_type.label if dp.data_prod_type else None,
                        "uri": uri,
                        "created_at": dp.created_at.isoformat()
                        if dp.created_at
                        else None,
                        "meta": dp.meta if hasattr(dp, "meta") else {},
                    }
                    data_prods.append(dp_dict)

                logger.debug(f"Queried {len(data_prods)} data products")
                return data_prods

        except Exception as e:
            logger.exception(f"Error querying data products: {e}")
            return []

    def query_raw_observations(
        self,
        obsnum: int | None = None,
        obstype: str | None = None,
        master: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query raw observations.

        Parameters
        ----------
        obsnum : int, optional
            Observation number
        obstype : str, optional
            Observation type
        master : str, optional
            Master network
        limit : int
            Maximum results

        Returns
        -------
        list[dict]
            List of observation metadata dictionaries
        """
        with self.get_session() as session:
            from tolteca_db.repository import DataProdRepository

            repo = DataProdRepository(session)
            
            # Query data products
            if obsnum is not None:
                # TODO: Add find_by_obsnum to repository
                results = repo.find_all(limit=limit)
                # Filter by obsnum in meta
                results = [r for r in results if r.meta.get('obsnum') == obsnum]
            else:
                results = repo.find_all(limit=limit)

            # Convert to dictionaries
            observations = []
            for result in results:
                if hasattr(result, "meta") and isinstance(result.meta, dict):
                    obs_data = result.meta.copy()
                    obs_data["uid"] = result.uid
                    obs_data["location_pk"] = result.location_pk
                    obs_data["created_at"] = result.created_at
                    observations.append(obs_data)

            return observations

    def get_data_prod_by_uid(self, uid: str) -> dict | None:
        """Get data product by UID.

        Parameters
        ----------
        uid : str
            Data product UID

        Returns
        -------
        dict or None
            Data product metadata
        """
        with self.get_session() as session:
            from tolteca_db.repository import DataProdRepository

            repo = DataProdRepository(session)
            data_prod = repo.get_by_id(uid)

            if data_prod is None:
                return None

            # Get primary source URI
            uri = None
            if data_prod.sources:
                primary_sources = [s for s in data_prod.sources if s.role == "PRIMARY"]
                if primary_sources:
                    uri = primary_sources[0].source_uri
                elif data_prod.sources:
                    uri = data_prod.sources[0].source_uri

            return {
                "uid": str(data_prod.pk),
                "data_prod_type": data_prod.data_prod_type.label if data_prod.data_prod_type else None,
                "uri": uri,
                "lifecycle_status": data_prod.lifecycle_status,
                "meta": data_prod.meta,
                "created_at": data_prod.created_at,
                "updated_at": data_prod.updated_at,
            }

    def query_sources_for_data_product(self, uid: str) -> list[dict]:
        """Query sources for a data product.
        
        Parameters
        ----------
        uid : str
            Data product UID (pk)
            
        Returns
        -------
        list[dict]
            List of source dictionaries with uri and metadata
        """
        if self._database is None:
            logger.error("Database not initialized")
            return []
        
        try:
            with self.get_session() as session:
                from tolteca_db.models.orm import DataProdSource
                
                # Query sources for this data product
                sources = (
                    session.query(DataProdSource)
                    .filter(DataProdSource.data_prod_fk == int(uid))
                    .all()
                )
                
                result = []
                for source in sources:
                    result.append({
                        "source_uri": source.source_uri,
                        "role": source.role,
                        "meta": source.meta if source.meta else {},
                        "availability_state": source.availability_state,
                    })
                
                return result
                
        except Exception as e:
            logger.exception(f"Error querying sources for {uid}: {e}")
            return []

    def get_associations(self, uid: str) -> list[dict]:
        """Get associations for a data product.

        This queries the DataProdAssoc table which is populated by
        tolteca_db's AssociationGenerator (using CalGroupCollator,
        DriveFitCollator, FocusGroupCollator).

        Parameters
        ----------
        uid : str
            Data product UID

        Returns
        -------
        list[dict]
            List of associated data products with their metadata
        """
        if self._database is None:
            logger.error("Database not initialized")
            return []

        try:
            with self.get_session() as session:
                from tolteca_db.models.orm import DataProd, DataProdAssoc

                # Get associations where this product is the source
                assocs = (
                    session.query(DataProdAssoc)
                    .filter(DataProdAssoc.src_data_prod_fk == uid)
                    .all()
                )

                # Convert to dictionaries with full product info
                result = []
                for assoc in assocs:
                    # Get the destination product
                    dst_prod = (
                        session.query(DataProd)
                        .filter(DataProd.pk == assoc.dst_data_prod_fk)
                        .first()
                    )

                    if dst_prod:
                        result.append(
                            {
                                "src_uid": assoc.src_data_prod_fk,
                                "dst_uid": assoc.dst_data_prod_fk,
                                "assoc_type": (
                                    assoc.data_prod_assoc_type.label
                                    if assoc.data_prod_assoc_type
                                    else None
                                ),
                                "dst_data_prod_type": (
                                    dst_prod.data_prod_type.label
                                    if dst_prod.data_prod_type
                                    else None
                                ),
                                "dst_meta": dst_prod.meta if dst_prod.meta else {},
                            }
                        )

                return result

        except Exception as e:
            logger.exception(f"Error getting associations for {uid}: {e}")
            return []


@functools.lru_cache(maxsize=1)
def get_tolteca_db_adapter(db_url: str | None = None) -> ToltecaDBAdapter | None:
    """Get or create tolteca_db adapter instance.

    Parameters
    ----------
    db_url : str, optional
        Database URL. If not provided, returns None.

    Returns
    -------
    ToltecaDBAdapter or None
        Adapter instance or None if db_url not provided
    """
    if db_url is None:
        logger.warning("No tolteca_db URL provided, adapter disabled")
        return None

    try:
        adapter = ToltecaDBAdapter(db_url=db_url)
        logger.info("Initialized tolteca_db adapter")
        return adapter
    except Exception as e:
        logger.error(f"Failed to initialize tolteca_db adapter: {e}")
        return None
