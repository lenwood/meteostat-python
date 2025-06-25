"""
Core Class - Data Loader

Meteorological data provided by Meteostat (https://dev.meteostat.net)
under the terms of the Creative Commons Attribution-NonCommercial
4.0 International Public License.

The code is licensed under the MIT license.
"""

from io import BytesIO
from gzip import GzipFile
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from typing import Callable, List, Optional
import pandas as pd
from meteostat.core.warn import warn
import logging # Added import for logging

_log = logging.getLogger(__name__) # Initialize logger at the module level

def processing_handler(
    datasets: List, load: Callable[[dict], None], cores: int, threads: int
) -> None:
    """
    Load multiple datasets (simultaneously)
    """

    # Data output
    output = []

    # Multi-core processing
    if cores > 1 and len(datasets) > 1:
        # Create process pool
        with Pool(cores) as pool:
            # Process datasets in pool
            output = pool.starmap(load, datasets)

            # Wait for Pool to finish
            pool.close()
            pool.join()

    # Multi-thread processing
    elif threads > 1 and len(datasets) > 1:
        # Create process pool
        with ThreadPool(threads) as pool:
            # Process datasets in pool
            output = pool.starmap(load, datasets)

            # Wait for Pool to finish
            pool.close()
            pool.join()

    # Single-thread processing
    else:
        for dataset in datasets:
            output.append(load(*dataset))

    # Remove empty DataFrames
    filtered = list(filter(lambda df: not df.empty, output))

    return pd.concat(filtered) if len(filtered) > 0 else output[0]

def load_handler(
    endpoint: str,
    path: str,
    proxy: Optional[str] = None,
    names: Optional[List] = None,
    dtype: Optional[dict] = None,
    parse_dates: Optional[List] = None,
    default_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Loads a single CSV file into a DataFrame.

    This method has been updated to address the pandas FutureWarning
    "Support for nested sequences for 'parse_dates' is deprecated."
    It now always reads date components as separate columns and then
    manually constructs the datetime column if 'parse_dates' indicates
    a multi-column date. This aligns with modern pandas best practices.

    Arguments:
    endpoint -- API endpoint URL
    path -- Path to the data file
    proxy -- HTTP proxy
    names -- names of the columns in the CSV
    dtype -- explicit data type for columns
    parse_dates -- original parse_dates argument (used to identify which columns to combine)
    default_df -- Default DataFrame to return on error
    """

    # Determine if a multi-column date parsing is intended by checking if
    # 'parse_dates' is provided and contains a nested list (e.g., [['year', 'month', 'day']]).
    is_multi_column_date = (
        isinstance(parse_dates, list) and
        len(parse_dates) > 0 and
        isinstance(parse_dates[0], list)
    )

    time_components = []
    target_time_col_name = None
    is_hourly_data = False

    if is_multi_column_date:
        # Extract the list of individual columns that form the date/time
        time_components = parse_dates[0]

        # Infer the name of the final combined datetime column.
        # In Meteostat's common usage, this is typically 'time' or 'time_local',
        # and it's usually the first column defined in the 'names' list.
        if names and len(names) > 0 and (names[0] == 'time' or names[0] == 'time_local'):
            target_time_col_name = names[0]
        elif 'time' in (names or []):
            target_time_col_name = 'time'
        elif 'time_local' in (names or []):
            target_time_col_name = 'time_local'
        elif names:
            # Fallback for unexpected scenarios, assuming first column in names
            # is the intended target for the datetime index.
            _log.warning(
                f"Could not definitively identify primary time column from names: {names}. "
                f"Assuming '{names[0]}' for datetime index construction."
            )
            target_time_col_name = names[0]
        else:
            # This case means 'names' is empty or None, which is problematic
            # if we need to identify the target column. Disable multi-column parsing.
            _log.error(
                f"Multi-column date detected via 'parse_dates' but 'names' list is empty or None. "
                f"Cannot determine target column. Skipping manual date parsing."
            )
            is_multi_column_date = False # Disable manual parsing

        # Determine if data includes an 'hour' component, implying hourly data.
        is_hourly_data = ('hour' in time_components)

    try:
        handlers = []
        if proxy:
            handlers.append(ProxyHandler({"http": proxy, "https": proxy}))

        # Read CSV file.
        # We explicitly set parse_dates to None because we will handle multi-column
        # date parsing manually *after* reading, aligning with Pandas' recommendation.
        # date_parser is also set to None as it's typically used in conjunction with parse_dates.
        with build_opener(*handlers).open(Request(endpoint + path)) as response:
            with GzipFile(fileobj=BytesIO(response.read()), mode="rb") as file:
                df = pd.read_csv(
                    file,
                    names=names,
                    dtype=dtype,
                    parse_dates=None, # Always None as we handle manually for multi-column
                    date_parser=None, # Always None
                )

        # If a multi-column date was detected, perform manual conversion and indexing.
        if is_multi_column_date and time_components and target_time_col_name:
            # Verify that all required component columns are present in the DataFrame.
            if not all(col in df.columns for col in time_components):
                _log.warning(
                    f"Missing one or more expected time component columns ({time_components}) "
                    f"in DataFrame from {path}. Cannot construct '{target_time_col_name}' column. "
                    f"Proceeding without datetime index for this file."
                )
            else:
                # Construct the datetime string from the component columns.
                # Use .str.zfill for consistent formatting (e.g., '1' -> '01').
                if not is_hourly_data:
                    # For daily data: 'year', 'month', 'day' -> 'YYYY-MM-DD'
                    datetime_str = (
                        df['year'].astype(str).str.zfill(4) + '-' +
                        df['month'].astype(str).str.zfill(2) + '-' +
                        df['day'].astype(str).str.zfill(2)
                    )
                    datetime_format = '%Y-%m-%d'
                else:
                    # For hourly data: 'year', 'month', 'day', 'hour' -> 'YYYY-MM-DD HH:MM:SS'
                    # Meteostat's hourly data assumes minutes and seconds are '00'.
                    datetime_str = (
                        df['year'].astype(str).str.zfill(4) + '-' +
                        df['month'].astype(str).str.zfill(2) + '-' +
                        df['day'].astype(str).str.zfill(2) + ' ' +
                        df['hour'].astype(str).str.zfill(2) + ':00:00'
                    )
                    datetime_format = '%Y-%m-%d %H:%M:%S'

                # Convert the combined string to datetime objects.
                # errors='coerce' will convert unparseable dates to NaT (Not a Time).
                df[target_time_col_name] = pd.to_datetime(datetime_str, format=datetime_format, errors='coerce')

                # Set the newly created datetime column as the DataFrame index.
                df = df.set_index(target_time_col_name)

                # Drop the original date component columns now that they've been used.
                df = df.drop(columns=time_components, errors='ignore')

        # If parse_dates was provided but was NOT a multi-column nested list (e.g., ['single_date_col']),
        # then we should still attempt to parse those single columns here.
        # This part handles the case where `parse_dates` might be `['date_column_name']`
        # for other types of CSVs Meteostat might load.
        elif parse_dates: # If parse_dates is not None and not a multi-column date
            for col in parse_dates:
                # Only process if it's a simple column name (not a nested list)
                if isinstance(col, str) and col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # If this column was intended to be the index, set it now.
                    # This is less explicit than Meteostat's common 'time' column,
                    # so this might need fine-tuning if other data types become problematic.
                    # For now, we assume 'time' or 'time_local' is always handled by the multi-column logic.
                    # This block primarily handles single-column date parsing if it ever occurs.
                    if col == target_time_col_name: # Re-use target_time_col_name logic if applicable
                         df = df.set_index(col)


    except (FileNotFoundError, HTTPError) as e:
        _log.error(f"Error loading data from {endpoint + path}: {e}")
        df = default_df if default_df is not None else pd.DataFrame(columns=names)
        warn(f"Cannot load {path} from {endpoint}")

    return df
