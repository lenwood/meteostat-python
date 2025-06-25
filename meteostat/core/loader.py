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
    Loads a single CSV file and returns a Pandas DataFrame.

    This method has been updated to address the pandas FutureWarning
    "Support for nested sequences for 'parse_dates' is deprecated."
    It now manually constructs the datetime column after reading the CSV
    if the original 'parse_dates' argument indicates such a nested format.

    Arguments:
    endpoint -- API endpoint URL
    path -- Path to the data file
    proxy -- HTTP proxy
    names -- names of the columns in the CSV
    dtype -- explicit data type for columns
    parse_dates -- original parse_dates argument (will be used to detect and then bypass nested sequence)
    default_df -- Default DataFrame to return on error
    """

    # --- Start of proposed fix for pandas FutureWarning ---

    # Check if parse_dates is a nested sequence (e.g., [['year', 'month', 'day']]),
    # which is the deprecated format causing the FutureWarning.
    is_nested_parse_dates = (
        isinstance(parse_dates, list) and
        len(parse_dates) > 0 and
        isinstance(parse_dates[0], list)
    )

    time_components = []  # e.g., ['year', 'month', 'day'] or ['year', 'month', 'day', 'hour']
    target_time_col_name = None
    is_hourly_data = False # Flag to determine if 'hour' component is present

    if is_nested_parse_dates:
        time_components = parse_dates[0] # Extract the list of time component column names

        # Infer the target time column name. In Meteostat, this is typically 'time' or 'time_local',
        # and it's usually the first column defined in the 'names' list.
        if names and len(names) > 0 and (names[0] == 'time' or names[0] == 'time_local'):
            target_time_col_name = names[0]
        elif 'time' in names: # Fallback if 'time' isn't the first column but is present
            target_time_col_name = 'time'
        elif 'time_local' in names: # Fallback if 'time_local' isn't the first column but is present
            target_time_col_name = 'time_local'
        elif names: # As a last resort, assume the first column in 'names' is the target
            _log.warning(
                f"Could not definitively identify primary time column from names: {names}. "
                f"Assuming '{names[0]}' for datetime index construction."
            )
            target_time_col_name = names[0]
        else:
            # This case means 'names' is empty or None, which is unexpected if parse_dates is nested.
            _log.error(
                f"Nested 'parse_dates' detected but 'names' list is empty or None. "
                f"Cannot construct datetime index. Skipping manual parsing."
            )
            is_nested_parse_dates = False # Disable manual parsing if names is missing

        # Determine if data is hourly based on the presence of 'hour' in time_components
        is_hourly_data = ('hour' in time_components)

    try:
        handlers = []

        # Set a proxy
        if proxy:
            handlers.append(ProxyHandler({"http": proxy, "https": proxy}))

        # Read CSV file from Meteostat endpoint
        with build_opener(*handlers).open(Request(endpoint + path)) as response:
            # Decompress the content
            with GzipFile(fileobj=BytesIO(response.read()), mode="rb") as file:
                # If we detected a nested parse_dates, set parse_dates to None for pd.read_csv
                # to avoid the FutureWarning and handle parsing manually later.
                df = pd.read_csv(
                    file,
                    names=names,
                    dtype=dtype,
                    parse_dates=None if is_nested_parse_dates else parse_dates,
                    # date_parser is also deprecated with nested sequences, so it's handled similarly.
                    date_parser=None if is_nested_parse_dates else None, # Meteostat doesn't pass a custom date_parser here anyway
                )

        # Manual datetime conversion and index setting if a nested parse_dates was detected
        if is_nested_parse_dates and time_components and target_time_col_name:
            # Ensure all required time component columns exist in the DataFrame after read.
            if not all(col in df.columns for col in time_components):
                _log.warning(
                    f"Missing one or more expected time component columns ({time_components}) "
                    f"in DataFrame from {path}. Cannot construct '{target_time_col_name}' column. "
                    f"Proceeding without datetime index for this file."
                )
                # If critical columns are missing, skip further time index processing
                # and leave the DataFrame as is (without a datetime index).
            else:
                # Construct the datetime string based on whether it's hourly or daily data.
                # Use .str.zfill to ensure consistent two-digit months/days/hours and four-digit years.
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
                    # Assuming minutes and seconds are always '00' for Meteostat hourly data.
                    datetime_str = (
                        df['year'].astype(str).str.zfill(4) + '-' +
                        df['month'].astype(str).str.zfill(2) + '-' +
                        df['day'].astype(str).str.zfill(2) + ' ' +
                        df['hour'].astype(str).str.zfill(2) + ':00:00'
                    )
                    datetime_format = '%Y-%m-%d %H:%M:%S'

                # Create the datetime column. Using errors='coerce' will turn invalid parses into NaT.
                df[target_time_col_name] = pd.to_datetime(datetime_str, format=datetime_format, errors='coerce')

                # Set the newly created datetime column as the DataFrame index.
                df = df.set_index(target_time_col_name)

                # Drop the original component columns after forming the datetime index.
                # `errors='ignore'` prevents errors if a column is somehow already dropped or not found.
                df = df.drop(columns=time_components, errors='ignore')

    except (FileNotFoundError, HTTPError):
        df = default_df if default_df is not None else pd.DataFrame(columns=names)

        # Display warning using Meteostat's custom warn function
        warn(f"Cannot load {path} from {endpoint}")

    # Return DataFrame
    return df
