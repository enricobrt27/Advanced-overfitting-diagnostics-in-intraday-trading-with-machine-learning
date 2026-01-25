import pandas as pd
import glob, re, os
from pathlib import Path
import fastparquet

#CONFIG AREA 

TZ = "UTC"                           #assume/convert to UTC
EXPECTED_FREQ = pd.Timedelta(minutes=1)

# Columns expected 
OHLC_COLS = ["open", "high", "low", "close"]

# Gap thresholds 
MIN_SHORT_GAP = pd.Timedelta(minutes=1)
MAX_SHORT_GAP = pd.Timedelta(days=2)
MIN_LONG_GAP  = pd.Timedelta(days=2)        # to ignore weekends
MAX_LONG_GAP  = pd.Timedelta(days=10)

# Stale quote detection
MAX_STALE_RUN = 60   # consecutive minutes with identical close → mark as stale

# FX session filter (UTC): Sunday 22:00 → Friday 22:00
APPLY_FX_SESSION_FILTER = True

def ensure_datetime_utc(df: pd.DataFrame, time_col: str, tz: str = "UTC") -> pd.DataFrame:
    #Parse time_col to timezone-aware UTC datetimes. Sort & drop NA timestamps.
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    if tz and tz.upper() != "UTC":
        #convert here to differenz time zones:
        df[time_col] = df[time_col].dt.tz_convert(tz)
    #Set to UTC:
    if tz and tz.upper() != "UTC":
        df[time_col] = df[time_col].dt.tz_convert("UTC")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df

def drop_duplicate_timestamps(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    #Keep last observation for duplicated timestamps.
    before = len(df)
    df = df.drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicated timestamps.")
    return df

def ohlc_sanity_filter(df: pd.DataFrame, ohlc_cols: list[str]) -> pd.DataFrame:
    #Remove rows where OHLC are inconsistent or zero/negative.
    df = df.copy()
    # Mark invalid OHLC zeros/negatives
    invalid_zero = (df[ohlc_cols] == 0).any(axis=1)
    invalid_neg  = (df[ohlc_cols] < 0).any(axis=1)

    # OHLC logical constraints
    o, h, l, c = ohlc_cols
    invalid_logic = (df[h] < df[[o, c]].max(axis=1)) | (df[l] > df[[o, c]].min(axis=1)) | (df[h] < df[l])

    invalid = invalid_zero | invalid_neg | invalid_logic
    removed = int(invalid.sum())
    if removed:
        print(f"Removed {removed} rows failing OHLC sanity checks.")
    return df.loc[~invalid].reset_index(drop=True)

def remove_stale_quotes(df: pd.DataFrame, time_col: str, price_col: str = "close", max_run: int = 60) -> pd.DataFrame:
    #Remove long runs of identical prices (stale quotes).
    if price_col not in df.columns:
        return df
    df = df.copy()
    # Identify consecutive equal prices
    same = df[price_col].eq(df[price_col].shift(1))
    # Run-length encoding
    run_id = (~same).cumsum()
    run_len = run_id.map(run_id.value_counts())
    stale = same & (run_len >= max_run)
    removed = int(stale.sum())
    if removed:
        print(f"Removed {removed} stale-quote rows (runs ≥ {max_run}).")
    return df.loc[~stale].reset_index(drop=True)

def fx_session_filter_utc(df, time_col="timestamp"):
    df = df.copy()
    ts = df[time_col]

    weekday = ts.dt.weekday     # Monday=0 ... Sunday=6
    hour    = ts.dt.hour
    minute  = ts.dt.minute

    # Monday → Thursday : always active
    in_week = weekday.between(0, 3)

    # Friday until 22:00
    is_fri_active = (weekday == 4) & ((hour < 22) | ((hour == 22) & (minute == 0)))

    # Sunday >= 22:00 (CORRECTED)
    is_sun_active = (weekday == 6) & ((hour > 22) | ((hour == 22) & (minute >= 0)))

    mask = in_week | is_fri_active | is_sun_active
    return df.loc[mask].reset_index(drop=True)

def detect_missing_periods(df: pd.DataFrame,
                           time_col: str,
                           min_gap: pd.Timedelta,
                           max_gap: pd.Timedelta) -> pd.DataFrame:
    
    #Return rows (the 'next' timestamps) where time diff > min_gap and < max_gap.
    #Also compute precise missing segment start/end (excluding valid borders).
    
    df = df.copy()
    df = df.sort_values(time_col).reset_index(drop=True)
    df["diff"] = df[time_col].diff()
    mp = df[(df["diff"] > min_gap) & (df["diff"] <= max_gap)].copy()

    # Build exact gap table
    gaps = []
    for idx, row in mp.iterrows():
        prev_ts = df.loc[idx - 1, time_col]
        next_ts = row[time_col]
        start_missing = prev_ts + EXPECTED_FREQ
        end_missing   = next_ts - EXPECTED_FREQ
        n_missing_min = max(int(row["diff"] / EXPECTED_FREQ) - 1, 0)
        gaps.append({
            "gap_start": start_missing,
            "gap_end": end_missing,
            "gap_duration_min": n_missing_min,
            "gap_diff": row["diff"]
        })
    cols = ["gap_start", "gap_end", "gap_duration_min", "gap_diff"]
    return pd.DataFrame(gaps, columns=cols)

def detect_invalid_blocks(df: pd.DataFrame,
                          time_col: str,
                          ohlc_cols: list[str],
                          min_block: pd.Timedelta) -> pd.DataFrame:
    
    #Identify consecutive blocks where OHLC are invalid (NaN or zero),
    #grouped at 1-minute expected frequency. Returns block start/end/duration.
   
    df = df.copy().sort_values(time_col).reset_index(drop=True)
    # invalid row definition
    invalid = df[ohlc_cols].isna().any(axis=1) | (df[ohlc_cols] == 0).any(axis=1)
    df["invalid"] = invalid

    step = df[time_col].diff()
    same_day   = df[time_col].dt.floor("D").eq(df[time_col].shift(1).dt.floor("D"))
    step_ok    = step.eq(EXPECTED_FREQ)
    same_state = df["invalid"].eq(df["invalid"].shift(1))

    new_block = (~same_day) | (~step_ok) | (~same_state)
    df["block_id"] = new_block.cumsum()

    blocks = (
        df.groupby("block_id")
          .agg(start=(time_col, "first"),
               end=(time_col, "last"),
               is_invalid=("invalid", "first"),
               n_rows=(time_col, "size"))
          .reset_index(drop=True)
    )
    blocks["duration"] = (blocks["end"] - blocks["start"]) + EXPECTED_FREQ
    return blocks[(blocks["is_invalid"]) & (blocks["duration"] >= min_block)].reset_index(drop=True)


def clean_transform_pipeline(file_path: str, TIME_COL: str) -> dict:
    
    #Full pipeline:
    #  1) structural cleaning
    #  2) missing-data detection (short/long gaps + invalid blocks)
    #  3) optional FX session filter

    #Returns a dict with cleaned df and artifacts (gaps, blocks).

    print(f"Loading {os.path.basename(file_path)} ...")
    usecols = [TIME_COL] + [c for c in OHLC_COLS if c] 
    usecols = [c for c in usecols if c is not None]  # drop Nones

    df = pd.read_csv(file_path, usecols=lambda c: c in set(usecols))
    df = ensure_datetime_utc(df, TIME_COL, tz="UTC")
    
    # Optional: filter to FX continuous window (UTC)
    if APPLY_FX_SESSION_FILTER:
        df = fx_session_filter_utc(df, TIME_COL) 
    
    df = drop_duplicate_timestamps(df, TIME_COL)
    df = ohlc_sanity_filter(df, OHLC_COLS)
    df = remove_stale_quotes(df, TIME_COL, price_col="close", max_run=MAX_STALE_RUN)

    # Detect gaps
    short_gaps = detect_missing_periods(df, TIME_COL, MIN_SHORT_GAP, MAX_SHORT_GAP)
    long_gaps  = detect_missing_periods(df, TIME_COL, MIN_LONG_GAP,  MAX_LONG_GAP)

    # Detect invalid blocks (NaN/zero OHLC runs)
    invalid_blocks = detect_invalid_blocks(df, TIME_COL, OHLC_COLS, min_block=MIN_SHORT_GAP)

   
    
    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
        
    out = {
        "clean_df": df.reset_index(drop=True),
        "short_gaps": short_gaps,
        "long_gaps": long_gaps,
        "invalid_blocks": invalid_blocks
    }

    # Console summary
    print(f"Rows after cleaning: {len(df)}")
    print(f"Short gaps (>1m ≤2d): {len(short_gaps)}")
    print(f"Long gaps  (>2d):    {len(long_gaps)}")
    print(f"Invalid blocks:      {len(invalid_blocks)}")

    return out




def main():    
    
    BASE_DIR = Path("C:/Users/enric/Desktop/tesi/gaps.py").resolve().parent
    DATA_DIR = BASE_DIR / "data"

    #This dictionary contains input and output folders, 
    #time is used to recognise the time column while suffix is the suffix used to recognise the datasets.
    
    folders = [
        {
            "input":  DATA_DIR / "dukascopy" / "raw",
            "output": DATA_DIR / "dukascopy" / "cleaned",
            "time":   "timestamp",
            "suffix": "*_2007-01-01_2025-09-30.csv",
        },
        {
            "input":  DATA_DIR / "ibkr" / "raw",
            "output": DATA_DIR / "ibkr" / "cleaned",
            "time":   "date",
            "suffix": "*_historical_data_allhours.csv",
        },
    ]

    
    for folder in folders:
            
            #Process all files in input_folder ending with 'file_suffix' and
            #save cleaned datasets to output_folder.
        
            #input_folder:  directory containing raw files
            #output_folder: directory where cleaned files will be written
            inputf = folder["input"]
            outputf = folder["output"]
            suffix = folder["suffix"]
            TIME_COL = folder["time"]
            print(inputf)
            #Create output folder if not existing
            os.makedirs(outputf, exist_ok=True)
        
            #Get list of matching files
            pattern = os.path.join(inputf, suffix)
            files = glob.glob(pattern)
        
        
            print(f"Found {len(files)} files.")
        
            #Process each file
            for file_path in files:
                fname = os.path.basename(file_path)
                print(f"\n=== Processing {fname} ===")
                m = re.search(r"([A-Z]{6})_(BID|ASK)", fname)
                if not m:
                    m = re.search(r"([A-Z]{3}_[A-Z]{3})_(BID|ASK)", fname.upper())
                    pair = m.group(1)
                    side = m.group(2)
                    pair_name = (f"{pair[:3]}{pair[3:]}_{side}")
                else:  
                    pair = m.group(1)
                    side = m.group(2)
                    pair_name = (f"{pair[:3]}_{pair[3:]}_{side}")
                    
                result = clean_transform_pipeline(file_path, TIME_COL)
                clean_df = result["clean_df"]
                short_gaps = result["short_gaps"]
                long_gaps = result["long_gaps"]
                invalid_blocks = result["invalid_blocks"]        
                
                
                out_file = os.path.join(outputf, f"{pair_name}_CLEAN.parquet")
                out_short = os.path.join(outputf, f"{pair_name}_short_gaps.parquet")
                out_long = os.path.join(outputf, f"{pair_name}_long_gaps.parquet")
                out_invalid = os.path.join(outputf, f"{pair_name}_invalid_blocks.parquet")
        
                #Save cleaned DF
                clean_df.to_parquet(out_file, index=False)
        
                # (optional) save diagnostics
                short_gaps.to_parquet(out_short, index=False)
                long_gaps.to_parquet(out_long, index=False)
                invalid_blocks.to_parquet(out_invalid, index=False)
        
                print(f"✅ Saved cleaned file: {out_file}")
        
        
        
        
            print("\nAll files processed successfully.")
        
        
if __name__ == "__main__":
    main() 
