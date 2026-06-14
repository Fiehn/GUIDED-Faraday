"""Visualise a raw time window from parquet files and the effect of a Normalizer.

Usage examples:
  python src/evaluation/visualise_norm.py --parquet-dir /Volumes/USB/parq_hourly \
	  --start 2022-01-01T00:00 --duration 24 --meter-id 12345 --normalizer models/normalizer.pt

The script will read parquet files under `--parquet-dir`, extract rows for the
requested time window (and optional meter id), pick sensible numeric feature
columns (phases if present) and plot raw vs normalised series side-by-side.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from data.normalizer import Normalizer


def find_time_column(df: pd.DataFrame) -> Optional[str]:
    # Check in regular columns first
    candidates = ['timestamp_dst', 'timestamp', 'time', 'ts', 'datetime', 'utc']
    for c in candidates:
        if c in df.columns:
            return c
    # Check in index
    if df.index.name and df.index.name in candidates:
        return df.index.name
    for c in candidates:
        if c == df.index.name:
            return c
    # fallback: any datetime-typed column
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # fallback: check index for datetime
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return df.index.name
    return None


def pick_feature_columns(df: pd.DataFrame, time_col: str, id_col: Optional[str]) -> list[str]:
    exclude = {time_col}
    if id_col:
        exclude.add(id_col)
    # common phase columns
    phases = [c for c in df.columns if 'active_power' in c or 'phase' in c or 'l1' in c.lower() or 'l2' in c.lower() or 'l3' in c.lower()]
    if phases:
        return [c for c in phases if c not in exclude]
    # otherwise pick numeric columns
    numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return numeric[:3] if len(numeric) > 3 else numeric


def load_window_from_parquet(parquet_dir: str, start: datetime, end: datetime, meter_id: Optional[int] = None) -> pd.DataFrame:
    """Load a window of data from parquet files, handling multi-indexed timestamp_dst column."""
    dataset = ds.dataset(parquet_dir, format='parquet')
    
    # Ensure start/end are timezone-aware (UTC)
    if start.tzinfo is None:
        start = start.replace(tzinfo=pd.Timestamp('2024-01-01', tz='UTC').tzinfo)
    if end.tzinfo is None:
        end = end.replace(tzinfo=pd.Timestamp('2024-01-01', tz='UTC').tzinfo)
    
    # Try to filter by timestamp_dst
    try:
        field = ds.field('timestamp_dst')
        # Use pd.Timestamp which handles timezone properly
        start_ts = pd.Timestamp(start).as_unit('us')
        end_ts = pd.Timestamp(end).as_unit('us')
        filtered_table = dataset.to_table(filter=(field >= start_ts) & (field < end_ts))
        if filtered_table.num_rows > 0:
            df = filtered_table.to_pandas()
            # The timestamp and meter_number might be in the index from parquet metadata
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            elif df.index.name in ('timestamp_dst', 'meter_number'):
                df = df.reset_index()
            
            # Ensure timestamp_dst is a proper datetime column
            if 'timestamp_dst' in df.columns:
                df['timestamp_dst'] = pd.to_datetime(df['timestamp_dst'])
            
            if meter_id is not None and 'meter_number' in df.columns:
                df = df[df['meter_number'] == int(meter_id)]
            
            return df
    except Exception as e:
        print(f"Filter attempt failed: {e}, falling back to full load...")

    # Fallback: load all and filter in memory
    table = dataset.to_table()
    df = table.to_pandas()
    
    # Handle multi-index or index columns
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name:
        df = df.reset_index()
    
    # Find and normalize timestamp column
    tcol = find_time_column(df)
    if tcol is None:
        raise RuntimeError('Could not find a timestamp column in parquet data. Available columns: ' + str(df.columns.tolist()))
    
    df[tcol] = pd.to_datetime(df[tcol])
    # Make comparison timezone-aware if needed
    start_cmp = pd.Timestamp(start)
    end_cmp = pd.Timestamp(end)
    mask = (df[tcol] >= start_cmp) & (df[tcol] < end_cmp)
    df = df.loc[mask]
    
    if meter_id is not None and 'meter_number' in df.columns:
        df = df[df['meter_number'] == int(meter_id)]
    
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet-dir', default='/Volumes/USB/parq_hourly')
    p.add_argument('--start', required=True, help='ISO start timestamp, e.g. 2022-01-01T00:00')
    p.add_argument('--duration', type=int, default=24, help='duration in hours')
    p.add_argument('--meter-id', type=int, default=None, help='meter_number to visualize (required; use --list-meters to see available)')
    p.add_argument('--list-meters', action='store_true', help='List available meters in the dataset and exit')
    p.add_argument('--normalizer-path', type=str, default=None)
    p.add_argument('--fit-on-window', action='store_true', help='If no normalizer given, fit one on the selected window')
    p.add_argument('--feature-cols', type=str, default=None, help='comma-separated feature columns to plot')
    p.add_argument('--out-file', type=str, default=None, help='Path to save the figure (PNG/PDF). If omitted, a name is auto-generated')
    p.add_argument('--broadcast-to-features', action='store_true', help='If normalizer expects multiple features but only one is provided, duplicate the single column to match expected feature dim (opt-in)')
    args = p.parse_args()

    start = datetime.fromisoformat(args.start)
    end = start + timedelta(hours=args.duration)

    # List available meters
    if args.list_meters:
        import pyarrow.dataset as ds
        dataset = ds.dataset(args.parquet_dir, format='parquet')
        table = dataset.to_table(['meter_number']).to_pandas()
        unique_meters = sorted(table['meter_number'].unique())
        print(f'Available meters ({len(unique_meters)} total):')
        print(unique_meters[:50], '...' if len(unique_meters) > 50 else '')
        return

    if args.meter_id is None:
        raise ValueError('--meter-id is required. Use --list-meters to see available meter IDs.')

    print('Reading parquet window', args.parquet_dir, start, '->', end, f'for meter {args.meter_id}')
    df = load_window_from_parquet(args.parquet_dir, start, end, meter_id=args.meter_id)
    if df is None or len(df) == 0:
        raise RuntimeError(f'No data found for the requested window (meter {args.meter_id})')

    print(f'Loaded {len(df)} rows')

    time_col = find_time_column(df)
    id_col = 'meter_number' if 'meter_number' in df.columns else None
    df = df.sort_values(time_col)

    if args.feature_cols:
        feat_cols = [c.strip() for c in args.feature_cols.split(',')]
    else:
        feat_cols = pick_feature_columns(df, time_col, id_col)

    if not feat_cols:
        raise RuntimeError('No numeric feature columns detected to plot')

    print('Using feature columns:', feat_cols)

    series = df.set_index(time_col)[feat_cols]

    # Load or compute normalizer
    normalizer = None
    raw_vals = None
    if args.normalizer_path:
        if not os.path.exists(args.normalizer_path):
            raise FileNotFoundError(f'normalizer file not found: {args.normalizer_path}')
        normalizer = Normalizer.load(args.normalizer_path, device='cpu')
        print('Loaded normalizer', args.normalizer_path, 'method=', normalizer.method)
    elif args.fit_on_window:
        print('Fitting a Normalizer on the selected window...')
        normalizer = Normalizer(method='log1p')
        # fit expects 2D or compatible; pass raw_vals
        raw_vals = series.values.astype(np.float32)
        try:
            normalizer.fit(raw_vals)
            print('Fitted normalizer from window, method=', normalizer.method)
        except Exception as e:
            print('Fitting normalizer failed:', e)
            normalizer = None
    
    if raw_vals is None:
        raw_vals = series.values.astype(np.float32)

    # Transform if normalizer is available
    norm_vals = None
    inv_vals = None
    if normalizer is not None:
        try:
            try:
                norm_vals = normalizer.transform(raw_vals)
            except ValueError as ve:
                # handle feature-dim mismatch gracefully
                msg = str(ve)
                print('Normalization failed:', msg)
                expected = normalizer._feature_dim_hint()
                if expected is not None and raw_vals.ndim == 2 and raw_vals.shape[1] == 1 and args.broadcast_to_features:
                    print(f'Attempting to broadcast single feature to expected dim {expected} (as requested by --broadcast-to-features)')
                    Xb = np.repeat(raw_vals, expected, axis=1)
                    norm_vals = normalizer.transform(Xb)
                    if hasattr(norm_vals, 'cpu'):
                        norm_vals = norm_vals.cpu().numpy()
                    else:
                        norm_vals = np.asarray(norm_vals)
                    # Use the first feature's normalization as representative for plotting
                    norm_vals = norm_vals[:, :1]
                else:
                    print('Normalizer expects feature dim', expected, 'but input has', raw_vals.shape[1])
                    print("Options: 1) use a matching normalizer (e.g. 'log1p' for aggregate series), 2) run with --fit-on-window, or 3) rerun with --broadcast-to-features to duplicate the single column to the expected feature count.")
                    norm_vals = None
            else:
                if hasattr(norm_vals, 'cpu'):
                    norm_vals = norm_vals.cpu().numpy()
                else:
                    norm_vals = np.asarray(norm_vals)

            # attempt inverse transform if we have norm_vals
            if norm_vals is not None:
                try:
                    inv_vals = normalizer.inverse_transform(norm_vals)
                    if hasattr(inv_vals, 'cpu'):
                        inv_vals = inv_vals.cpu().numpy()
                    else:
                        inv_vals = np.asarray(inv_vals)
                except Exception:
                    inv_vals = None
        except Exception as e:
            print('Normalization failed:', type(e).__name__, e)
            norm_vals = None
            inv_vals = None

    # Print sample values for debugging
    print('\n=== Data Comparison ===')
    print(f'Raw data shape: {raw_vals.shape}, dtype: {raw_vals.dtype}')
    print(f'Raw first 5 rows:\n{raw_vals[:5]}')
    print(f'Raw min/max: {raw_vals.min():.6f} / {raw_vals.max():.6f}')
    
    if norm_vals is not None:
        print(f'\nNormalized data shape: {norm_vals.shape}, dtype: {norm_vals.dtype}')
        print(f'Normalized first 5 rows:\n{norm_vals[:5]}')
        print(f'Normalized min/max: {norm_vals.min():.6f} / {norm_vals.max():.6f}')
    
    if inv_vals is not None:
        print(f'\nInverse-transformed shape: {inv_vals.shape}, dtype: {inv_vals.dtype}')
        print(f'Inverse first 5 rows:\n{inv_vals[:5]}')
        print(f'Inverse min/max: {inv_vals.min():.6f} / {inv_vals.max():.6f}')

    # Plotting: side-by-side before / after in same row per feature
    ncols = 2 if norm_vals is not None else 1
    fig, axes = plt.subplots(nrows=len(feat_cols), ncols=ncols, figsize=(6 * ncols, 3 * len(feat_cols)), squeeze=False)

    for i, col in enumerate(feat_cols):
        ax_raw = axes[i][0]
        hours = np.arange(raw_vals.shape[0])
        ax_raw.plot(hours, raw_vals[:, i]*1000, label='raw', color='C0', linewidth=2.0, marker='o', markersize=4)
        ax_raw.set_title(f'Raw: {col} — SM {args.meter_id} — {start.date()}')
        ax_raw.set_xlabel('Hour')
        ax_raw.set_ylabel(f'Consumption (Wh)')
        ax_raw.set_ylim(-1400, 2800)
        ax_raw.grid(True, alpha=0.3)

        if norm_vals is not None:
            ax_norm = axes[i][1]
            ax_norm.plot(hours, norm_vals[:, i]*1000, label='normalized', color='C1', linewidth=2.0, marker='o', markersize=4)
            ax_norm.set_title(f'Normalized: {col} — SM {args.meter_id} — {start.date()}')
            ax_norm.set_xlabel('Hour')
            ax_norm.set_ylabel(f'Consumption (normalized)')
            ax_norm.set_ylim(-1400, 2800)
            ax_norm.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure if requested
    out_file = args.out_file
    if out_file is None:
        safe_start = start.isoformat().replace(':', '')
        out_file = f'visualise_norm_{safe_start}_meter{args.meter_id}.png'
    try:
        fig.savefig(out_file, bbox_inches='tight')
        print('Saved figure to', out_file)
    except Exception as e:
        print('Failed to save figure:', e)

    plt.show()


if __name__ == '__main__':
    main()

