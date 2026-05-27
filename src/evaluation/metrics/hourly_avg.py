import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

from src.evaluation.metrics.base.base import SyntheticMetric, _save_standalone_metric_result, _sanitize_numeric_array, _resolve_output_directory
from src.evaluation.metrics.flexibility_distance import FlexibilityDistance

class HourlyProfilesByMonth(SyntheticMetric):
    """
    Analyze hourly consumption profiles by month and compute flexibility distances.
    
    For each month, computes the average hourly consumption profile (24-hour) for both real 
    and synthetic data by averaging all windows within that month. Then computes the 
    Flexibility Distance between the two average profiles (1-to-1 pairing). Visualizes both 
    average profiles with 10th--90th percentile bands derived from all windows.
    
    Returns mean flexibility distance across all months as the metric value.
    
    Args:
        window_size (int): Hours per window (default: 24)
        plot_filename (str): Name of plot file (default: hourly_profiles_by_month.png)
    """
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    timestamps: np.ndarray = None
    output_path: str = None
    
    def __init__(self, window_size: int = 24, plot_filename: str = 'hourly_profiles_by_month.pdf'):
        self.window_size = window_size
        self.plot_filename = plot_filename
    
    @property
    def name(self) -> str:
        return "Hourly Profiles by Month (Flexibility Distance)"
    
    def _uses_phase_first_flatten(self) -> bool:
        """Check if model uses phase-first flattening for multi-phase data."""
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, '_to_phase_first')
    
    def _extract_consumption_feature(self, data: np.ndarray) -> np.ndarray:
        """Extract consumption feature (first feature) from potentially multi-phase flattened data.
        
        Handles:
        - [N, 24] single-feature data
        - [N, 24*F] multi-feature flattened data (phase-first or time-first)
        
        Returns:
            np.ndarray: [N, 24] consumption feature array
        """
        arr = _sanitize_numeric_array(np.asarray(data))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D flattened array [N, D], got {arr.shape}")

        if arr.shape[1] == self.window_size:
            return arr  # Already consumption only

        if arr.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Flattened dimension {arr.shape[1]} is not divisible by window_size {self.window_size}"
            )

        n_features = arr.shape[1] // self.window_size
        if n_features == 1:
            return arr.reshape(-1, self.window_size)

        # Extract first feature (consumption) depending on flatten order
        if self._uses_phase_first_flatten():
            # Phase-first: [f0_t0..f0_tL-1, f1_t0..], so feature 0 is contiguous
            return arr.reshape(-1, n_features, self.window_size)[:, 0, :]
        else:
            # Time-first: [t0_f0, t0_f1, ..., t1_f0, ...]
            return arr.reshape(-1, self.window_size, n_features)[:, :, 0]
    
    def compute(self) -> float:
        """Compute hourly profiles by month and return mean flexibility distance.
        
        For each month:
        1. Average all real windows in that month into one profile
        2. Average all synthetic windows in that month into one profile
        3. Compute FlexibilityDistance between the two average profiles
        4. Visualize with percentile bands from all windows
        """
        try:
            # Validate inputs
            if self.real_data is None or self.synthetic_data is None:
                raise ValueError(f"Data not injected: real_data={self.real_data is not None}, synthetic_data={self.synthetic_data is not None}")
            
            # Extract consumption features
            real_data = self._extract_consumption_feature(self.real_data)
            synthetic_data = self._extract_consumption_feature(self.synthetic_data)
            
            # Equalize lengths between real and synthetic data
            n = min(len(real_data), len(synthetic_data))
            real_data = real_data[:n]
            synthetic_data = synthetic_data[:n]
            
            # Handle timestamps - be robust to None or invalid timestamps
            verbose = getattr(self, 'verbose', False)
            if self.timestamps is None:
                # Fallback: generate dummy timestamps covering full year
                self.timestamps = np.array([datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n)])
            else:
                # Make sure timestamps match data length and are in correct format
                self.timestamps = np.asarray(self.timestamps)[:n]
            
            # Extract month from timestamps using robust parsing
            months = self._extract_months()
            
            # Verify all arrays have matching lengths
            if len(months) != n:
                raise ValueError(f"Array length mismatch: months={len(months)}, data={n}")
            
            # Get unique months - all values should be valid (1-12) now due to improved _extract_months
            unique_months = sorted(np.unique(months))
            unique_months = [m for m in unique_months if 1 <= m <= 12]  # Filter to valid months
            
            if len(unique_months) == 0:
                if verbose:
                    print("[WARN] HourlyProfilesByMonth: No valid months extracted, defaulting all data to January")
                # Should not happen with improved _extract_months, but handle gracefully
                months = np.ones(n, dtype=int)
                unique_months = [1]
            elif verbose:
                print(f"[INFO] HourlyProfilesByMonth: Found data for months: {unique_months}")
            
            # First pass: count samples per month to find minimum
            month_sample_counts = {}
            for month in unique_months:
                month_mask = (months == month)
                real_month = real_data[month_mask]
                synth_month = synthetic_data[month_mask]
                min_count = min(len(real_month), len(synth_month))
                month_sample_counts[month] = min_count
            
            # Find global minimum to equalize across all months
            if month_sample_counts:
                global_min = min(month_sample_counts.values())
            else:
                global_min = 0
            
            if verbose:
                print(f"[INFO] HourlyProfilesByMonth: Equalizing samples across months to minimum={global_min}")
                print(f"[INFO] HourlyProfilesByMonth: Sample counts before equalization: {month_sample_counts}")
            
            flexibility_distances_by_month = {}
            
            for month in unique_months:
                month_mask = (months == month)
                real_month = real_data[month_mask]  # All windows with timestamp in this month
                synth_month = synthetic_data[month_mask]  # All windows with timestamp in this month
                
                if len(real_month) < 1 or len(synth_month) < 1:
                    continue
                
                # Equalize to global minimum
                real_month = real_month[:global_min]
                synth_month = synth_month[:global_min]
                
                # Average the windows into single profiles
                real_avg = np.mean(real_month, axis=0)  # [24]
                synth_avg = np.mean(synth_month, axis=0)  # [24]
                
                # Compute FlexibilityDistance between the two average profiles (1-to-1)
                try:
                    fd = FlexibilityDistance.compute_fd_between_series(
                        real_avg, synth_avg, tau='auto', normalize=True
                    )
                    flexibility_distances_by_month[month] = fd
                except Exception:
                    flexibility_distances_by_month[month] = np.nan
            
            if not flexibility_distances_by_month:
                raise ValueError("No flexibility distances computed")
            
            # Compute mean FD across months
            all_fds = np.array([v for v in flexibility_distances_by_month.values()])
            mean_fd = float(np.nanmean(all_fds))
            
            # Plot (pass global_min for consistent sample counts in visualization)
            self._plot_hourly_profiles(real_data, synthetic_data, months, flexibility_distances_by_month, global_min=global_min)
            
            # Save results
            _save_standalone_metric_result(
                metric_key='HourlyProfilesByMonth',
                payload={
                    'value': mean_fd,
                    'unique_months': list(unique_months),
                    'n_months': len(unique_months),
                    'mean_fd': mean_fd,
                    'fd_by_month': {int(m): float(v) for m, v in flexibility_distances_by_month.items()},
                    'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
                },
                output_file=self.output_path or 'models/eval_results.json'
            )
            
            return mean_fd
        except Exception:
            import traceback
            traceback.print_exc()
            raise  # Re-raise so framework sees the error
    
    
    def _extract_months(self) -> np.ndarray:
        """Extract month (1–12) from timestamps via a single vectorised call.

        Handles both Unix-integer arrays (the normal dataloader path) and
        datetime-object arrays (the None-fallback path in compute()).
        """
        if self.timestamps is None:
            return np.ones(len(self.real_data), dtype=int)
        import pandas as pd
        ts = np.asarray(self.timestamps)
        if ts.dtype.kind in ('i', 'u', 'f'):          # Unix seconds
            dts = pd.to_datetime(ts.astype(np.int64), unit='s', utc=True)
        else:                                          # datetime objects / datetime64
            dts = pd.to_datetime(ts, utc=True)
        return dts.month.to_numpy(dtype=int)
    

    def _plot_hourly_profiles(self, real_data, synthetic_data, months, 
                             flexibility_distances_by_month, global_min=None):
        """Plot hourly profiles by month.
        
        For each month:
        - Compute average profile from all real windows
        - Compute average profile from all synthetic windows
        - Plot both with 10th-90th percentile bands from raw windows
        
        Args:
            global_min: If provided, use this as the sample count for all months (for consistency)
        """
        try:
            output_dir = _resolve_output_directory(self.output_path)
            if not output_dir:
                return
            
            unique_months = sorted(flexibility_distances_by_month.keys())
            num_months = len(unique_months)
            
            fig, axes = plt.subplots(num_months, 1, figsize=(12, 4 * num_months))
            if num_months == 1:
                axes = [axes]
            
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            
            for month_idx, month in enumerate(unique_months):
                ax = axes[month_idx]
                month_mask = (months == month)
                real_m = real_data[month_mask]
                synth_m = synthetic_data[month_mask]
                
                # Equalize sample counts: use global_min if provided, otherwise local min
                if global_min is not None:
                    sample_count = global_min
                else:
                    sample_count = min(len(real_m), len(synth_m))
                real_m = real_m[:sample_count]
                synth_m = synth_m[:sample_count]
                
                # Compute average profiles (1-to-1)
                real_avg = np.mean(real_m, axis=0)  # [24]
                synth_avg = np.mean(synth_m, axis=0)  # [24]
                
                # Compute percentile bands from all windows
                hours_list = np.arange(self.window_size)
                real_p10 = np.percentile(real_m, 10, axis=0)
                real_p90 = np.percentile(real_m, 90, axis=0)
                synth_p10 = np.percentile(synth_m, 10, axis=0)
                synth_p90 = np.percentile(synth_m, 90, axis=0)
                
                # Plot percentile bands first (background)
                ax.fill_between(hours_list, real_p10, real_p90, alpha=0.2, color='#1f77b4', label='Real (10th-90th %ile)')
                ax.fill_between(hours_list, synth_p10, synth_p90, alpha=0.2, color='#ff7f0e', label='Synthetic (10th-90th %ile)')
                
                # Plot average profiles
                ax.plot(hours_list, real_avg, '-o', label='Real (avg)', color='#1f77b4', linewidth=2, markersize=6)
                ax.plot(hours_list, synth_avg, '-o', label='Synthetic (avg)', color='#ff7f0e', linewidth=2, markersize=6)
                
                # Add FD value to title
                fd_value = flexibility_distances_by_month[month]
                month_name = month_names.get(month, f'Month {month}')
                ax.set_title(
                    f'{month_name} - Hourly Consumption Profiles (FD={fd_value:.4f}, n_real={len(real_m)}, n_synthetic={len(synth_m)})',
                    fontsize=12, fontweight='bold'
                )
                ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
                ax.set_ylabel('Consumption (kWh)', fontsize=12, fontweight='bold')
                ax.set_xticks(hours_list)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10, loc='best')
                ax.set_xlim(-0.5, 23.5)
            
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, self.plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', format='pdf')
            plt.close(fig)
        except Exception as e:
            import traceback
            traceback.print_exc()


class HourlyProfilesByDayofWeek(SyntheticMetric):
    """
    Analyze hourly consumption profiles by day of week and compute flexibility distances.
    
    For each day of week (Monday-Sunday), computes the average hourly consumption profile 
    (24-hour) for both real and synthetic data by averaging all windows within that day. 
    Then computes the Flexibility Distance between the two average profiles (1-to-1 pairing). 
    Visualizes both average profiles with 10th--90th percentile bands derived from all windows.
    
    Returns mean flexibility distance across all days of week as the metric value.
    
    Args:
        window_size (int): Hours per window (default: 24)
        plot_filename (str): Name of plot file (default: hourly_profiles_by_dayofweek.pdf)
    """
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    timestamps: np.ndarray = None
    output_path: str = None
    
    def __init__(self, window_size: int = 24, plot_filename: str = 'hourly_profiles_by_dayofweek.pdf'):
        self.window_size = window_size
        self.plot_filename = plot_filename
    
    @property
    def name(self) -> str:
        return "Hourly Profiles by Day of Week (Flexibility Distance)"
    
    def _uses_phase_first_flatten(self) -> bool:
        """Check if model uses phase-first flattening for multi-phase data."""
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, '_to_phase_first')
    
    def _extract_consumption_feature(self, data: np.ndarray) -> np.ndarray:
        """Extract consumption feature (first feature) from potentially multi-phase flattened data.
        
        Handles:
        - [N, 24] single-feature data
        - [N, 24*F] multi-feature flattened data (phase-first or time-first)
        
        Returns:
            np.ndarray: [N, 24] consumption feature array
        """
        arr = _sanitize_numeric_array(np.asarray(data))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D flattened array [N, D], got {arr.shape}")

        if arr.shape[1] == self.window_size:
            return arr  # Already consumption only

        if arr.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Flattened dimension {arr.shape[1]} is not divisible by window_size {self.window_size}"
            )

        n_features = arr.shape[1] // self.window_size
        if n_features == 1:
            return arr.reshape(-1, self.window_size)

        # Extract first feature (consumption) depending on flatten order
        if self._uses_phase_first_flatten():
            # Phase-first: [f0_t0..f0_tL-1, f1_t0..], so feature 0 is contiguous
            return arr.reshape(-1, n_features, self.window_size)[:, 0, :]
        else:
            # Time-first: [t0_f0, t0_f1, ..., t1_f0, ...]
            return arr.reshape(-1, self.window_size, n_features)[:, :, 0]
    
    def compute(self) -> float:
        """Compute hourly profiles by day of week and return mean flexibility distance.
        
        For each day of week (0-6):
        1. Average all real windows on that day into one profile
        2. Average all synthetic windows on that day into one profile
        3. Compute FlexibilityDistance between the two average profiles
        4. Visualize with percentile bands from all windows
        
        Robustly handles different timestamp formats and data sources.
        """
        try:
            # Validate inputs
            if self.real_data is None or self.synthetic_data is None:
                raise ValueError(f"Data not injected: real_data={self.real_data is not None}, synthetic_data={self.synthetic_data is not None}")
            
            # Extract consumption features
            real_data = self._extract_consumption_feature(self.real_data)
            synthetic_data = self._extract_consumption_feature(self.synthetic_data)
            
            # Equalize lengths between real and synthetic data
            n = min(len(real_data), len(synthetic_data))
            real_data = real_data[:n]
            synthetic_data = synthetic_data[:n]
            
            # Handle timestamps - be robust to None or invalid timestamps
            verbose = getattr(self, 'verbose', False)
            if self.timestamps is None:
                if verbose:
                    print("[WARN] HourlyProfilesByDayofWeek: No timestamps available, using fallback (all data treated as Monday)")
                # Fallback: generate dummy timestamps covering full week
                self.timestamps = np.array([datetime(2023, 1, 2) + timedelta(hours=i) for i in range(n)])  # Start on Monday
            else:
                # Make sure timestamps match data length and are in correct format
                self.timestamps = np.asarray(self.timestamps)[:n]
            
            # Extract day of week from timestamps using robust parsing
            days_of_week = self._extract_days_of_week()
            
            # Verify all arrays have matching lengths
            if len(days_of_week) != n:
                raise ValueError(f"Array length mismatch: days_of_week={len(days_of_week)}, data={n}")
            
            # Get unique days - all values should be valid (0-6) now due to improved _extract_days_of_week
            unique_days = sorted(np.unique(days_of_week))
            unique_days = [d for d in unique_days if 0 <= d <= 6]  # Filter to valid weekdays
            
            if len(unique_days) == 0:
                if verbose:
                    print("[WARN] HourlyProfilesByDayofWeek: No valid days extracted, defaulting all data to Monday")
                # Should not happen with improved _extract_days_of_week, but handle gracefully
                days_of_week = np.zeros(n, dtype=int)
                unique_days = [0]
            # First pass: count samples per day to find minimum
            day_sample_counts = {}
            for day in unique_days:
                day_mask = (days_of_week == day)
                real_day = real_data[day_mask]
                synth_day = synthetic_data[day_mask]
                min_count = min(len(real_day), len(synth_day))
                day_sample_counts[day] = min_count
            
            # Find global minimum to equalize across all days
            if day_sample_counts:
                global_min = min(day_sample_counts.values())
            else:
                global_min = 0
            
            if verbose:
                print(f"[INFO] HourlyProfilesByDayofWeek: Equalizing samples across days to minimum={global_min}")
                print(f"[INFO] HourlyProfilesByDayofWeek: Sample counts before equalization: {day_sample_counts}")
            
            flexibility_distances_by_day = {}
            
            for day in unique_days:
                day_mask = (days_of_week == day)
                real_day = real_data[day_mask]  # All windows with timestamp on this day
                synth_day = synthetic_data[day_mask]  # All windows with timestamp on this day
                
                if len(real_day) < 1 or len(synth_day) < 1:
                    continue
                
                # Equalize to global minimum
                real_day = real_day[:global_min]
                synth_day = synth_day[:global_min]
                
                # Average the windows into single profiles
                real_avg = np.mean(real_day, axis=0)  # [24]
                synth_avg = np.mean(synth_day, axis=0)  # [24]
                
                # Compute FlexibilityDistance between the two average profiles (1-to-1)
                try:
                    fd = FlexibilityDistance.compute_fd_between_series(
                        real_avg, synth_avg, tau='auto', normalize=True
                    )
                    flexibility_distances_by_day[day] = fd
                except Exception:
                    flexibility_distances_by_day[day] = np.nan
            
            if not flexibility_distances_by_day:
                raise ValueError("No flexibility distances computed")
            
            # Compute mean FD across days
            all_fds = np.array([v for v in flexibility_distances_by_day.values()])
            mean_fd = float(np.nanmean(all_fds))
            
            # Plot (pass global_min for consistent sample counts in visualization)
            self._plot_hourly_profiles(real_data, synthetic_data, days_of_week, flexibility_distances_by_day, global_min=global_min)
            
            # Save results
            _save_standalone_metric_result(
                metric_key='HourlyProfilesByDayofWeek',
                payload={
                    'value': mean_fd,
                    'unique_days': list(unique_days),
                    'n_days': len(unique_days),
                    'mean_fd': mean_fd,
                    'fd_by_day': {int(d): float(v) for d, v in flexibility_distances_by_day.items()},
                    'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
                },
                output_file=self.output_path or 'models/eval_results.json'
            )
            
            return mean_fd
        except Exception:
            import traceback
            traceback.print_exc()
            raise  # Re-raise so framework sees the error
    
    def _extract_days_of_week(self) -> np.ndarray:
        """Extract day of week (0=Monday, 6=Sunday) from timestamps via a single vectorised call.

        Handles both Unix-integer arrays (the normal dataloader path) and
        datetime-object arrays (the None-fallback path in compute()).
        """
        if self.timestamps is None:
            return np.zeros(len(self.real_data), dtype=int)
        import pandas as pd
        ts = np.asarray(self.timestamps)
        if ts.dtype.kind in ('i', 'u', 'f'):          # Unix seconds
            dts = pd.to_datetime(ts.astype(np.int64), unit='s', utc=True)
        else:                                          # datetime objects / datetime64
            dts = pd.to_datetime(ts, utc=True)
        return dts.day_of_week.to_numpy(dtype=int)
    
    def _plot_hourly_profiles(self, real_data, synthetic_data, days_of_week, 
                             flexibility_distances_by_day, global_min=None):
        """Plot hourly profiles by day of week.
        
        For each day:
        - Compute average profile from all real windows
        - Compute average profile from all synthetic windows
        - Plot both with 10th-90th percentile bands from raw windows
        
        Args:
            global_min: If provided, use this as the sample count for all days (for consistency)
        """
        try:
            output_dir = _resolve_output_directory(self.output_path)
            if not output_dir:
                return
            
            unique_days = sorted(flexibility_distances_by_day.keys())
            num_days = len(unique_days)
            
            fig, axes = plt.subplots(num_days, 1, figsize=(12, 4 * num_days))
            if num_days == 1:
                axes = [axes]
            
            day_names = {
                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                4: 'Friday', 5: 'Saturday', 6: 'Sunday'
            }
            
            for day_idx, day in enumerate(unique_days):
                ax = axes[day_idx]
                day_mask = (days_of_week == day)
                real_d = real_data[day_mask]
                synth_d = synthetic_data[day_mask]
                
                # Equalize sample counts: use global_min if provided, otherwise local min
                if global_min is not None:
                    sample_count = global_min
                else:
                    sample_count = min(len(real_d), len(synth_d))
                real_d = real_d[:sample_count]
                synth_d = synth_d[:sample_count]
                
                # Compute average profiles (1-to-1)
                real_avg = np.mean(real_d, axis=0)  # [24]
                synth_avg = np.mean(synth_d, axis=0)  # [24]
                
                # Compute percentile bands from all windows
                hours_list = np.arange(self.window_size)
                real_p10 = np.percentile(real_d, 10, axis=0)
                real_p90 = np.percentile(real_d, 90, axis=0)
                synth_p10 = np.percentile(synth_d, 10, axis=0)
                synth_p90 = np.percentile(synth_d, 90, axis=0)
                
                # Plot percentile bands first (background)
                ax.fill_between(hours_list, real_p10, real_p90, alpha=0.2, color='#1f77b4', label='Real (10th-90th %ile)')
                ax.fill_between(hours_list, synth_p10, synth_p90, alpha=0.2, color='#ff7f0e', label='Synthetic (10th-90th %ile)')
                
                # Plot average profiles
                ax.plot(hours_list, real_avg, '-o', label='Real (avg)', color='#1f77b4', linewidth=2, markersize=6)
                ax.plot(hours_list, synth_avg, '-o', label='Synthetic (avg)', color='#ff7f0e', linewidth=2, markersize=6)
                
                # Add FD value to title
                fd_value = flexibility_distances_by_day[day]
                day_name = day_names.get(day, f'Day {day}')
                ax.set_title(
                    f'{day_name} - Hourly Consumption Profiles (FD={fd_value:.4f}, n_real={len(real_d)}, n_synthetic={len(synth_d)})',
                    fontsize=12, fontweight='bold'
                )
                ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
                ax.set_ylabel('Consumption (kWh)', fontsize=12, fontweight='bold')
                ax.set_xticks(hours_list)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10, loc='best')
                ax.set_xlim(-0.5, 23.5)
            
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, self.plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', format='pdf')
            plt.close(fig)
        except Exception as e:
            import traceback
            traceback.print_exc()