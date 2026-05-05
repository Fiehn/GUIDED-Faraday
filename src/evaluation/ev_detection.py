import numpy as np
import pandas as pd


def detect_platos_unified_dP2_std(
    df_sm,
    sm_num,
    start_threshold_min=3000,
    start_threshold_max=7500,
    end_drop_threshold=-3000,
    fluctuation_limit=1500,
    min_duration_hours=2,
    interval_minutes=15,
):
    """
    Detect EV charging events (platos) in smart meter data using unified dP2 and std-based criteria.
    
    Parameters
    ----------
    df_sm : pd.DataFrame
        DataFrame containing smart meter power data indexed by datetime
    sm_num : str or int
        Smart meter number identifier
    start_threshold_min : float, default=3000
        Minimum power jump (dP2) to trigger start detection (Watts)
    start_threshold_max : float, default=7500
        Maximum power jump (dP2) to trigger start detection (Watts)
    end_drop_threshold : float, default=-3000
        Power drop threshold to trigger end detection (Watts)
    fluctuation_limit : float, default=1500
        Maximum acceptable power fluctuation during plateau
    min_duration_hours : float, default=2
        Minimum duration for a valid EV charging event (hours)
    interval_minutes : int, default=15
        Time interval between measurements (minutes)
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing detected EV charging events with columns:
        - plato_datetime: Datetime of detection
        - start_time: Event start time
        - end_time: Event end time
        - duration_hours: Event duration in hours
        - phase_type: Type of event ('1ph', '2ph', or '3ph')
        - active_phases: List of active phases
        - inactive_phases: List of inactive phases
        - start_dP_l{1,2,3}: Power jump on each phase
        - end_dP_l{1,2,3}: Power drop on each phase
    """
    
    min_duration_intervals = int((min_duration_hours * 60) / interval_minutes)

    l1_col = f"active_power_p14_l1_{sm_num}"
    l2_col = f"active_power_p14_l2_{sm_num}"
    l3_col = f"active_power_p14_l3_{sm_num}"
    required_cols = [l1_col, l2_col, l3_col]

    # numeric + arrays
    working_df = df_sm[required_cols].apply(pd.to_numeric, errors="coerce")
    phs_power_array = working_df.to_numpy(dtype=float)  # shape [n, 3]
    n_timesteps = phs_power_array.shape[0]

    # power difference with lag 2 for better stability (less sensitivity to slow starts and ends)
    dP2 = np.full_like(phs_power_array, np.nan) 
    dP2[2:] = phs_power_array[2:] - phs_power_array[:-2]

    # candidate starts: positive jump in [min,max] on at least one phase
    start_active_mask2 = (dP2 >= start_threshold_min) & (dP2 <= start_threshold_max)
    candidate_positions = np.flatnonzero(np.any(start_active_mask2, axis=1))

    # end condition building block
    end_drop_mask2 = dP2 <= end_drop_threshold

    phase_names = np.array(["l1", "l2", "l3"], dtype=object)
    events = []
    last_end_pos = -1

    for start_pos in candidate_positions:
        if start_pos <= 0 or start_pos <= last_end_pos:
            continue

        active_flags = start_active_mask2[start_pos]
        
        n_active = int(active_flags.sum())

        active_idx = np.flatnonzero(active_flags)
        inactive_idx = np.flatnonzero(~active_flags)

        # classify type from active phases at start
        if n_active == 3:
            phase_type = "3ph"
        elif n_active == 2:
            phase_type = "2ph"
        else:
            phase_type = "1ph"

        # end condition: active phases must drop <= end_drop_threshold, inactive phases must remain "inactive" (abs dP < start_threshold_min)
        end_ok = np.ones(n_timesteps, dtype=bool)
        end_ok &= np.all(end_drop_mask2[:, active_idx], axis=1)
        end_candidates_rel = np.flatnonzero(end_ok[start_pos + 1 :])
        
        if end_candidates_rel.size == 0:  # needed if no end is found after spike
            continue
        
        end_pos = int(end_candidates_rel[0] + start_pos + 1)

        # duration check
        duration_intervals = end_pos - start_pos
        if duration_intervals < min_duration_intervals:
            continue

        plateau_slice = phs_power_array[start_pos:(end_pos-1), :]  # end and one before exclusive
        if plateau_slice.shape[0] == 0:
            continue

        # AFTER (new - relative to baseline noise)
        pre_window = phs_power_array[max(0, start_pos - 8) : start_pos, :]
        std_before = np.std(pre_window, axis=0)
        std_during = np.std(plateau_slice, axis=0)
        
        # guard: if pre_window is too short or flat, fall back to fixed threshold
        std_before = np.where(std_before < 50, 50, std_before)  # minimum 50W noise floor
        
        if np.any(std_during > std_before * 1.5):
            continue
    
        start_dP = dP2[start_pos]
        end_dP = dP2[end_pos-1]

        events.append(
            {
                "plato_datetime": working_df.index[start_pos],
                "start_time": working_df.index[start_pos],
                "end_time": working_df.index[end_pos],
                "duration_hours": (duration_intervals * interval_minutes) / 60.0,
                "phase_type": phase_type,
                "active_phases": phase_names[active_idx].tolist(),
                "inactive_phases": phase_names[inactive_idx].tolist(),
                "start_dP_l1": float(start_dP[0]) if not np.isnan(start_dP[0]) else np.nan,
                "start_dP_l2": float(start_dP[1]) if not np.isnan(start_dP[1]) else np.nan,
                "start_dP_l3": float(start_dP[2]) if not np.isnan(start_dP[2]) else np.nan,
                "end_dP_l1": float(end_dP[0]) if not np.isnan(end_dP[0]) else np.nan,
                "end_dP_l2": float(end_dP[1]) if not np.isnan(end_dP[1]) else np.nan,
                "end_dP_l3": float(end_dP[2]) if not np.isnan(end_dP[2]) else np.nan,
            }
        )

        # prevent overlapping duplicate registrations
        last_end_pos = end_pos

    return pd.DataFrame(events)

