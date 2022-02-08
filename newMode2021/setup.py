import numpy as np
import pandas as pd
import scipy.stats as sps

from collections import defaultdict
from utility.split_prep_util import train_test_norm
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def get_data_p1(
    fname, split=0.15, include_probe=True, filterByCorr=False, filter_cols=[]
):
    pathname = f"newMode2021/data/{fname}"

    inp_df = pd.DataFrame()
    out_df = pd.DataFrame()
    all_df = pd.read_pickle(pathname)

    for name, column in all_df.iteritems():
        cols = ["epics_", "ebeam_", "gmd_", "xgmd_"]
        if include_probe:
            cols.append("probe")
        inp = [i in name for i in cols]
        if any(inp):
            inp_df[name] = column
        if "pump" in name:
            out_df[name] = column

    # Load pump probe pulse data
    pp_inp = inp_df
    pp_out = out_df

    print("Filtering output columns...")
    # select only delay columns
    pump_out = pp_out.loc[:, ["vls_com_pump"]].copy()

    print(pump_out.shape[1], "columns left.")

    print("Filtering events...")
    # get prepared mask
    if include_probe:
        pump_nan = (
            pump_out["vls_com_pump"].notna().values
            & pp_inp["vls_com_probe"].notna().values
            & pp_inp["vls_width_probe"].notna().values
        )
    else:
        pump_nan = pump_out["vls_com_pump"].notna().values
    pump_mask = pump_nan.astype(np.bool)  # Also mask NaN values

    # apply masking of events
    pump_inp = pp_inp.loc[pump_mask].copy()
    pump_out = pp_out.loc[pump_mask, "vls_com_pump"]  # only select delay copy

    print(pump_inp.shape[0], "events left.")
    print("Filtering input columns...")
    # Filter input features by variance
    var_thresh = 10
    feat_columns = [c for c in pump_inp if len(np.unique(pump_inp[c])) > var_thresh]
    pump_inp = pump_inp[feat_columns]
    if filter_cols:
        pump_inp = pump_inp[filter_cols]
    if filterByCorr:
        corr = spearmanr(pump_inp.values).correlation

        # Ensure the correlation matrix is symmetric
        corr = np.abs((corr + corr.T) / 2)
        np.fill_diagonal(corr, 1)

        # We convert the correlation matrix to a distance matrix before performing
        # hierarchical clustering using Ward's linkage.
        distance_matrix = 1 - np.abs(corr)
        dist_linkage = hierarchy.ward(squareform(distance_matrix))
        dendro = hierarchy.dendrogram(
            dist_linkage,
            labels=[i for i in range(len(pump_inp.columns))],
            leaf_rotation=90,
        )
        dendro_idx = np.arange(0, len(dendro["ivl"]))
        cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        pump_inp = pump_inp[pump_inp.columns[selected_features]]
    print(pump_inp.shape[1], "columns left.")
    print("Filtering MAD & Energy...")
    # Get mean absolute deviation of outputs
    mad_delays = abs(
        (pump_out.values - np.median(pump_out.values))
        / sps.median_abs_deviation(pump_out.values)
    )

    mad_mask = mad_delays < 4

    if include_probe:
        mad_2_delays = abs(
            (
                pump_inp["vls_com_probe"].values
                - np.median(pump_inp["vls_com_probe"].values)
            )
            / sps.median_abs_deviation(pump_inp["vls_com_probe"].values)
        )
        mad_2_mask = mad_2_delays < 4
        mad_mask = mad_mask & mad_2_mask

    arg_mask = np.argwhere(mad_mask).flatten()

    # Apply arg_mask
    pump_inp = pump_inp.iloc[arg_mask]
    pump_out = pump_out.iloc[arg_mask]
    print(pump_inp.shape[0], "events left.")
    print("Done.")
    return train_test_norm(pump_inp, pump_out, split=split)


def get_data_p2(fname, split=0.15, include_pump=True, filter_cols=[]):
    pathname = f"newMode2021/data/{fname}"

    inp_df = pd.DataFrame()
    out_df = pd.DataFrame()
    all_df = pd.read_pickle(pathname)

    for name, column in all_df.iteritems():
        cols = ["epics_", "ebeam_", "gmd_", "xgmd_"]
        if include_pump:
            cols.append("pump")
        inp = [i in name for i in cols]
        if any(inp):
            inp_df[name] = column
        if "probe" in name:
            out_df[name] = column

    # Load probe pump pulse data
    pp_inp = inp_df
    pp_out = out_df

    print("Filtering output columns...")
    # select only delay columns
    probe_out = pp_out.loc[:, ["vls_com_probe"]].copy()

    print(probe_out.shape[1], "columns left.")

    print("Filtering events...")
    # get prepared mask
    if include_pump:
        probe_nan = (
            probe_out["vls_com_probe"].notna().values
            & pp_inp["vls_com_pump"].notna().values
            & pp_inp["vls_width_pump"].notna().values
        )
    else:
        probe_nan = probe_out["vls_com_probe"].notna().values
    probe_mask = probe_nan.astype(np.bool)  # Also mask NaN values

    # apply masking of events
    probe_inp = pp_inp.loc[probe_mask].copy()
    probe_out = pp_out.loc[probe_mask, "vls_com_probe"]  # only select delay copy

    print(probe_inp.shape[0], "events left.")
    print("Filtering input columns...")
    # Filter input features by variance
    var_thresh = 10
    feat_columns = [c for c in probe_inp if len(np.unique(probe_inp[c])) > var_thresh]
    probe_inp = probe_inp[feat_columns]

    if filter_cols:
        probe_inp = probe_inp[filter_cols]
    print(probe_inp.shape[1], "columns left.")
    print("Filtering MAD & Energy...")
    # Get mean absolute deviation of outputs
    mad_delays = abs(
        (probe_out.values - np.median(probe_out.values))
        / sps.median_abs_deviation(probe_out.values)
    )

    mad_mask = mad_delays < 4

    if include_pump:
        mad_2_delays = abs(
            (
                probe_inp["vls_com_pump"].values
                - np.median(probe_inp["vls_com_pump"].values)
            )
            / sps.median_abs_deviation(probe_inp["vls_com_pump"].values)
        )
        mad_2_mask = mad_2_delays < 4
        mad_mask = mad_mask & mad_2_mask

    arg_mask = np.argwhere(mad_mask).flatten()

    # Apply arg_mask
    probe_inp = probe_inp.iloc[arg_mask]
    probe_out = probe_out.iloc[arg_mask]
    print(probe_inp.shape[0], "events left.")
    print("Done.")
    return train_test_norm(probe_inp, probe_out, split=split)
