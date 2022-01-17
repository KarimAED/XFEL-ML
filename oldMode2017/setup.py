import os
import logging

import numpy as np
import pandas as pd
import scipy.stats as sps

from utility.split_prep_util import train_test_norm
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# set parameters to load double pulse data
source = os.path.join(os.getcwd(), "oldMode2017/data")
files = sorted(os.listdir(source))
split = 0.15

logger = logging.getLogger("old_data")


def format_inp():
    """
    Function to collect all data used for input features and format it properly.

    :return: tuple of 2d-array + 1d-array, corresponding to feature data & names respectively
    """
    logger.info("Getting inputs from %i files." % len(files))
    # init empty placeholders
    ebeam_labels = []
    ebeam_data = []
    epic_labels = []
    epic_data = []
    gmd_labels = []
    gmd_data = []

    # select data and column names from different files for input features
    for f in files:
        if any([i in f for i in ["EBeam", "EPICS", "GMD"]]):  # avoid unnecessary data loading
            data = np.load(os.path.join(source, f))
        if "EBeam" in f:
            if not ebeam_labels:
                ebeam_data = data["EBeamValuesList"]
                ebeam_labels = data["EBeamParameterNames"].tolist()
            else:
                ebeam_data = np.append(ebeam_data, data["EBeamValuesList"], axis=0)
        elif "EPICS" in f:
            if not epic_labels:
                epic_data = data["EPICSValuesList"]
                epic_labels = data["EPICSParameterNames"].tolist()
            else:
                epic_data = np.append(epic_data, data["EPICSValuesList"], axis=0)
        elif "GMD" in f:
            if not gmd_labels:
                gmd_data = data["GMDValuesList"]
                gmd_labels = data["GMDParameterNames"].tolist()
            else:
                gmd_data = np.append(gmd_data, data["GMDValuesList"], axis=0)

    # join both data and labels for input features
    labels = ebeam_labels + epic_labels + gmd_labels
    inp_labels = [str(label, "utf-8") for label in labels]
    temp_data = np.append(ebeam_data, epic_data, axis=1)
    inp_data = np.append(temp_data, gmd_data, axis=1)
    logger.info("input labels: %s" % str(inp_labels))
    logger.info("input shape: %s" % str(inp_data.shape))

    return inp_data, inp_labels


def format_outp():
    """
    Function to collect all data used for output labels and format it properly.

    :return: tuple of 2d-array + 1d-array, corresponding to label data & names respectively
    """
    logger.info("Getting outputs from %i files." % len(files))
    # initialise output features
    delay_data = []
    delay_labels = []
    tof_data = []
    tof_labels = []

    for f in files:
        if "Delay" in f:
            data = np.load(os.path.join(source, f))
            if not delay_labels:
                delay_labels = ["Delays", "DelayMask"]
                delay_data = np.array([data["DelayValuesList"].flatten(), data["DelayValuesListMask"]]).T
            else:
                temp_data = np.array([data["DelayValuesList"].flatten(), data["DelayValuesListMask"]]).T
                delay_data = np.append(delay_data, temp_data, axis=0)

        elif "TOF" in f:
            data = np.load(os.path.join(source, f))
            if not tof_labels:
                tof_labels = ["LowGaussAmp", "LowGaussMean_eV", "LowGaussStd_eV", "HighGaussAmp", "HighGaussMean_eV",
                              "HighGaussStd_eV"] + data["xTOF"].astype("str").tolist()

                fit_info = np.append(data["TOFDoubleFitList"], np.array([data["TOFDoubleFitListMask"]]).T, axis=1)
                tof_data = np.append(fit_info, data["TOFProfileList"], axis=1)

            else:
                fit_info = np.append(data["TOFDoubleFitList"], np.array([data["TOFDoubleFitListMask"]]).T, axis=1)
                temp_data = np.append(fit_info, data["TOFProfileList"], axis=1)
                tof_data = np.append(tof_data, temp_data, axis=0)

    output = np.append(delay_data, tof_data, axis=1)
    output_labels = delay_labels + tof_labels
    logger.info("output labels: %s" % str(output_labels))
    logger.info("output shape: %s" % str(output.shape))

    return output, output_labels


def get_data(filter_by_corr=False, filter_cols=[]):
    """
    Function to collect both input and output data for oldMode2017, and format the data.

    :param filter_by_corr: bool, if highly correlated input features should be removed
    :param filter_cols: array-like of str, manually select only a few columns to be included
    :return: array-like, contains x_train, x_test, y_train, y_test, input_reference, output_reference
    """
    inp_data, inp_labels = format_inp()
    output, output_labels = format_outp()
    double_inp = pd.DataFrame(data=inp_data, columns=inp_labels)
    double_out = pd.DataFrame(data=output, columns=output_labels)

    print("Filtering output columns...")
    # select only delay columns
    delay_out = double_out.loc[:, ["Delays", "DelayMask"]].copy()

    print(delay_out.shape[1], "columns left.")

    print("Filtering events...")
    # get prepared mask
    delay_mask = delay_out["DelayMask"].values
    delays_nan = delay_out["Delays"].notna().values
    delay_mask = delay_mask.astype(np.bool) & delays_nan.astype(np.bool)  # Also mask NaN values
    delay_mask = delay_mask  # create arg_mask to apply to inps and outputs

    # apply masking of events
    delay_inp = double_inp.iloc[delay_mask].copy()
    delay_out = delay_out.loc[delay_mask, "Delays"]  # only select delay copy

    print(delay_inp.shape[0], "events left.")
    print("Filtering MAD & Energy...")
    # Get mean absolute deviation of outputs
    mad_delays = abs((delay_out.values
                      - np.median(delay_out.values)) / sps.median_abs_deviation(delay_out.values))

    # create mad and beam energy mask from relevant arrays
    mad_mask = mad_delays < 4

    emask = (delay_inp["f_63_ENRC"].values > 0.005) & (delay_inp["f_64_ENRC"].values > 0.005)

    arg_mask = np.argwhere(mad_mask & emask).flatten()  # generate yet another arg_mask

    # Apply arg_mask
    delay_inp = delay_inp.iloc[arg_mask]
    delay_out = delay_out.iloc[arg_mask]
    print(delay_inp.shape[0], "events left.")
    print("Filtering input columns...")
    # Filter input features by variance
    var_thresh = 10
    feat_columns = [c for c in delay_inp if len(np.unique(delay_inp[c])) > var_thresh]
    delay_inp = delay_inp[feat_columns]

    if filter_cols:
        delay_inp = delay_inp[filter_cols]

    if filter_by_corr:
        corr = spearmanr(delay_inp.values).correlation

        # Ensure the correlation matrix is symmetric
        corr = np.abs((corr + corr.T) / 2)
        np.fill_diagonal(corr, 1)

        # We convert the correlation matrix to a distance matrix before performing
        # hierarchical clustering using Ward's linkage.
        distance_matrix = 1 - np.abs(corr)
        dist_linkage = hierarchy.ward(squareform(distance_matrix))
        dendro = hierarchy.dendrogram(
            dist_linkage, labels=[i for i in range(len(delay_inp.columns))], leaf_rotation=90
        )
        dendro_idx = np.arange(0, len(dendro["ivl"]))
        cluster_ids = hierarchy.fcluster(dist_linkage, .7, criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        delay_inp = delay_inp[delay_inp.columns[selected_features]]
    print(delay_inp.shape[1], "columns left.")

    print("Done.")
    # Reuse training_test split and normalisation across inputs
    return train_test_norm(delay_inp, delay_out, split)
