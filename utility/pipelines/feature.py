import time as t
import pandas as pd

from utility.pipelines.ann import ann_pipeline
from utility import helpers


def feature_ranking_pipeline(
    data,
    string_data,
    estimator_pipeline
):
    """Pipeline used to apply feature selection

    Pipeline used to fit an estimator and then perform feature selection,
    ranking features and only using the top 10 features for refit,
    if desired.

    Args:
        data (array[object]): contains x_train, x_test, y_train, y_test,
            input_reference, output_reference, as returned by data scripts
        string_data (dict): contains strings as labels, filenames, etc.
        vmax (int): vmax to use for the colorbar of the 2d-hist
        legend (bool): if a colorbar legend should be displayed
        no_refit (bool): if true, the function terminates after having
            found the key features, otherwise ann is fit again

    Returns:
        object: if no_refit returns features ranked by feature importance,
            else returns refit estimator, top 10 features used in the refit
    """
    # unpack the data
    _, x_test, _, y_test, input_reference, _ = data
    print("Performing original fit...")
    start_t = t.time()
    est, _ = estimator_pipeline(
        data, string_data, save=False, plot=False, verbose=0
    )  # perform initial fitting
    print(f"Finished original fit after {round(t.time()-start_t, 2)}s")
    # set up everything to rank features
    i_ref = input_reference
    scores = []
    permuted_features = []
    permuted_index = []
    print("\nPerforming feature ranking...")
    start_t = t.time()
    # loop over all features one by oneto exclude them
    for i, col in enumerate(i_ref.columns):
        score = 0
        for k in range(5):  # average over 5 permutations
            max_col = len(i_ref.columns)
            print(
                f"feature {i + 1} / {max_col}; permutation {k + 1} / 5",
                end="\r",
            )
            x_te_masked = helpers.permute(x_test, i)
            score += est.evaluate(x_te_masked, y_test, verbose=0)[1] / 5
        permuted_features.append(col)
        permuted_index.append(i)
        # evaluate estimator performance
        # with one feature scrambled (on test set)
        scores.append(score)

    print(f"Finished feature ranking after {round(t.time()-start_t, 2)}s")
    # get data frame ranking all features
    feature_rank = pd.DataFrame(
        {
            "features": permuted_features,
            "mae_score": scores,
            "feat_ind": permuted_index,
        }
    )
    feature_rank.sort_values(
        "mae_score", inplace=True, ascending=False
    )  # sort data frame by feature importance


    return feature_rank


def selected_refit_pipeline(
    data,
    string_data,
    estimator_pipeline,
    feature_rank,
    step_size=1
):

    _, _, _, y_test, input_reference, output_reference = data
    scores = []
    ranking = feature_rank["feat_ind"].values

    print("Starting full refits with top x features...")
    start_t = t.time()
    for feat_ind in range(0, len(ranking), step_size):
        print(f"Full refit with top {feat_ind+1}/{len(ranking)} features.", end="\r")
        data_temp = helpers.top_x_data(data, ranking, feat_ind)
        temp_est, _ = estimator_pipeline(data_temp, string_data, save=False, plot=False, verbose=0)
        # collect scores with top x features included
        if estimator_pipeline is ann_pipeline:
            raw_score = temp_est.evaluate(data_temp[1], y_test)[1]
        else:
            raw_score = helpers.mae(
                temp_est.predict(data_temp[1]),
                y_test
            )
        scores.append(
            raw_score
            * output_reference.loc[
                "test_std",
                string_data["feat_name"]
            ]
        )
    print(f"Finished full refits after {round(t.time()-start_t, 2)}s")
    feature_rank["feat_selected_score"] = scores

    return feature_rank


def final_refit(data, str_data, feat_rank_and_selected, estimator_pipeline, num_feats=10):
    x_train, x_test, y_train, y_test, input_reference, output_reference = data