import pandas as pd
import numpy as np
import pytest

from utility.split_prep_util import norm, train_test_norm

class TestNorm:

    @pytest.mark.parametrize("data,ref,label", [((100, 1), pd.DataFrame(columns=["a"]), "label"),
                                                (np.random.random((100, 2, 3)), pd.DataFrame(columns=["a"]), "label"),
                                                (np.array(["a", "b"]), pd.DataFrame(columns=["a"]), "label"),
                                                (np.random.random((100, 1)), "wrong_type", "label"),
                                                (np.random.random((100, 1)), pd.DataFrame(columns=["a"]), 123),
                                                (np.random.random((100, 2)), pd.DataFrame(columns=["a"]), "label"),
                                                (np.zeros((100, 1)), pd.DataFrame(columns=["a"]), "label")])
    def test_should_fail(self, data, ref, label):
        with pytest.raises(AssertionError):
            norm(data, ref, label)

    @pytest.mark.parametrize("data,ref,label", [(np.random.random((100, 1)), pd.DataFrame(columns=["test"]), "label")])
    def test_should_work(self, data, ref, label):
        norm_data, norm_ref = norm(data, ref, label)

        assert norm_ref.values.shape == (2, data.shape[1])
        assert np.all(np.mean(data, axis=0) == norm_ref.loc[label+"_mean", :])
        assert np.all(np.std(data, axis=0) == norm_ref.loc[label+"_std", :])
        np.testing.assert_almost_equal(
            (norm_data*norm_ref.loc[label+"_std", :].values)+norm_ref.loc[label+"_mean", :].values, data)


class TestTrainTestNorm:

    @pytest.mark.parametrize("inp_df,out_df,split,normalize", [])
    def test_should_fail(self, inp_df, out_df, split, normalize):
        with pytest.raises(AssertionError):
            train_test_norm(inp_df, out_df, split, normalize=normalize)

    @pytest.mark.parametrize("inp_df,out_df,split,normalize", [])
    def test_should_work(self, inp_df, out_df, split, normalize):
        x_tr, x_te, y_tr, y_te, i_ref, o_ref = train_test_norm(inp_df, out_df, split, normalize=normalize)
