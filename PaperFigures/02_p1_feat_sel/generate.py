from newMode2021.setup import get_data_p1
from utility.pipelines.ann import ann_pipeline
from utility.pipelines import feature

U2_DATASET = "u2_273_37026_events.pkl"

DATA = get_data_p1(U2_DATASET)

STR_DATA = {
    "feat_name": "vls_com_pump"
}

FEAT_RANK = feature.feature_ranking_pipeline(DATA, STR_DATA, ann_pipeline)
FEAT_RANK.to_csv("PaperFigures/02_p1_feat_sel/feat_sel_data.csv")


FEAT_RANK_AND_SELECTED = feature.selected_refit_pipeline(
    DATA,
    STR_DATA,
    ann_pipeline,
    FEAT_RANK.iloc[::5,:]
)
FEAT_RANK_AND_SELECTED.to_csv("PaperFigures/02_p1_feat_sel/feat_sel_data.csv")
