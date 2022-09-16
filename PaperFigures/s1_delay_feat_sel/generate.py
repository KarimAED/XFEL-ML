from oldMode2017.setup import get_data
from utility.pipelines.ann import ann_pipeline
from utility.pipelines import feature

DATA = get_data()

STR_DATA = {
    "feat_name": "Delays"
}

FEAT_RANK = feature.feature_ranking_pipeline(DATA, STR_DATA, ann_pipeline)
FEAT_RANK_AND_SELECTED = feature.selected_refit_pipeline(
    DATA,
    STR_DATA,
    ann_pipeline,
    FEAT_RANK,
)
FEAT_RANK_AND_SELECTED.to_csv("PaperFigures/s1_delay_feat_sel/feat_sel_data.csv")
