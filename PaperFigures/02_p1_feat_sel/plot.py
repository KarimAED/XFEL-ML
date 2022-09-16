import pandas as pd
import matplotlib.pyplot as plt

from utility.plotting import plot_features as plt_feat

DATA = pd.read_csv("PaperFigures/02_p1_feat_sel/feat_sel_data.csv")

plt_feat.plot_both(
    DATA["mae_score"],
    DATA["features"],
    DATA["feat_selected_score"]
)

plt.savefig("PaperFigures/02_p1_feat_sel/feat_sel_plot.pdf")
