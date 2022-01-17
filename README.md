# XFEL-ML

Efficient prediction of double-pulse X-ray free-electron laser with machine learning


## Motivation

X-Ray Free Electron Lasers are often used for imaging (spectroscopy) in fields like biology,
chemistry, and material science. The short duration of their coherent burst
(on the scale of ~100as), make them ideal for high-energy imaging without destroying the imaged structure.
Most recently, XFELs have been pushed to higher and higher operating rates, from 10s of pulses per second,
into the regime of 1000s of pulses. The issue with this is the relative instability in XFEL pulse properties.
Both between individual shots, and slowly over time the pulse properties such as mean energy vary. Before, this would
have been addressed by fully characterizing the pulse through imaging, but this approach is prohibitively slow and
creates huge amounts of data that needs to be processed quickly. However,
technical constraints often prohibit this also.
As a solution, machine learning can be used to make prediction of the pulse from other data on each event, that can be
collected at a fraction of the time and data cost.

## Method and results

Through utilizing artificial neural networks, gradient boosting, feature selection via permutation feature importance,
and further analysis, we have not only extended this research to a new mode of XFEL operation (which has replaced the 
original mode on which machine learning was demonstrated). We have also reduced the number of features required for
accurate predictions down to 10, and increased speed of estimator fitting by a minimum factor of 2. Furthermore,
we have discovered that for dual pulses, the quality of predictions for the second pulse depends substantially on the
number of undulators the second pulse additionally traverses.

## Installation

If possible, using poetry, you can simply get all dependencies by running

```
poetry install
```

Alternatively, you can use pip with the pyproject.toml also.

## Running instructions

The two main experiments as featured in the paper and suppl. are contained in the Folders 
newMode2021 and oldMode2017 respectively. There you will find notebooks running through the 
main analysis steps for each of the datasets.

For regenerating the figures as seen in the paper, the PaperFigures folder contains a
subfolder code, which contains the code to generate most of the figures seen in the paper
(Figure 2 is generated from the notebooks mentioned above).

## Repository overview

For full reference for the interested reader, this is the repository file structure:
```
XFEL-ML
|--- README.md
|--- pyproject.toml
|--- poetry.lock
|--- log_config.yml
|--- newMode2021
|    |
|    |--- newMode2021.ipynb
|    |--- setup.py
|    |--- data
|        |...
|    |--- results
|        |
|        |--- ex_1_pump_pred
|            |...
|        |--- ex_2_pump_probe_corr
|            |...
|        |--- ex_3_undulator_vary
|            |...
|        |--- ex_4_probe_pred
|            |...
|--- doublePulse2017
|    |
|    |--- doublePulse2017.ipynb
|    |--- setup.py
|    |--- data
|        |...
|    |--- results
|        |
|        |--- ex_1_ann_feat
|            |...
|        |--- ex_2_gb_perf
|            |...
|--- PaperFigures
|    |
|    |--- bar plots
|        |...
|    |--- code
|        |
|        |--- epoch_convergence.py
|        |--- figure3.py
|        |--- figure4.py
|        |--- sample_convergence.py
|        |--- final.mplstyle
|    |--- Figure Data
|        |...
|    |... (individual figure folders + files)
|--- utility
|    |
|    |--- helpers.py
|    |--- split_prep_util.py
|    |--- estimators
|        |
|        |--- grad_boost.py
|        |--- neural_network.py
|    |--- pipelines
|        |
|        |--- ann.py
|        |--- gb.py
|        |--- lin.py
|    |--- plotting
|        |
|        |--- plot_features.py
|        |--- plot_fit.py
|        |--- scatter_diff.py
|        |--- styling.mplstyle
```

## More resources

For a blog style article on this subject, visit karimaed.github.io/XFEL-ML/

Paper reference will be added here once it is on the archive.


## About

I, Karim Alaa El-Din, am the sole author of this repository. It is the code for my
bachelor's project turned part-time research studentship, which is currently being
written up as a peer-reviewed paper. Other people who contributed to the project are:
1. Rick Mukherjee, My supervisor
2. Florian Mintert, My assessor
3. Oliver Alexander, with experimental insights
4. John Marangos, with continued input and support
5. Leszek Frasinski, with continued input and support