This repository contains code for reproducing analysis in the submitted paper 

The natural-audio datasets analyzed in this study are derived from copyrighted third-party material and cannot be redistributed. The strings and drums samples were obtained from the Splice sample library under its standard license; the vocal recordings are from the commercially released catalog of a single pop artist, obtained legally, with vocal stems isolated using Demucs. We release the activation-extraction and analysis pipeline, together with the derived per-neuron activation features and feature labels required to reproduce all figures and tables. The underlying audio is not distributed; researchers may reproduce the extraction on their own legally obtained copies of the source material.

## Reproducing Results from Distributed Data  

Provided in `results/` as the derived per-neuron activation features and feature labels required to reproduce all figures and tables

1. `generate_csvs.sh` in `permutations/` with generate all the tables needed for analysis

2. Code in `analyse-within-layer-correlations/` provides analysis for layer depth and natural vs synthetic

3. Code in `analyse-cross-layer-correlations/` provides analysis for clustering. Change the for loop at the top to do across k values if you want comparisons

This will print out results in a structured manner, as well as generate any plots. 

## The Original Pipeline

We provide the pipeline for replicating results though are unable to distribute source audio for licencing reasons

1. Get Models  

You will need to request access to get the trained models from 

[HERE](https://drive.google.com/drive/folders/1UCSV50c-Z2EiYY-j5_Ir06yHNZk1D8lU?usp=sharing)

This will be 3 `.ckpt` and `.gin` combinations for the models


2. Correlations and Clusters 

`get_correlations_clusters.py` will run the analysis that gets the activations from the models and correlates with audio features. It will also do the cross-layer correlations. 

This will make a `results` folder that will need to be used for the further analysis.

3. Permutation Tests

`run_baselines.sh` will generate the permutation baseline. Warning, this will take a long time (500 permutations per combination!)




