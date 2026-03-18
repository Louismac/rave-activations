This repository contains code for reproducing analysis in the submitted paper ```Where Pitch and Tempo Live: Interpreting Feature Encoding in Neural Audio Synthesis```

1. Get Data 

You will need to request access to get the trained models and datasets from 

[HERE](https://drive.google.com/drive/folders/1UCSV50c-Z2EiYY-j5_Ir06yHNZk1D8lU?usp=sharing)

This will be 3 `.ckpt` and `.gin` combinations for the models and 4 `.pkl` files for the datasets 


2. Correlations and Clusters 

`get_correlations_clusters.py` will run the analysis that gets the activations from the models and correlates with audio features. It will also do the cross-layer correlations. 

This will make a `results` folder that will need to be used for the further analysis.

3. Permutation Tests

`run_permutation_baseline.py` will generate the permutation baseline. Warning, this will take a long time (500 permutations per combination!)

4. Further Analysis 

Code in `analyse-within-layer-correlations/` provides analysis for Study 1

Code in `analyse-cross-layer-correlations/` provides analysis for Study 2. Change the for loop at the top to do across k values if you want comparisons

This will print out results in a structured manner, as well as generate any plots. 
