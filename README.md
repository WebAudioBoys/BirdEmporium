# BirdEmporium
MIR Final Project by Brad and Dave.

## Running the classifier:
1. grabFramesBySpecies.py - Selects relevant audio by onset detection, creates buffered audio matrix of selected frames for each species.
1. createFeatures.py - Creates MFCCs for each species, concatenates results into single master matrix of MFCCs.
1. classify.py - Split up test/train features, run random forest algorithm, return classification test results.
