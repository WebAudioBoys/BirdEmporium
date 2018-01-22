# BirdEmporium
Final project for MIR, NYU MPATE-GE 2623 Fall 2017.
Employs onset detection and MFCC analysis to identify the calls of cardinals, chickadees, goldfinches, orioles, and robins in field recordings.

## Running the classifier:
1. grabFramesBySpecies.py - Selects relevant audio by onset detection, creates buffered audio matrix of selected frames for each species.
1. createFeatures.py - Creates MFCCs for each species, concatenates results into single master matrix of MFCCs.
1. classify.py - Split up test/train features, run random forest algorithm, return classification test results.
1. whatBird.py - Predict the species in a field recording.
