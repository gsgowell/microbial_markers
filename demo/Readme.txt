Neural Network-based classifier

This is an example that using the classifier for classification. Since D2 does not have negative samples, 30% negative samples of D1 were randomly selected as the negative samples of D2, and the remain samples of D1 (i.e., D1-) with the biomarkers were used for training. Meanwhile, the samples of D2 and the added negative samples with the biomarkers were used for testing. 'RF_MarkerSpecies_D1_minus_trained.h5' and 'RF_MarkerGenes_D1_minus_trained.h5'are our trained models. Users can also train the models themselves.   

Requirements: Python environment.

Useage: once Python is installed, 'Main_for_test.py' can be implemented.