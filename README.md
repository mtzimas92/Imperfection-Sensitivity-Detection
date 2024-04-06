This is a public repository used as supplementary material for the paper "Imperfection Sensitivity Detection in Pultruded Columns using Machine Learning and Synthetic Data" for the Buildings Journal (resubmitted, pending review). 

The repository contains 3 python files: 

1) Buckle.py ABAQUS script: Creates the model used for this paper in ABAQUS FEA. It also solves the buckling problem for this column and extracts some necessary information. It is recommended to read the script and see what information is needed prior to running it, since it can be used as an automation script. 
2) Classification.py script: Classifies the columns created in ABAQUS as imperfection sensitive or not imperfection sensitive. Also creates a labels.txt and features.txt useful for training the ML algorithm.
3) imperfection-tensorflow.py script: Trains and validates the ML algorithm used in the paper. 

The repository also contains a folder with various figures that did not make it into the paper (following the first review). The figures are for comparison for accuracy and loss per epoch and how loss variations may become different when implementing various methods. 

The scripts can be used as is however the user will not be able to get anything useful from Classification.py and imperfection-tensorflow.py. They will need to run a RIKS analysis in ABAQUS following the Buckle analysis. The script will be provided after contacting the authors.
