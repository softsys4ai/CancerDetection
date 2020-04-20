# CancerDetection

Usage of **extrFeat.py**: extract a small number of features by using HSIC Lasso
```
# python <script> <original_data_filepath> <label_name>
#   label_name = "Diagnosis", "Class"
```

Usage of **Bo.py**: apply Bayesian Optimization to tune a classification model
```
python <script_name> <classifier_name> <label_name> <original_data_filepath>
#    classifier_name  = "SVM", "RF"
#    label_name =  "Diagnosis", "Class"
#    original_data_filepath:
#       e.g., Sample_Master_Source_and_Platform_batch_removed_Extra_Sample_Info.exported_for_AI.txt
#
# The script should be in the same directory level as directories where the extracted features files locate
```
Clustering Analysis On Extracted Features: check the notebook **postAnalysis.ipynb**
