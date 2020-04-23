#!/bin/bash

# python <script_name> <classifier_name> <label_name> <class_weight> <original_data_filepath>
#    classifier_name  = "SVM", "RF"
#    label_name =  "Diagnosis", "Class"
#    class_weight = "equal", "balanced", float
#    original_data_filepath: Optional. It is only for "Diagnosis" Task
#       e.g., 

DATASET_FILEPATH=Sample_Master_Source_and_Platform_batch_removed_Extra_Sample_Info.exported_for_AI.txt

python Bo.py SVM Diagnosis equal ${DATASET_FILEPATH} 
python Bo.py SVM Diagnosis balanced ${DATASET_FILEPATH} 


python Bo.py RF Diagnosis equal ${DATASET_FILEPATH} 
python Bo.py RF Diagnosis balanced ${DATASET_FILEPATH} 

