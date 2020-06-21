#!/bin/bash

# python <script_name> <classifier_name> <label_name> <class_weight> <original_data_filepath>
#    classifier_name  = "SVM", "RF"
#    label_name =  "Diagnosis", "Class", "AgeGroup"
#    class_weight = "equal", "balanced", float
#    original_data_filepath: Optional. It is only for "Diagnosis" Task
#       e.g., 

TASK=$1 # "Class"
echo -e "Task: detect ${TASK}"

python Bo.py SVM ${TASK} equal  
#python Bo.py SVM ${TASK} balanced  


#python Bo.py RF ${TASK} equal  
#python Bo.py RF ${TASK} balanced  

