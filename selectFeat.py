# coding: utf-8
# Usage:
# python <script> <original_data_filepath> <label_name>

import os
import sys
import numpy as np

from fsUtil import saveDF, saveFeatures, getNumClass, dataProcess, extractFeatures

# for age detection
top_features = {
    "No-Cancer": ["cg05415871", "cg07804728", "cg13202751", "cg22736354", "cg06489728"],
    "Cancer"   : ["cg06358970", "cg15934776", "cg24461669", "cg27133034", "cg05630556"],
    "All"      : ["cg17590135", "cg21051964", "cg15256305", "cg00590036", "cg10584587"],
    }

def main(argv):
    original_data_filepath = argv[0]
    label_name = argv[1]

    df = dataProcess(original_data_filepath, label_name, selectAgeGender=True)
    print("entire dataset is Loaded")

    for group, selected_feat_names in top_features.items():
        columns = df.columns.tolist()[:4]
        print("{} : {}".format(group, columns))
        columns.extend(selected_feat_names)
        print("====extended: {}".format(group, columns))
        selected_df = df[columns]
        fp = "{}_top{}features.csv".format(group, len(selected_feat_names))
        selected_df.to_csv(fp, index=False, header=True)
   
if __name__ == "__main__":
    main(sys.argv[1:])
