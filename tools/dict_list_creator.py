import pandas as pd
import os
from io import StringIO

def create_nested_dict_list(file_path: str):
    df = pd.read_csv(file_path, header=None, names=['frame', 'id'], delimiter=',', skipinitialspace=True)
    combinations = df.drop_duplicates().reset_index(drop=True)
    dict_template = {}
    for _, row in combinations.iterrows():
        frame_str = str(row['frame'])
        id_str = str(row['id'])
        if frame_str not in dict_template:
            dict_template[frame_str] = {}
        dict_template[frame_str][id_str] = 0
    nested_dict_list = [dict_template.copy() for _ in range(4)]
    return nested_dict_list

def create_one_dict_list(file_path: str):
    df = pd.read_csv(file_path, header=None, names=['frame', 'id'], delimiter=',', skipinitialspace=True)
    combinations = df.drop_duplicates().reset_index(drop=True)
    dict_list = {}
    for _, row in combinations.iterrows():
        frame_str = str(row['frame'])
        id_str = str(row['id'])
        if frame_str not in dict_list:
            dict_list[frame_str] = {}
        dict_list[frame_str][id_str] = 0
    return dict_list