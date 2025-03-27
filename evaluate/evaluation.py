from typing import List
import numpy as np
import pandas as pd
import motmetrics as mm
import os
import json

from yacs.config import CfgNode
from tools.metrics import iou
from tools.conversion import to_frame_list, load_motchallenge_format
from evaluate.experimental import greedy_matching


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates of the same id in the same frame (should never happen)"""
    return df.drop_duplicates(subset=["frame", "track_id"], keep="first")

def remove_single_cam_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Remove tracks from df that only appear on one camera ('cam' column needed)"""
    subdf = df[["cam", "track_id"]].drop_duplicates()
    track_cnt = subdf[["track_id"]].groupby(["track_id"]).size()
    good_ids = track_cnt[track_cnt > 1].index
    return df[df["track_id"].isin(good_ids)]

def load_annots(paths: List[str], nested_dict_list):
    """Load one txt annot for each camera, and return them in a merged dataframe."""
    dicts = [load_motchallenge_format(path) for path in paths]
    dfs = [pd.DataFrame(d) for d in dicts]
    max_frame = max([int(k) for k in nested_dict_list[0].keys()])

    by_frames = []
    for i, df in enumerate(dfs):
        #df["frame"] = df["frame"].apply(lambda x: x + max_frame)
        df["cam"] = i
        df = remove_duplicates(df)
        by_frame = to_frame_list(df, max_frame)
        #by_frameの要素を1つずつby_framesに追加
        for frame in by_frame:
            by_frames.append(frame)

        """
        folder_path = "/home/fujii/Documents/Projects/vehicle_mtmc-master/vehicle_mtmc/output/bayesian/"
        os.makedirs(folder_path, exist_ok=True)
        file_name = f"by_frame_{i}.txt"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w") as f:
            for frame in by_frame:
                f.write(f"{frame}\n")
        """
    return by_frames

def evaluate_dfs(test_by_frame, pred_by_frame, nested_dict_list, min_iou=0.5, ignore_fp=False):
    """Evaluate MOT (or merged MTMC) predictions against the ground truth annotations."""

    acc = mm.MOTAccumulator(auto_id=True)

    for gt, preds in zip(test_by_frame, pred_by_frame):
        mat_gt = np.array([x[:4] for x in gt])
        mat_pred = np.array([x[:4] for x in preds])
        iou_matrix = mm.distances.iou_matrix(mat_gt, mat_pred, 1 - min_iou)
        n, m = len(gt), len(preds)

        if ignore_fp:
            # remove preds that are unmatched (would be false positives)
            matched_gt, matched_pred = mm.lap.linear_sum_assignment(iou_matrix)
            remain_preds = set(matched_pred)
            remain_pred_idx = [-1] * m
            for i, p in enumerate(remain_preds):
                remain_pred_idx[p] = i
            m = len(remain_preds)

            # now we can create the distance matrix rigged for our matching
            iou_matrix = np.full((n, m), np.nan)
            for i_gt, i_pred in zip(matched_gt, matched_pred):
                iou_matrix[i_gt, remain_pred_idx[i_pred]] = 0.0
        else:
            remain_pred_idx = list(range(m))

        pred_ids = [x[4]
                    for i, x in enumerate(preds) if remain_pred_idx[i] >= 0]
        gt_ids = [x[4] for x in gt]
        acc.update(gt_ids, pred_ids, iou_matrix)

    """
    df_reset = acc.mot_events.reset_index()
    data_dict = df_reset.to_dict(orient='records')
    folder_path = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/output/"
    file_name = "acc.json"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "w") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)
    """

    folder_path = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/output/"
    os.makedirs(folder_path, exist_ok=True)
    max_frame = 0
    total_dict = {}
    for i, nested_dict in enumerate(nested_dict_list):
        for frame_key in nested_dict.keys():
            if frame_key not in total_dict.keys():
                total_dict[frame_key] = {}
            frame_key_int = int(frame_key) + max_frame -2
            if frame_key_int in acc.mot_events.index.get_level_values('FrameId'):
                for id_key in nested_dict[frame_key].keys():
                    id_key_int = int(id_key)
                    # "FrameID"がframe_keyに一致し、"Type"が"MATCH"である行をフィルタリング
                    match_events = acc.mot_events.loc[(frame_key_int, slice(None)), :]
                    match_events = match_events[(match_events['Type'] == 'MATCH') & (match_events['OId'] == id_key_int)]
                    # 対応する行がある場合、nested_dict[frame_key][id_key]に1を加算
                    if not match_events.empty:
                        nested_dict[frame_key][id_key] += 1
                    if id_key not in total_dict[frame_key].keys():
                        total_dict[frame_key][id_key] = 0
                    total_dict[frame_key][id_key] += nested_dict[frame_key][id_key]
        max_frame += max([int(k) for k in nested_dict.keys()])

        """
        file_name_gt = f"output_gt_{i}.txt"
        file_path = os.path.join(folder_path, file_name_gt)
        with open(file_path, "w") as f:
            for frame_key in total_dict.keys():
                for id_key in total_dict[frame_key].keys():
                    f.write(f"{frame_key},{id_key},{total_dict[frame_key][id_key]}\n")

        file_name = f"output_{i}.txt"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w") as f:
            for frame_key in nested_dict.keys():
                for id_key in nested_dict[frame_key].keys():
                    f.write(f"{frame_key},{id_key},{nested_dict[frame_key][id_key]}\n")

        print(frame_int_list)
        """

    """
    #nested_dict_listの要素を1つずつファイルにに出力。Frame，ID，countの3列。フォルダ、ファイルの指定方法も記述する(名前は適当)
    folder_path = "/home/fujii/Documents/Projects/vehicle_mtmc-master/vehicle_mtmc/output/bayesian/"
    os.makedirs(folder_path, exist_ok=True)
    for i, nested_dict in enumerate(nested_dict_list):
        file_name = f"output_{i}.txt"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w") as f:
            for frame_key in nested_dict.keys():
                for id_key in nested_dict[frame_key].keys():
                    f.write(f"{frame_key},{id_key},{nested_dict[frame_key][id_key]}\n")
    """

    #total_dict["frame"][id]が1以上である要素の数をカウント
    total_count = 0
    total_gt_count = 0

    for frame_key in total_dict.keys():
        for id_key in total_dict[frame_key].keys():
            if total_dict[frame_key][id_key] > 0:
                total_count += 1
            total_gt_count += 1

    #全体の評価結果を算出
    print(f"total_count: {total_count}")
    print(f"total_gt_count: {total_gt_count}")
    score = total_count / total_gt_count
    return score


def single_load_annots(paths: List[str]) -> pd.DataFrame:
    """Load one txt annot for each camera, and return them in a merged dataframe."""
    dicts = [load_motchallenge_format(path) for path in paths]
    dfs = [pd.DataFrame(d) for d in dicts]
    max_frame = 0
    df_new = pd.DataFrame()
    for i, df in enumerate(dfs):
        df["frame"] = df["frame"].apply(lambda x: x + max_frame)
        df["cam"] = i
        max_frame = max(df["frame"])
        df_new = pd.concat([df_new, df])
    #df = pd.concat(dfs)
    return remove_duplicates(df_new)


def single_evaluate_dfs(test_df: pd.DataFrame, pred_df: pd.DataFrame, dict_list, min_iou=0.5, ignore_fp=False):
    """Evaluate MOT (or merged MTMC) predictions against the ground truth annotations."""

    acc = mm.MOTAccumulator(auto_id=True)

    total_frames = max(max(pred_df["frame"]), max(test_df["frame"])) + 1
    test_by_frame = to_frame_list(test_df, total_frames)
    pred_by_frame = to_frame_list(pred_df, total_frames)

    for gt, preds in zip(test_by_frame, pred_by_frame):
        mat_gt = np.array([x[:4] for x in gt])
        mat_pred = np.array([x[:4] for x in preds])
        iou_matrix = mm.distances.iou_matrix(mat_gt, mat_pred, 1 - min_iou)
        n, m = len(gt), len(preds)

        if ignore_fp:
            # remove preds that are unmatched (would be false positives)
            matched_gt, matched_pred = mm.lap.linear_sum_assignment(iou_matrix)
            remain_preds = set(matched_pred)
            remain_pred_idx = [-1] * m
            for i, p in enumerate(remain_preds):
                remain_pred_idx[p] = i
            m = len(remain_preds)

            # now we can create the distance matrix rigged for our matching
            iou_matrix = np.full((n, m), np.nan)
            for i_gt, i_pred in zip(matched_gt, matched_pred):
                iou_matrix[i_gt, remain_pred_idx[i_pred]] = 0.0
        else:
            remain_pred_idx = list(range(m))

        pred_ids = [x[4]
                    for i, x in enumerate(preds) if remain_pred_idx[i] >= 0]
        gt_ids = [x[4] for x in gt]
        acc.update(gt_ids, pred_ids, iou_matrix)

    #max_frame = 0
    #frame_int_list = []
    for frame_key in dict_list.keys():
        frame_key_int = int(frame_key) -1
        if frame_key_int in acc.mot_events.index.get_level_values('FrameId'):
            for id_key in dict_list[frame_key].keys():
                id_key_int = int(id_key)
                # "FrameID"がframe_keyに一致し、"Type"が"MATCH"である行をフィルタリング
                match_events = acc.mot_events.loc[(frame_key_int, slice(None)), :]
                match_events = match_events[(match_events['Type'] == 'MATCH') & (match_events['OId'] == id_key_int)]
                # 対応する行がある場合、nested_dict[frame_key][id_key]に1を加算
                if not match_events.empty:
                    dict_list[frame_key][id_key] += 1

    #total_dict["frame"][id]が1以上である要素の数をカウント
    single_total_count = 0
    single_total_gt_count = 0

    for frame_key in dict_list.keys():
        for id_key in dict_list[frame_key].keys():
            if dict_list[frame_key][id_key] > 0:
                single_total_count += 1
            single_total_gt_count += 1

    #全体の評価結果を算出
    print(f"single_total_count: {single_total_count}")
    print(f"single_total_gt_count: {single_total_gt_count}")
    single_score = single_total_count / single_total_gt_count

    return single_score
