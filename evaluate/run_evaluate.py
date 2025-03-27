import os
import sys
import argparse
from pprint import pprint
from yacs.config import CfgNode
import motmetrics as mm

from config.verify_config import check_eval_config
from config.config_tools import expand_relative_paths
from config.defaults import get_cfg_defaults
from tools.conversion import load_motchallenge_format, load_csv_format
from tools import log
from tools.dict_list_creator import create_nested_dict_list, create_one_dict_list
from evaluate import evaluation


def run_evaluation(cfg: CfgNode):
    """Evaluate mot or mtmc results defined by a config."""

    if not check_eval_config(cfg):
        return None

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    if len(cfg.EVAL.GROUND_TRUTHS) != len(cfg.EVAL.PREDICTIONS):
        log.error("EVAL: lengths of GROUND_TRUTHS and PREDICTIONS do not match.")
        return None

    #nested_dict_list = create_nested_dict_list(cfg.EVAL.GROUND_TRUTH)
    path = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/gt.txt"
    nested_dict_list = create_nested_dict_list(path)

    ground_truths_empty = all(os.path.getsize(gt) == 0 for gt in cfg.EVAL.GROUND_TRUTHS)
    predictions_empty = all(os.path.getsize(pred) == 0 for pred in cfg.EVAL.PREDICTIONS)
    if ground_truths_empty or predictions_empty:
        score = 0
        return score
    by_frames_pred = evaluation.load_annots(cfg.EVAL.PREDICTIONS, nested_dict_list)
    #if cfg.EVAL.DROP_SINGLE_CAM and len(cfg.EVAL.PREDICTIONS) > 1:
    #    pred_df = evaluation.remove_single_cam_tracks(pred_df)
    by_frames_test = evaluation.load_annots(cfg.EVAL.GROUND_TRUTHS, nested_dict_list)

    score = evaluation.evaluate_dfs(
        by_frames_test, by_frames_pred, nested_dict_list, min_iou=cfg.EVAL.MIN_IOU, ignore_fp=cfg.EVAL.IGNORE_FP)

    # output summary
    log.info("Evaluation results:\n" + str(score) + "\n")
    with open(os.path.join(cfg.OUTPUT_DIR, "evaluation.txt"), "w") as f:
        f.write(str(score))
        f.write("\n")
    return score


def single_run_evaluation(cfg: CfgNode):
    """Evaluate mot or mtmc results defined by a config."""

    if not check_eval_config(cfg):
        return None

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    if len(cfg.EVAL.GROUND_TRUTHS) != len(cfg.EVAL.PREDICTIONS):
        log.error("EVAL: lengths of GROUND_TRUTHS and PREDICTIONS do not match.")
        return None

    #GTが空かどうか判定 or Predが空　→　score＝0
    ground_truths_empty = all(os.path.getsize(gt) == 0 for gt in cfg.EVAL.GROUND_TRUTHS)
    predictions_empty = all(os.path.getsize(pred) == 0 for pred in cfg.EVAL.PREDICTIONS)

    if ground_truths_empty or predictions_empty:
        single_score = 0
        return single_score

    pred_df = evaluation.single_load_annots(cfg.EVAL.PREDICTIONS)
    if cfg.EVAL.DROP_SINGLE_CAM and len(cfg.EVAL.PREDICTIONS) > 1:
        pred_df = evaluation.remove_single_cam_tracks(pred_df)
    test_df = evaluation.single_load_annots(cfg.EVAL.GROUND_TRUTHS)

    path = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/gt.txt"
    dict_list = create_one_dict_list(path)

    single_score = evaluation.single_evaluate_dfs(
        test_df, pred_df, dict_list, min_iou=cfg.EVAL.MIN_IOU, ignore_fp=cfg.EVAL.IGNORE_FP)

    # output summary
    log.info("Evaluation results:\n" + str(single_score) + "\n")
    return single_score


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="config yaml file")
    # parser.add_argument("--log_level", default="info", help="logging level")
    # parser.add_argument("--log_filename", default="log.txt",
    #                     help="log file under output dir")
    # parser.add_argument("--no_log_stdout", action="store_true",
    #    help="do not log to stdout")
    parser.add_argument("--print_metric_info", action="store_true",
                        help="Print description of computed metrics, then exit.")
    parser.add_argument("--experimental_solver", action="store_true",
                        help="use experimental implementation instead of the motmetrics package")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args("Evaluate MOT or MTMC results.")

    """
    if args.print_metric_info:
        metric_info = {
            "IDF1": ("IDF1 score", "idf1"),
            "IDP": ("ID Precision", "idp"),
            "IDR": ("ID Recall", "idr"),
            "Rcll": ("Recall", "recall"),
            "Prcn": ("Precision", "precision"),
            "GT": ("num_unique (number of ground truth tracks)", "num_unique_objects"),
            "MT": ("Mostly tracked (found in >=80%)", "mostly_tracked"),
            "PT": ("Partially tracked (found in <80%, >=20%)", "partially_tracked"),
            "ML": ("Mostly lost (found in <20%)", "mostly_lost"),
            "FP": ("False positives", None),
            "FN": ("False negatives", None),
            "IDs": ("ID switches", "num_switches"),
            "FM": ("Fragmentations", "num_fragmentations"),
            "MOTA": ("Mean Object Tracking Accuracy", "mota"),
            "MOTP": ("Mean Object Tracking Precision", "motp"),
            "IDt": ("num_transfer", "num_transfer"),
            "IDa": ("num_ascend", "num_ascend"),
            "IDm": ("num_migrate", "num_migrate"),
            "idfp": ("false positives after id matching", "idfp"),
            "idfn": ("false negative after id matching", "idfn"),
            "idtp": ("true positives after id matching", "idtp"),
        }
        infos = []
        mh = mm.metrics.create()
        for k, (short, keyword) in metric_info.items():
            desc = f"\n\t{mh.metrics[keyword]['help']}" if keyword else ""
            infos.append(f"{k}: {short}{desc}")
        print("\n".join(infos))
        sys.exit(0)
    """

    cfg = get_cfg_defaults()
    if not args.config:
        log.error("--config param not provided, aborting...")
        sys.exit(2)
    cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
    cfg = expand_relative_paths(cfg)
    cfg.freeze()

    run_evaluation(cfg)
