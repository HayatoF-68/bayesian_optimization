import sys
import os
import shutil
import subprocess
import bayes_util
import experimental
from experimental import CallableObject, CallableObject_standard, generate_initial_condition
import image_to_video
import numpy as np
import datetime
import itertools
import random
import time

from yacs.config import CfgNode
from tools.util import parse_args
from tools import log
from config.defaults import get_cfg_defaults
from config.config_tools import expand_relative_paths
from config.verify_config import  check_express_config, global_checks, check_mot_config

from mot.run_tracker import run_mot, MOT_OUTPUT_NAME
from mtmc.run_mtmc import run_mtmc
from mtmc.run_express_mtmc import run_express_mtmc
from mtmc.output import save_tracklets_per_cam, save_tracklets_csv_per_cam, save_tracklets_txt_per_cam, annotate_video_mtmc
from evaluate.run_evaluate import run_evaluation

from sklearn.preprocessing import StandardScaler
import torch
import botorch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_model
#from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import gpytorch
from botorch.models.transforms import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from gpytorch.priors import LogNormalPrior, GammaPrior

awsimProjectRootPath = "/home/fujii/Documents/Projects/bayes-arrangement"
unityBinaryPath = "/home/fujii/Unity/Hub/Editor/2021.1.7f1/Editor/Unity"
save_dir = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/output"

# TODO 読み込むpthファイルの指定（数字部分を変更）
if os.path.exists(os.path.join(save_dir, "x_suggest_31.pth")):
    x_suggest = torch.load(os.path.join(save_dir, "x_suggest_31.pth"))

# else:
#     bounds = torch.tensor([
#         [-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0,-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0],
#         [60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0,60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0]
#         ], dtype=torch.float32)
#     # 条件を満たすサンプルを生成
#     callable_object = CallableObject()
#     x_suggest_list = []
#     # ランダムサンプリング
#     while len(x_suggest_list) < 2:
#         # 各次元ごとにランダムサンプリング
#         x_random = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, bounds.shape[1])
#         # callable_object の条件を満たすかをチェック
#         if callable_object(x_random) >= 0:
#             x_suggest_list.append(x_random)
#     x_suggest = torch.cat(x_suggest_list, dim=0)

else:
    #ベストスコア(192周目)
    #x_suggest = torch.tensor([[11.81876,-1.82564,50.40118,3.40359,-133.49689,12.12846,12.16882,-1.24480,-20.33683,5.27910,24.50269,23.46427,-18.15512,-0.49771,-56.68274,22.27353,51.96133,28.99672,47.20786,-1.61435,20.80215,24.77179,-159.88277,22.24188],[11.81876,-1.82564,50.40118,3.40359,-133.49689,12.12846,12.16882,-1.24480,-20.33683,5.27910,24.50269,23.46427,-18.15512,-0.49771,-56.68274,22.27353,51.96133,28.99672,47.20786,-1.61435,20.80215,24.77179,-159.88277,22.24188]])
    #1周目
    x_suggest = torch.tensor([[23.49128,-3.05251,-11.69620,3.73720,-146.98679,28.80864,47.09251,-3.61613,28.41834,3.80008,-176.60133,28.37472,31.78236,-0.68584,-11.46286,21.87878,-30.46082,18.87181,-17.18087,-2.57414,37.08733,24.59307,87.61929,24.83850],[-36.96875,-0.11099,-13.56612,20.70624,104.20670,29.82711,-36.96875,-0.11099,-13.56612,20.70624,104.20670,29.82711,34.02129,-2.26731,-12.62063,12.74348,-160.73717,15.64807,34.02129,-2.26731,-12.62063,12.74348,-160.73717,15.64807]])
    #x_suggest = torch.tensor([[16.34413,-1.97546,-24.25388,23.21188,-59.05220,13.41623,-14.07741,-3.15594,52.62189,7.96958,161.56049,21.11538,-14.89391,-0.84543,-17.22018,8.36512,89.19284,27.48003,10.24940,-3.11311,-10.62706,10.63922,73.39537,25.11854],[16.34413,-1.97546,-24.25388,23.21188,-59.05220,13.41623,-14.07741,-3.15594,52.62189,7.96958,161.56049,21.11538,-14.89391,-0.84543,-17.22018,8.36512,89.19284,27.48003,10.24940,-3.11311,-10.62706,10.63922,73.39537,25.11854]])
    #ベストスコア(24周目)　01234567>00234567
    #x_suggest = torch.tensor([[-13.82111,-2.43049,45.40679,16.86851,-139.60043,21.16823,-13.82111,-2.43049,45.40679,16.86851,-139.60043,21.16823,-16.14295,-3.63421,-29.26904,6.51583,71.33745,12.05106,-16.97007,-3.57683,-24.33042,14.65077,49.81635,16.50722],[-13.82111,-2.43049,45.40679,16.86851,-139.60043,21.16823,55.18008,-2.77013,-19.29512,12.30482,-30.61313,24.99692,-16.14295,-3.63421,-29.26904,6.51583,71.33745,12.05106,-16.97007,-3.57683,-24.33042,14.65077,49.81635,16.50722]])
    #天候変化ベストスコア(24周目)
    #x_suggest = torch.tensor([[-10.126758575439453,-2.714646816253662,44.314300537109375,6.933996677398682,132.45394897460938,22.357097625732422,28.47931671142578,-2.7366673946380615,-10.68524169921875,11.955851554870605,48.31401062011719,20.20355224609375,18.36400604248047,-2.6645874977111816,33.83680725097656,11.556488990783691,176.55670166015625,14.131566047668457,-31.26288604736328,-3.1710023880004883,29.115554809570312,4.384685516357422,-174.5219268798828,13.436285018920898]])

    #24:-13.82111,-2.43049,45.40679,16.86851,-139.60043,21.16823,55.18008,-2.77013,-19.29512,12.30482,-30.61313,24.99692,-16.14295,-3.63421,-29.26904,6.51583,71.33745,12.05106,-16.97007,-3.57683,-24.33042,14.65077,49.81635,16.50722
    #76:(-20.19599),(-2.41285),(-10.38531),(5.24808),(139.68750),(28.90949),(-22.67828),(-0.24368),(20.04003),(1.42007),(62.02554),(17.27127),(-10.23662),(-0.17269),(-24.03545),(20.43560),(-45.05414),(21.18891),(35.02604),(-0.69930),(21.08334),(7.43314),(168.33334),(12.84593)
    #107:(-16.53238),(-2.81453),(35.57478),(11.64919),(-150.33226),(17.32344),(-14.40484),(-3.21373),(43.34420),(19.97635),(81.33371),(21.06280),(11.09916),(-2.05980),(-21.50803),(0.67759),(19.32042),(17.52503),(38.74634),(-3.69407),(-12.25840),(27.52979),(-85.90146),(11.66607)
    #122:(-48.77448),(-2.18212),(-10.25956),(4.52233),(30.58379),(27.58411),(-19.45414),(-2.23600),(-42.75806),(9.78601),(8.54344),(28.79263),(19.39378),(-3.25866),(-21.41929),(23.23621),(35.46359),(18.14221),(-19.27752),(-2.07426),(22.30427),(2.53716),(44.63857),(24.26570)

    #4: 17.21343994140625,-2.4317941665649414,22.454010009765625,21.843463897705078,154.3568115234375,14.167081832885742,27.00354766845703,-1.702561855316162,23.524795532226562,12.334166526794434,-58.877838134765625,24.178756713867188,14.947883605957031,-3.9903461933135986,-30.898345947265625,15.46946907043457,-72.18538665771484,12.735721588134766,12.2760009765625,-3.7354602813720703,47.96112823486328,8.658060073852539,-89.96105194091797,16.796463012695312
    #27: -16.57012939453125,-0.8854765892028809,23.607162475585938,12.711281776428223,47.488983154296875,11.773448944091797,-10.976608276367188,-3.955110788345337,-21.348594665527344,2.478686571121216,91.4281005859375,21.510108947753906,-13.264781951904297,-2.664555549621582,-24.914073944091797,12.885775566101074,70.55986022949219,26.562055587768555,-16.423431396484375,-3.797466516494751,-34.04075241088867,3.0261826515197754,-12.516769409179688,10.61103343963623
    #30 12.633224487304688,-3.011385440826416,32.82380676269531,10.246377944946289,162.567138671875,22.059165954589844,-36.0755615234375,-3.375627279281616,20.76898193359375,20.002655029296875,99.64517211914062,12.73790454864502,18.320556640625,-1.7704153060913086,-38.88059997558594,22.11100959777832,-59.59752655029297,22.243696212768555,-17.621009826660156,-2.621119499206543,32.49980163574219,12.78707218170166,77.30429077148438,18.76268768310547
    #69 33.101585388183594,-2.9771780967712402,-17.073562622070312,8.368343353271484,-8.592056274414062,20.401357650756836,12.809593200683594,-3.3661842346191406,59.480438232421875,2.3562991619110107,-122.16749572753906,12.42773151397705,-19.791175842285156,-0.9039013385772705,-25.406906127929688,7.039366245269775,25.584091186523438,23.057443618774414,47.76483917236328,-0.11998963356018066,-10.651630401611328,21.1389102935791,45.33027648925781,14.985851287841797
    #74 13.565391540527344,-2.7527554035186768,-13.855083465576172,19.186317443847656,48.288665771484375,11.988805770874023,-22.16710662841797,-1.5087900161743164,-19.484580993652344,16.085105895996094,72.88003540039062,15.790003776550293,-16.41055679321289,-3.823078155517578,58.910682678222656,11.101909637451172,122.21856689453125,28.99325180053711,-48.110008239746094,-0.6138172149658203,-13.886981964111328,29.180883407592773,-12.040451049804688,15.427323341369629



#x_suggest = torch.load(os.path.join(save_dir, "x_suggest_36.pth")) if os.path.exists(os.path.join(save_dir, "x_suggest_36.pth")) else torch.tensor([[12.75,-4,-53.5,-0.9,-28.61,12.07,-55.12,-4,20,11,15.6,21.3,54.38,-4,23.23,5.9,-130.21,26.1,-14.3,-4,57.5,5.5,147.13,14.99],[-22.2,0,-27.17,7.5,22.8,30,19.3,0,-19.3,9.62,-51.14,30,-11.3,-4,35.53,12.8,32.1,10,17.27,0,21.91,7.5,-158.4,30]])
train_x = torch.load(os.path.join(save_dir, "train_x_29.pth")) if os.path.exists(os.path.join(save_dir, "train_x_29.pth")) else torch.tensor([[]])
train_y = torch.load(os.path.join(save_dir, "train_y_29.pth")) if os.path.exists(os.path.join(save_dir, "train_y_29.pth")) else torch.tensor([[]])
train_y_1 = torch.load(os.path.join(save_dir, "train_y_1_29.pth")) if os.path.exists(os.path.join(save_dir, "train_y_1_29.pth")) else torch.tensor([[]])
train_y_2 = torch.load(os.path.join(save_dir, "train_y_2_29.pth")) if os.path.exists(os.path.join(save_dir, "train_y_2_29.pth")) else torch.tensor([[]])
train_y_3 = torch.load(os.path.join(save_dir, "train_y_3_29.pth")) if os.path.exists(os.path.join(save_dir, "train_y_3_29.pth")) else torch.tensor([[]])
train_y_4 = torch.load(os.path.join(save_dir, "train_y_4_29.pth")) if os.path.exists(os.path.join(save_dir, "train_y_4_29.pth")) else torch.tensor([[]])

kwargs = {'camera1_pos_x':x_suggest[-1][0],'camera1_pos_y':x_suggest[-1][1],'camera1_pos_z':x_suggest[-1][2],'camera1_angle_x':x_suggest[-1][3],'camera1_angle_y':x_suggest[-1][4],'camera1_focalLength':x_suggest[-1][5],
        'camera2_pos_x':x_suggest[-1][6],'camera2_pos_y':x_suggest[-1][7],'camera2_pos_z':x_suggest[-1][8],'camera2_angle_x':x_suggest[-1][9],'camera2_angle_y':x_suggest[-1][10],'camera2_focalLength':x_suggest[-1][11],
        'camera3_pos_x':x_suggest[-1][12],'camera3_pos_y':x_suggest[-1][13],'camera3_pos_z':x_suggest[-1][14],'camera3_angle_x':x_suggest[-1][15],'camera3_angle_y':x_suggest[-1][16],'camera3_focalLength':x_suggest[-1][17],
        'camera4_pos_x':x_suggest[-1][18],'camera4_pos_y':x_suggest[-1][19],'camera4_pos_z':x_suggest[-1][20],'camera4_angle_x':x_suggest[-1][21],'camera4_angle_y':x_suggest[-1][22],'camera4_focalLength':x_suggest[-1][23],
        'camera5_pos_x':x_suggest[-2][0],'camera5_pos_y':x_suggest[-2][1],'camera5_pos_z':x_suggest[-2][2],'camera5_angle_x':x_suggest[-2][3],'camera5_angle_y':x_suggest[-2][4],'camera5_focalLength':x_suggest[-2][5],
        'camera6_pos_x':x_suggest[-2][6],'camera6_pos_y':x_suggest[-2][7],'camera6_pos_z':x_suggest[-2][8],'camera6_angle_x':x_suggest[-2][9],'camera6_angle_y':x_suggest[-2][10],'camera6_focalLength':x_suggest[-2][11],
        'camera7_pos_x':x_suggest[-2][12],'camera7_pos_y':x_suggest[-2][13],'camera7_pos_z':x_suggest[-2][14],'camera7_angle_x':x_suggest[-2][15],'camera7_angle_y':x_suggest[-2][16],'camera7_focalLength':x_suggest[-2][17],
        'camera8_pos_x':x_suggest[-2][18],'camera8_pos_y':x_suggest[-2][19],'camera8_pos_z':x_suggest[-2][20],'camera8_angle_x':x_suggest[-2][21],'camera8_angle_y':x_suggest[-2][22],'camera8_focalLength':x_suggest[-2][23]
        }

#現在の日時を取得
now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')
save_path = os.path.join(save_dir, now)
os.makedirs(save_path, exist_ok=True)
output_file = os.path.join(save_path, "process_times.txt")

with open(output_file, "w") as f:
    f.write('Cycle,Step1,Step2,Step3,Step4\n')

"""
for i in range(300):
    # Set export directory
    # bayes_util.GenerateScriptToSetExportDirectory( awsimProjectRootPath, "trial_"+str(i))
    # Create a unity script that changes camera parameters (position/angle/focalLength)
    # You need to specify all the parameters.
    # print("Generating camera parameter script...")
    print(str(i)+"回目")

    bayes_util.GenerateScriptToChangeParameter(awsimProjectRootPath, **kwargs)
    print("Done.")

    # Build a player with the specified camera paraemters
    # ref: https://docs.unity3d.com/ja/2018.4/Manual/CommandLineArguments.html
    print("Building player...")
    commandToBuildPlayer = [
        unityBinaryPath,
        "-quit",
        #"-batchmode",
        "-buildTarget",
        "Linux64",
        "-executeMethod",
        "BuildPlayer.MyBuild",
        "-projectPath",
        awsimProjectRootPath
        ]
    subprocess.call(commandToBuildPlayer)
    print("Done.")

    start_time = time.time()
    # Run the built player
    print("Running player...")
    # commandToRunPlayer =[awsimProjectRootPath+"/Linux/Player"]
    commandToRunPlayer =["sudo", awsimProjectRootPath+"/Linux/Player"]
    subprocess.call(commandToRunPlayer)
    print("Done.")

    path1 = awsimProjectRootPath + "/Assets/Addition/TrafficData0.txt"
    path2 = awsimProjectRootPath + "/Assets/Addition/TrafficData1.txt"
    path3 = awsimProjectRootPath + "/Assets/Addition/TrafficData2.txt"
    path4 = awsimProjectRootPath + "/Assets/Addition/TrafficData3.txt"
    path5 = awsimProjectRootPath + "/Assets/Addition/TrafficData4.txt"
    path6 = awsimProjectRootPath + "/Assets/Addition/TrafficData5.txt"
    path7 = awsimProjectRootPath + "/Assets/Addition/TrafficData6.txt"
    path8 = awsimProjectRootPath + "/Assets/Addition/TrafficData7.txt"
    path9 = awsimProjectRootPath + "/Assets/Addition/TrafficData8.txt"

    path10 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages1"
    path11 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages2"
    path12 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages3"
    path13 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages4"
    path14 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages5"
    path15 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages6"
    path16 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages7"
    path17 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages8"

    path18 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_1.txt"
    path19 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_2.txt"
    path20 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_3.txt"
    path21 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_4.txt"
    path22 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_5.txt"
    path23 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_6.txt"
    path24 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_7.txt"
    path25 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_8.txt"

    path26 = awsimProjectRootPath + "/Assets/Addition/CapturedImages"
    path27 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix"

    pathes = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, path14, path15, path16, path17, path18, path19, path20, path21, path22, path23, path24, path25, path26, path27]
    pathes2 = [path26, path27, path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, path14, path15, path16, path17, path18, path19, path20, path21, path22, path23, path24, path25]


    for path in pathes2:
        subprocess.call(["sudo", "chown", "fujii:fujii", path])

    #video creation
    print("Creating video...")
    image_to_video.image_to_video()
    print("Done.")

    for path in pathes:
        subprocess.call(["sudo", "rm", "-r", path])

    step1_time = time.time() - start_time
    start_time = time.time()

    args = parse_args("Express MTMC: run MOT on all cameras and then MTMC.")
    cfg = get_cfg_defaults()

    if args.config:
        cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
    cfg = expand_relative_paths(cfg)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, now, "epoch"+str(i))
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    if not global_checks["OUTPUT_DIR"](cfg.OUTPUT_DIR):
        log.error(
            "Invalid param value in: OUTPUT_DIR. Provide an absolute path to a directory, whose parent exists.")
        sys.exit(2)

    log.log_init(os.path.join(cfg.OUTPUT_DIR, args.log_filename),
                args.log_level, not args.no_log_stdout)

    MTMC_OUTPUT_NAME = "mtmc"
    #Run Express MTMC on a given config.
    mot_configs = []
    cam_names, cam_dirs = [], []
    for cam_idx, cam_info in enumerate(cfg.EXPRESS.CAMERAS):
        cam_cfg = cfg.clone()
        cam_cfg.defrost()
        for key, val in cam_info.items():
            setattr(cam_cfg.MOT, key.upper(), val)

        cam_video_name = os.path.split(cam_cfg.MOT.VIDEO)[1].split(".")[0]
        cam_names.append(cam_video_name)

        # set output dir of MOT to a unique folder under the root OUTPUT_DIR
        cam_dir = os.path.join(cfg.OUTPUT_DIR, f"{cam_idx}_{cam_video_name}")
        cam_dirs.append(cam_dir)
        cam_cfg.OUTPUT_DIR = cam_dir
        if len(cfg.EVAL.GROUND_TRUTHS) == len(cfg.EXPRESS.CAMERAS):
            cam_cfg.EVAL.GROUND_TRUTHS = [cfg.EVAL.GROUND_TRUTHS[cam_idx]]
        cam_cfg.freeze()
        mot_configs.append(cam_cfg)

    single_score_list = []
    # run MOT in all cameras
    for mot_conf in mot_configs:
        _, single_score = run_mot(mot_conf)
        single_score_list.append(single_score)
    print(f"single_score: {single_score_list}")
    top_6_ids = sorted(range(len(single_score_list)), key=lambda j: single_score_list[j], reverse=True)[:6]
    print(f"top_6_ids: {top_6_ids}")
    pickle_paths = [os.path.join(path, f"{MOT_OUTPUT_NAME}.pkl") for path in cam_dirs]

    step2_time = time.time() - start_time
    start_time = time.time()

    log.info("Express: Running MOT on all cameras finished. Running MTMC...")

    combinations = list(itertools.combinations(top_6_ids, 4))
    print(f"combinations: {combinations}")

    for combo in combinations:
        combo = tuple(random.sample(combo, len(combo)))
        print(f"combo: {combo}")

        combostr = "".join([str(j) for j in combo])

        mtmc_cfg = cfg.clone()
        mtmc_cfg.defrost()
        mtmc_cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "combo"+ combostr)
        mtmc_cfg.EXPRESS.CAMERAS = [cfg.EXPRESS.CAMERAS[j] for j in combo]
        mtmc_cfg.EVAL.GROUND_TRUTHS = [cfg.EVAL.GROUND_TRUTHS[j] for j in combo]
        mtmc_cfg.MTMC.PICKLED_TRACKLETS = [pickle_paths[j] for j in combo]
        mtmc_cfg.freeze()

        mot_configs_combo = [mot_configs[j] for j in combo]

        all_ground_truths_empty = all(all(os.path.getsize(gt) == 0 for gt in mot_conf.EVAL.GROUND_TRUTHS) for mot_conf in mot_configs_combo)
        all_predictions_empty = all(all(os.path.getsize(pred) == 0 for pred in mot_conf.EVAL.PREDICTIONS) for mot_conf in mot_configs_combo)
        if all_ground_truths_empty or all_predictions_empty:
            score = 0
        else:
            mtracks = run_mtmc(mtmc_cfg)
            log.info("Express: Running MTMC on all cameras finished. Saving final results ...")

            cam_combo_dirs = [os.path.join(mtmc_cfg.OUTPUT_DIR, f"{j}_{cam_names[j]}") for j in combo]
            for directory in cam_combo_dirs:
                os.makedirs(directory, exist_ok=True)

            # save single cam tracks
            final_pickle_paths = [os.path.join(d, f"{MTMC_OUTPUT_NAME}.pkl") for d in cam_combo_dirs]
            final_csv_paths = [os.path.join(d, f"{MTMC_OUTPUT_NAME}.csv") for d in cam_combo_dirs]
            final_txt_paths = [os.path.join(d, f"{MTMC_OUTPUT_NAME}.txt") for d in cam_combo_dirs]
            save_tracklets_per_cam(mtracks, final_pickle_paths)
            save_tracklets_txt_per_cam(mtracks, final_txt_paths)
            save_tracklets_csv_per_cam(mtracks, final_csv_paths)


            # if cfg.EXPRESS.FINAL_VIDEO_OUTPUT:
            #     for j, cam_dir in enumerate(cam_combo_dirs):
            #         video_in = mot_configs[combo[j]].MOT.VIDEO
            #         video_ext = video_in.split(".")[1]
            #         video_out = os.path.join(cam_dir, f"{MTMC_OUTPUT_NAME}.{video_ext}")
            #         annotate_video_mtmc(video_in, video_out, mtracks, j, font=cfg.FONT, fontsize=cfg.FONTSIZE)
            #         log.info(f"Express: video {j} saved.")


            if len(cfg.EVAL.GROUND_TRUTHS) == 0:
                log.info("Ground truths are not provided for evaluation, terminating.")

            log.info("Ground truth annotations are provided, trying to evaluate MTMC ...")
            if len(cfg.EVAL.GROUND_TRUTHS) != len(cam_names):
                log.error(
                    "Number of ground truth files != number of cameras, aborting evaluation ...")

            mtmc_cfg.defrost()
            mtmc_cfg.EVAL.PREDICTIONS = final_txt_paths
            mtmc_cfg.freeze()
            score = run_evaluation(mtmc_cfg)

        if score:
            log.info("Evaluation successful.")
        else:
            log.error("Evaluation unsuccessful: probably EVAL config had some errors.")


        x_test=[]
        for k,l in enumerate(combo):
            if k==0:
                if l==0: x_test[:6] = x_suggest[-1][:6]
                elif l==1: x_test[:6] = x_suggest[-1][6:12]
                elif l==2: x_test[:6] = x_suggest[-1][12:18]
                elif l==3: x_test[:6] = x_suggest[-1][18:]
                elif l==4: x_test[:6] = x_suggest[-2][:6]
                elif l==5: x_test[:6] = x_suggest[-2][6:12]
                elif l==6: x_test[:6] = x_suggest[-2][12:18]
                elif l==7: x_test[:6] = x_suggest[-2][18:]
            elif k==1:
                if l==0: x_test[6:12] = x_suggest[-1][:6]
                elif l==1: x_test[6:12] = x_suggest[-1][6:12]
                elif l==2: x_test[6:12] = x_suggest[-1][12:18]
                elif l==3: x_test[6:12] = x_suggest[-1][18:]
                elif l==4: x_test[6:12] = x_suggest[-2][:6]
                elif l==5: x_test[6:12] = x_suggest[-2][6:12]
                elif l==6: x_test[6:12] = x_suggest[-2][12:18]
                elif l==7: x_test[6:12] = x_suggest[-2][18:]
            elif k==2:
                if l==0: x_test[12:18] = x_suggest[-1][:6]
                elif l==1: x_test[12:18] = x_suggest[-1][6:12]
                elif l==2: x_test[12:18] = x_suggest[-1][12:18]
                elif l==3: x_test[12:18] = x_suggest[-1][18:]
                elif l==4: x_test[12:18] = x_suggest[-2][:6]
                elif l==5: x_test[12:18] = x_suggest[-2][6:12]
                elif l==6: x_test[12:18] = x_suggest[-2][12:18]
                elif l==7: x_test[12:18] = x_suggest[-2][18:]
            elif k==3:
                if l==0: x_test[18:] = x_suggest[-1][:6]
                elif l==1: x_test[18:] = x_suggest[-1][6:12]
                elif l==2: x_test[18:] = x_suggest[-1][12:18]
                elif l==3: x_test[18:] = x_suggest[-1][18:]
                elif l==4: x_test[18:] = x_suggest[-2][:6]
                elif l==5: x_test[18:] = x_suggest[-2][6:12]
                elif l==6: x_test[18:] = x_suggest[-2][12:18]
                elif l==7: x_test[18:] = x_suggest[-2][18:]

        print(f"x_test: {x_test}")
        train_x = torch.cat((train_x, torch.tensor([x_test])), dim=0) if train_x.size(1) > 0 else torch.tensor([x_test])
        print(f"train_x.shape: {train_x.shape}")
        with open(os.path.join(save_dir, now, "x_test.txt"), "a") as f:
            f.write(str(i+1)+":"+str(combo)+":")
            for x in x_test:
                f.write(str(x)+",")
            f.seek(f.tell()-1, os.SEEK_SET)
            f.truncate()
            f.write("\n")

        print(f"score: {score}")
        train_y = torch.cat((train_y, torch.tensor([[score]])), dim=0) if train_y.size(1) > 0 else torch.tensor([[score]])
        print(f"train_y.shape: {train_y.shape}")
        with open(os.path.join(save_dir, now, "score.txt"), "a") as f:
            f.write(str(i+1)+":"+str(combo)+":"+str(score)+"\n")

        single_score_1 = single_score_list[combo[0]] if isinstance(single_score_list, list) else 0
        single_score_2 = single_score_list[combo[1]] if isinstance(single_score_list, list) else 0
        single_score_3 = single_score_list[combo[2]] if isinstance(single_score_list, list) else 0
        single_score_4 = single_score_list[combo[3]] if isinstance(single_score_list, list) else 0
        train_y_1 = torch.cat((train_y_1, torch.tensor([[single_score_1]])), dim=0) if train_y_1.size(1) > 0 else torch.tensor([[single_score_1]])
        train_y_2 = torch.cat((train_y_2, torch.tensor([[single_score_2]])), dim=0) if train_y_2.size(1) > 0 else torch.tensor([[single_score_2]])
        train_y_3 = torch.cat((train_y_3, torch.tensor([[single_score_3]])), dim=0) if train_y_3.size(1) > 0 else torch.tensor([[single_score_3]])
        train_y_4 = torch.cat((train_y_4, torch.tensor([[single_score_4]])), dim=0) if train_y_4.size(1) > 0 else torch.tensor([[single_score_4]])
        print(f"train_y_1.shape: {train_y_1.shape}")
        print(f"train_y_2.shape: {train_y_2.shape}")
        print(f"train_y_3.shape: {train_y_3.shape}")
        print(f"train_y_4.shape: {train_y_4.shape}")
        with open(os.path.join(save_dir, now, "single_score.txt"), "a") as f:
            f.write(str(i+1)+":"+str(combo)+":"+str(single_score_1)+","+str(single_score_2)+","+str(single_score_3)+","+str(single_score_4)+"\n")

    with open(os.path.join(save_dir, now, "x_test.txt"), "a") as f:
        f.write("\n")
    #shutil.copytree(os.path.join(os.getcwd(), "datasets"), cfg.OUTPUT_DIR, dirs_exist_ok=True)
    with open(os.path.join(save_dir, now, "score.txt"), "a") as f:
        f.write("\n")
    with open(os.path.join(save_dir, now, "single_score.txt"), "a") as f:
        f.write("\n")

    step3_time = time.time() - start_time
    start_time = time.time()


    bounds_min = torch.tensor([-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0,-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0])
    bounds_max = torch.tensor([60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0,60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0])
    train_x_scaled = (train_x - bounds_min) / (bounds_max - bounds_min)

    i0 = torch.zeros(train_x.shape[0], 1)
    i1 = torch.ones(train_x.shape[0], 1)
    i2 = 2 * torch.ones(train_x.shape[0], 1)
    i3 = 3 * torch.ones(train_x.shape[0], 1)
    i4 = 4 * torch.ones(train_x.shape[0], 1)

    x_multi = torch.cat([
        torch.cat([train_x_scaled, i0], dim=-1),
        torch.cat([torch.cat([train_x_scaled[:, :6]]*4,dim=1), i1], dim=-1),
        torch.cat([torch.cat([train_x_scaled[:, 6:12]]*4, dim=1), i2], dim=-1),
        torch.cat([torch.cat([train_x_scaled[:, 12:18]]*4, dim=1), i3], dim=-1),
        torch.cat([torch.cat([train_x_scaled[:, 18:]]*4, dim=1), i4], dim=-1),
        ], dim=0)
    y_multi = torch.cat([train_y, train_y_1, train_y_2, train_y_3, train_y_4])
    print(f"x_multi.shape: {x_multi.shape}")
    print(f"y_multi.shape: {y_multi.shape}")

    with gpytorch.settings.cholesky_jitter(1e-3):
        dim = x_multi.shape[-1] - 1  # 入力次元数（タスク情報を除く）
        #print(f"dim: {dim}")
        #print("x_multi shape:", x_multi.shape)  # 入力データの形状
        #print("y_multi shape:", y_multi.shape)  # 出力データの形状
        mu_0 = 0.0  # LogNormalPriorのmuパラメータ
        sigma_0 = 1.0  # LogNormalPriorのsigmaパラメータ

        #model=MultiTaskGP(x_multi, y_multi, task_feature=-1)
        model = MultiTaskGP(
            train_X=x_multi,
            train_Y=y_multi,
            task_feature=-1,  # タスクインデックスの特徴量（最後の次元がタスクを示す場合）
            #input_transform=Normalize(d=dim),
            #outcome_transform=Standardize(m=y_multi.shape[-1]),  # タスク数で標準化
            # ---- 追加 ---
            covar_module=ScaleKernel(
                base_kernel=MaternKernel(
                    nu=2.5,
                    ard_num_dims=dim,
                    # ---- 変更されたlengthscale_prior ----
                    lengthscale_prior=LogNormalPrior(mu_0 + np.log(dim) / 2, sigma_0),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        )

        model.likelihood.noise_covar.noise = torch.tensor(1e-3, requires_grad=True)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # TODO 変更箇所② num_samplesの値
        ref_point=torch.tensor([0,0,0,0,0])
        pareto_y = torch.tensor([[0.5,0.15,0.15,0.15,0.15]])
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_y)

        weights = torch.tensor([0.4, 0.15, 0.15, 0.15, 0.15])
        objective = WeightedMCMultiOutputObjective(weights=weights)
        sampler = SobolQMCNormalSampler(num_samples=3000)

        qEHVI = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            objective=objective,
            partitioning=partitioning,
            sampler=sampler,
        )

        # bounds = torch.tensor([
        #     [-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0],
        #     [60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0]
        # ], dtype=torch.float32)

        bounds = torch.stack([torch.zeros(24, dtype=torch.float32), torch.ones(24, dtype=torch.float32)])

        # TODO 変更箇所③ num_restartsの値
        num_restarts  = 100
        #scales = [120, 4, 120, 30, 360, 20, 120, 4, 120, 30, 360, 20, 120, 4, 120, 30, 360, 20, 120, 4, 120, 30, 360, 20]
        callable_object_standard = CallableObject_standard()

        batch_initial_conditions = torch.empty(0, 1, 24, requires_grad=True)

        while batch_initial_conditions.size(0) < num_restarts:
            batch_initial_condition = generate_initial_condition()
            if callable_object_standard(batch_initial_condition) >= torch.tensor(0):
                # 条件を満たす場合、batch_initial_conditionsに追加
                if batch_initial_conditions.size(0) == 0:
                    batch_initial_conditions = batch_initial_condition
                else:
                    batch_initial_conditions = torch.cat((batch_initial_conditions, batch_initial_condition), dim=0)

        # # 初期条件の設定
        # initial_condition = train_x_scaled[-1].view(1, 1, 24).clone().detach().requires_grad_(True)
        # # バッチ初期条件の初期化
        # batch_initial_conditions = torch.empty(0, 1, 24, requires_grad=True)
        # # 条件を満たすまで繰り返す
        # while batch_initial_conditions.size(0) < num_restarts:
        #     noise = torch.randn_like(initial_condition)
        #     # TODO 変更箇所④ ratioの値
        #     ratio=0.1
        #     for d in range(24):
        #         #noise[:, :, d] = noise[:, :, d] * scales[d] * ratio
        #         noise[:, :, d] = noise[:, :, d] * ratio
        #     noise.requires_grad_(True)
        #     noisy_initial_condition = initial_condition + noise

        #     if callable_object(noisy_initial_condition) >= torch.tensor(0):
        #         # 条件を満たす場合、batch_initial_conditionsに追加
        #         if batch_initial_conditions.size(0) == 0:
        #             batch_initial_conditions = noisy_initial_condition
        #             print(f'noisy_initial_condition: {noisy_initial_condition}')
        #         else:
        #             batch_initial_conditions = torch.cat((batch_initial_conditions, noisy_initial_condition), dim=0)

        #下記は省くか要検討
        #batch_initial_conditions = (batch_initial_conditions - bounds_min) / (bounds_max - bounds_min)


        x_next, acq_value = optimize_acqf(
            qEHVI,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            nonlinear_inequality_constraints=[CallableObject_standard()],
            batch_initial_conditions=batch_initial_conditions,
        )

    x_next = x_next * (bounds_max - bounds_min) + bounds_min

    x_suggest = torch.cat((x_suggest, x_next))
    print(f'次の候補点は{x_next}です。')
    with open(os.path.join(save_dir, now, "x_next.txt"), "a") as f:
        f.write(str(i+1)+",")
        for x in x_next:
            f.write(str(x)+",")
        f.seek(f.tell()-1, os.SEEK_SET)
        f.truncate()
        f.write("\n")

    torch.save(model.state_dict(), os.path.join(save_dir, now, "epoch"+str(i), f'model_checkpoint_{i}.pth'))
    torch.save(x_suggest, os.path.join(save_dir, now, "epoch"+str(i), f'x_suggest_{i}.pth'))
    torch.save(train_x, os.path.join(save_dir, now, "epoch"+str(i), f'train_x_{i}.pth'))
    torch.save(train_y, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_{i}.pth'))
    torch.save(train_y_1, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_1_{i}.pth'))
    torch.save(train_y_2, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_2_{i}.pth'))
    torch.save(train_y_3, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_3_{i}.pth'))
    torch.save(train_y_4, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_4_{i}.pth'))

    kwargs = {'camera1_pos_x':x_suggest[-1][0],'camera1_pos_y':x_suggest[-1][1],'camera1_pos_z':x_suggest[-1][2],'camera1_angle_x':x_suggest[-1][3],'camera1_angle_y':x_suggest[-1][4],'camera1_focalLength':x_suggest[-1][5],
        'camera2_pos_x':x_suggest[-1][6],'camera2_pos_y':x_suggest[-1][7],'camera2_pos_z':x_suggest[-1][8],'camera2_angle_x':x_suggest[-1][9],'camera2_angle_y':x_suggest[-1][10],'camera2_focalLength':x_suggest[-1][11],
        'camera3_pos_x':x_suggest[-1][12],'camera3_pos_y':x_suggest[-1][13],'camera3_pos_z':x_suggest[-1][14],'camera3_angle_x':x_suggest[-1][15],'camera3_angle_y':x_suggest[-1][16],'camera3_focalLength':x_suggest[-1][17],
        'camera4_pos_x':x_suggest[-1][18],'camera4_pos_y':x_suggest[-1][19],'camera4_pos_z':x_suggest[-1][20],'camera4_angle_x':x_suggest[-1][21],'camera4_angle_y':x_suggest[-1][22],'camera4_focalLength':x_suggest[-1][23],
        'camera5_pos_x':x_suggest[-2][0],'camera5_pos_y':x_suggest[-2][1],'camera5_pos_z':x_suggest[-2][2],'camera5_angle_x':x_suggest[-2][3],'camera5_angle_y':x_suggest[-2][4],'camera5_focalLength':x_suggest[-2][5],
        'camera6_pos_x':x_suggest[-2][6],'camera6_pos_y':x_suggest[-2][7],'camera6_pos_z':x_suggest[-2][8],'camera6_angle_x':x_suggest[-2][9],'camera6_angle_y':x_suggest[-2][10],'camera6_focalLength':x_suggest[-2][11],
        'camera7_pos_x':x_suggest[-2][12],'camera7_pos_y':x_suggest[-2][13],'camera7_pos_z':x_suggest[-2][14],'camera7_angle_x':x_suggest[-2][15],'camera7_angle_y':x_suggest[-2][16],'camera7_focalLength':x_suggest[-2][17],
        'camera8_pos_x':x_suggest[-2][18],'camera8_pos_y':x_suggest[-2][19],'camera8_pos_z':x_suggest[-2][20],'camera8_angle_x':x_suggest[-2][21],'camera8_angle_y':x_suggest[-2][22],'camera8_focalLength':x_suggest[-2][23]
        }

    # Reset script to its original contents
    print("Resetting scripts...")
    bayes_util.ResetScripts(awsimProjectRootPath)
    print("Done.")

    step4_time = time.time() - start_time

    with open(output_file, "a") as f:
        f.write(f'{i},{step1_time:.4f},{step2_time:.4f},{step3_time:.4f},{step4_time:.4f}\n')

"""


"""
    # BoTorchのためのデータ変換
    train_x = torch.tensor(x_array, dtype=torch.float32)
    train_y = torch.tensor(y_array, dtype=torch.float32)
    # ガウシアンプロセスモデルの定義とフィッティング
    gp = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    # 獲得関数の定義
    UCB = UpperConfidenceBound(gp, beta=0.1)
    # パラメータの範囲
    bounds = torch.tensor([
        [-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0],
        [60.0, 0.0, 60.0, 45.0, 180.0, 30.0, 60.0, 0.0, 60.0, 45.0, 180.0, 30.0, 60.0, 0.0, 60.0, 45.0, 180.0, 30.0, 60.0, 0.0, 60.0, 45.0, 180.0, 30.0]
        ], dtype=torch.float32)

    # 最適化
    candidate, acq_value = optimize_acqf(
        acq_function=UCB,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=100,
    )
    # 予測
    gp.eval()
    with torch.no_grad():
        pred = gp(candidate)
        y_mean = pred.mean
        y_variance = pred.variance
    # 結果の出力
    x_suggest = candidate.numpy()
    y_mean = y_mean.numpy()
    y_variance = y_variance.numpy()
    x_array = np.vstack((x_array, x_suggest))
    print(f'つぎは{x_suggest}、yの予測平均は{y_mean}、分散は{y_variance}')

    kwargs = {
        'camera1_pos_x': x_suggest[-1][0], 'camera1_pos_y': x_suggest[-1][1], 'camera1_pos_z': x_suggest[-1][2],
        'camera1_angle_x': x_suggest[-1][3], 'camera1_angle_y': x_suggest[-1][4], 'camera1_focalLength': x_suggest[-1][5],
        'camera2_pos_x': x_suggest[-1][6], 'camera2_pos_y': x_suggest[-1][7], 'camera2_pos_z': x_suggest[-1][8],
        'camera2_angle_x': x_suggest[-1][9], 'camera2_angle_y': x_suggest[-1][10], 'camera2_focalLength': x_suggest[-1][11],
        'camera3_pos_x': x_suggest[-1][12], 'camera3_pos_y': x_suggest[-1][13], 'camera3_pos_z': x_suggest[-1][14],
        'camera3_angle_x': x_suggest[-1][15], 'camera3_angle_y': x_suggest[-1][16], 'camera3_focalLength': x_suggest[-1][17],
        'camera4_pos_x': x_suggest[-1][18], 'camera4_pos_y': x_suggest[-1][19], 'camera4_pos_z': x_suggest[-1][20],
        'camera4_angle_x': x_suggest[-1][21], 'camera4_angle_y': x_suggest[-1][22], 'camera4_focalLength': x_suggest[-1][23]
        }
    """
    #kwargs, x_array = experimental.bayesian_optimization(x_array, y_array)


"""
kwargs = {'camera1_pos_x': 0,'camera1_pos_y': 0,'camera1_pos_z': -60,'camera1_angle_x': 0,'camera1_angle_y': 0,'camera1_focalLength': 20,
        'camera2_pos_x': -60,'camera2_pos_y': 0,'camera2_pos_z': 0,'camera2_angle_x': 0,'camera2_angle_y': 90,'camera2_focalLength': 20,
        'camera3_pos_x': 60,'camera3_pos_y': 0,'camera3_pos_z': 0,'camera3_angle_x': 0,'camera3_angle_y': -90,'camera3_focalLength': 20,
        'camera4_pos_x': 0,'camera4_pos_y': -4,'camera4_pos_z': 60,'camera4_angle_x': 0,'camera4_angle_y': 180,'camera4_focalLength': 20
        }
"""

"""
train_x =torch.tensor([[-22.2, 0, -27.17, 7.5, 22.8, 30, 19.3, 0, -19.3, 9.62, -51.14, 30, -11.3, -4, 35.53, 12.8, 32.1, 10, 17.27, 0, 21.91, 7.5, -158.4, 30],
                    [12.75, -4, -53.5, -0.9, -28.61, 12.07, -55.12, -4, 19.6, 11, 135.6, 21.3, 54.38, -4, 23.23, 5.9, -130.21, 26.1, -14.3, -4, 57.5, 5.5, 147.13, 14.99],
                    [-22.9, 0, -23.1, 0.4, 33.9, 16.7, -23.8, 0, 25.36, -2.4, 133.07, 12.7, 20.5, 0, -19.81, 5.71, -43.8, 16.1, -11.45, -4, 60, 2.8, 118.3, 10],
                    [-14.7, -4, -55.9, 5, 19.7, 34.6, -57.28, -4, 21.7, 2.9, 113.47, 22, 55.21, -4, 19.1, 8.2, -113.3, 31.5, 11.9, -4, 54.7, 6.8, 209.06, 20.34],
                    [-11.99, 0, -60, 36.21, 29.26, 10, -59.98, 0, 18.28, 18.4, 118.9, 10, 60, -4, 20.2, 16.3, -162.6, 10, 11.6, -4, 60, 33.3, 228.37,10],
                    [-17.28, 0, 25.5, 24.8, 132.5, 20, -13.1, 0, -19, 37.95, 17.6, 10, 10.44, 0, 25.33, 25.01, -150.2, 20, 11.8, 0, -20.02, 31.5, -19.4, 10],
                    [20.6, -4, -22.64, 0, -35.7, 20, -22.3, -4, 21.6, 0, 124.3, 20, -19.5, -4, -14.9, 0, 46.8, 20, 16.1, -4, 31, 0, 217.4, 20],
                    [0,0,-60,0,0,20,-60,0,0,0,90,20,60,0,0,0,-90,20,0,-4,60,0,180,20]])
train_y = torch.tensor([[0.48378123635026093],[0.5099057614050118],[0.38811120832118207],[0.49696131801031135],[0.36051245740585036],[0.42181986323039106],[0.45015281957957115]])
train_y_1 = torch.tensor([[0.21452691061953],[0.14123480914395847],[0.17281573448661255],[0.19435323598029897],[0.09058171438686552],[0.18338565622642428],[0.210568062521871]])
train_y_2 = torch.tensor([[0.21452691061953],[0.14123480914395847],[0.17281573448661255],[0.19435323598029897],[0.09058171438686552],[0.18338565622642428],[0.210568062521871]])
train_y_3 = torch.tensor([[0.21452691061953],[0.14123480914395847],[0.17281573448661255],[0.19435323598029897],[0.09058171438686552],[0.18338565622642428],[0.210568062521871]])
train_y_4 = torch.tensor([[0.21452691061953],[0.14123480914395847],[0.17281573448661255],[0.19435323598029897],[0.09058171438686552],[0.18338565622642428],[0.210568062521871]])
"""

"""
#ayazu
kwargs = {'camera1_pos_x': -22.2,'camera1_pos_y': 0,'camera1_pos_z': -27.17,'camera1_angle_x': 7.5,'camera1_angle_y':22.8,'camera1_focalLength': 30,
        'camera2_pos_x': 19.3,'camera2_pos_y': 0,'camera2_pos_z': -19.3,'camera2_angle_x': 9.62,'camera2_angle_y': -51.14,'camera2_focalLength': 30,
        'camera3_pos_x': -11.3,'camera3_pos_y': -4,'camera3_pos_z': 35.53,'camera3_angle_x': 12.8,'camera3_angle_y': 32.1,'camera3_focalLength': 10,
        'camera4_pos_x': 17.27,'camera4_pos_y': 0,'camera4_pos_z': 21.91,'camera4_angle_x': 7.5,'camera4_angle_y': -158.4,'camera4_focalLength': 30
        }
#ide
kwargs = {'camera1_pos_x': 12.75,'camera1_pos_y': -4,'camera1_pos_z': -53.5,'camera1_angle_x': -0.9,'camera1_angle_y': -28.61,'camera1_focalLength': 12.07,
        'camera2_pos_x': -55.12,'camera2_pos_y': -4,'camera2_pos_z': 19.6,'camera2_angle_x': 11,'camera2_angle_y': 135.6,'camera2_focalLength': 21.3,
        'camera3_pos_x': 54.38,'camera3_pos_y': -4,'camera3_pos_z': 23.23,'camera3_angle_x': 5.9,'camera3_angle_y': -130.21,'camera3_focalLength': 26.1,
        'camera4_pos_x': -14.3,'camera4_pos_y': -4,'camera4_pos_z': 57.5,'camera4_angle_x': 5.5,'camera4_angle_y': 147.13,'camera4_focalLength': 14.99
        }

#kawawaki
kwargs = {'camera1_pos_x': -22.9,'camera1_pos_y': 0,'camera1_pos_z': -23.1,'camera1_angle_x': 0.4,'camera1_angle_y': 33.9,'camera1_focalLength': 16.7,
        'camera2_pos_x': -23.8,'camera2_pos_y': 0,'camera2_pos_z': 25.36,'camera2_angle_x': -2.4,'camera2_angle_y': 133.07,'camera2_focalLength': 12.7,
        'camera3_pos_x': 20.5,'camera3_pos_y': 0,'camera3_pos_z': -19.81,'camera3_angle_x': 5.71,'camera3_angle_y': -43.8,'camera3_focalLength': 16.1,
        'camera4_pos_x': -11.45,'camera4_pos_y': -4,'camera4_pos_z': 60,'camera4_angle_x': 2.8,'camera4_angle_y': 118.3,'camera4_focalLength': 10
        }
#kikuzumi
kwargs = {'camera1_pos_x': -14.7,'camera1_pos_y': -4,'camera1_pos_z': -55.9,'camera1_angle_x': 5,'camera1_angle_y': 19.7,'camera1_focalLength': 34.6,
        'camera2_pos_x': -57.28,'camera2_pos_y': -4,'camera2_pos_z': 21.7,'camera2_angle_x': 2.9,'camera2_angle_y': 113.47,'camera2_focalLength': 22,
        'camera3_pos_x': 55.21,'camera3_pos_y': -4,'camera3_pos_z': 19.1,'camera3_angle_x': 8.2,'camera3_angle_y': -113.3,'camera3_focalLength': 31.5,
        'camera4_pos_x': 11.9,'camera4_pos_y': -4,'camera4_pos_z': 54.7,'camera4_angle_x': 6.8,'camera4_angle_y': 209.06,'camera4_focalLength': 20.34
        }
#murakami
kwargs = {'camera1_pos_x': -11.99,'camera1_pos_y': 0,'camera1_pos_z': -60,'camera1_angle_x': 36.21,'camera1_angle_y': 29.26,'camera1_focalLength': 10,
        'camera2_pos_x': -59.98,'camera2_pos_y': 0,'camera2_pos_z': 18.28,'camera2_angle_x': 18.4,'camera2_angle_y': 118.9,'camera2_focalLength': 10,
        'camera3_pos_x': 60,'camera3_pos_y': -4,'camera3_pos_z': 20.2,'camera3_angle_x': 16.3,'camera3_angle_y': -162.6,'camera3_focalLength': 10,
        'camera4_pos_x': 11.6,'camera4_pos_y': -4,'camera4_pos_z': 60,'camera4_angle_x': 33.3,'camera4_angle_y': 228.37,'camera4_focalLength': 10
        }
#yamaguchi
kwargs = {'camera1_pos_x': -17.28,'camera1_pos_y': 0,'camera1_pos_z': 25.5,'camera1_angle_x': 24.8,'camera1_angle_y': 132.5,'camera1_focalLength': 20,
        'camera2_pos_x': -13.1,'camera2_pos_y': 0,'camera2_pos_z': -19,'camera2_angle_x': 37.95,'camera2_angle_y': 17.6,'camera2_focalLength': 10,
        'camera3_pos_x': 10.44,'camera3_pos_y': 0,'camera3_pos_z': 25.33,'camera3_angle_x': 25.01,'camera3_angle_y': -150.2,'camera3_focalLength': 20,
        'camera4_pos_x': 11.8,'camera4_pos_y': 0,'camera4_pos_z': -20.02,'camera4_angle_x': 31.5,'camera4_angle_y': -19.4,'camera4_focalLength': 10
        }

#Li
kwargs = {'camera1_pos_x': 20.6,'camera1_pos_y': -4,'camera1_pos_z': -22.64,'camera1_angle_x': 0,'camera1_angle_y': -35.7,'camera1_focalLength': 20,
        'camera2_pos_x': -22.3,'camera2_pos_y': -4,'camera2_pos_z': 21.6,'camera2_angle_x': 0,'camera2_angle_y': 124.3,'camera2_focalLength': 20,
        'camera3_pos_x': -19.5,'camera3_pos_y': -4,'camera3_pos_z': -14.9,'camera3_angle_x': 0,'camera3_angle_y': 46.8,'camera3_focalLength': 20,
        'camera4_pos_x': 16.1,'camera4_pos_y': -4,'camera4_pos_z': 31,'camera4_angle_x': 0,'camera4_angle_y': 217.4,'camera4_focalLength': 20
        }
"""


"""
if os.path.exists(os.path.join(save_dir, "train_x.pth")):
    train_x = torch.load(os.path.join(save_dir, "train_x.pth"))
    #train_x = torch.cat([train_x, torch.tensor([[0,0,-60,0,0,20,-60,0,0,0,90,20,60,0,0,0,-90,20,0,-4,60,0,180,20]])], dim=0)
else:
    train_x = torch.tensor([[0,0,-60,0,0,20,-60,0,0,0,90,20,60,0,0,0,-90,20,0,-4,60,0,180,20]])
if os.path.exists(os.path.join(save_dir, "train_y.pth")):
    train_y = torch.load(os.path.join(save_dir, "train_y.pth"))
else:
    train_y = torch.tensor([[]])
if os.path.exists(os.path.join(save_dir, "train_y_1.pth")):
    train_y_1 = torch.load(os.path.join(save_dir, "train_y_1.pth"))
else:
    train_y_1 = torch.tensor([[]])
if os.path.exists(os.path.join(save_dir, "train_y_2.pth")):
    train_y_2 = torch.load(os.path.join(save_dir, "train_y_2.pth"))
else:
    train_y_2 = torch.tensor([[]])
if os.path.exists(os.path.join(save_dir, "train_y_3.pth")):
    train_y_3 = torch.load(os.path.join(save_dir, "train_y_3.pth"))
else:
    train_y_3 = torch.tensor([[]])
if os.path.exists(os.path.join(save_dir, "train_y_4.pth")):
    train_y_4 = torch.load(os.path.join(save_dir, "train_y_4.pth"))
else:
    train_y_4 = torch.tensor([[]])
"""






"""
scaler_x = StandardScaler()
scaler_y = StandardScaler()
# x_multiとy_multiの標準化
x_multi = torch.tensor(scaler_x.fit_transform(x_multi), dtype=torch.float32)
y_multi = torch.tensor(scaler_y.fit_transform(y_multi), dtype=torch.float32)
"""

"""
input_transform = Normalize(d=24)
# TODO 変更箇所① nuの値
gp = SingleTaskGP(train_x, train_y, input_transform=input_transform,  covar_module=MaternKernel(nu=2.5))
gp_1 = SingleTaskGP(torch.cat([train_x[:, :6]] * 4, dim=1), train_y_1, input_transform=input_transform, covar_module=MaternKernel(nu=2.5))
gp_2 = SingleTaskGP(torch.cat([train_x[:, 6:12]] * 4, dim=1), train_y_2, input_transform=input_transform, covar_module=MaternKernel(nu=2.5))
gp_3 = SingleTaskGP(torch.cat([train_x[:, 12:18]] * 4, dim=1), train_y_3, input_transform=input_transform, covar_module=MaternKernel(nu=2.5))
gp_4 = SingleTaskGP(torch.cat([train_x[:, 18:]] * 4, dim=1), train_y_4, input_transform=input_transform, covar_module=MaternKernel(nu=2.5))
model = ModelListGP(gp, gp_1, gp_2, gp_3, gp_4)
mll = SumMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)
"""

"""
train_y = torch.tensor([[0.48378123635026093],[0.5099057614050118],[0.38811120832118207],[0.49696131801031135],[0.36051245740585036],[0.42181986323039106],[0.45015281957957115]])
train_y_1 = torch.tensor([[0.21452691061953],[0.14123480914395847],[0.17281573448661255],[0.19435323598029897],[0.09058171438686552],[0.18338565622642428],[0.210568062521871]])
train_y_2 = torch.tensor([[0.21452691061953],[0.14123480914395847],[0.17281573448661255],[0.19435323598029897],[0.09058171438686552],[0.18338565622642428],[0.210568062521871]])
train_y_3 = torch.tensor([[0.21452691061953],[0.14123480914395847],[0.17281573448661255],[0.19435323598029897],[0.09058171438686552],[0.18338565622642428],[0.210568062521871]])
train_y_4 = torch.tensor([[0.21452691061953],[0.14123480914395847],[0.17281573448661255],[0.19435323598029897],[0.09058171438686552],[0.18338565622642428],[0.210568062521871]])
"""

# for i in range(300):
#     # Set export directory
#     # bayes_util.GenerateScriptToSetExportDirectory( awsimProjectRootPath, "trial_"+str(i))
#     # Create a unity script that changes camera parameters (position/angle/focalLength)
#     # You need to specify all the parameters.
#     # print("Generating camera parameter script...")
#     print(str(i)+"回目")

#     bayes_util.GenerateScriptToChangeParameter(awsimProjectRootPath, **kwargs)
#     print("Done.")

#     volumeFile = awsimProjectRootPath + "/Assets/AWSIM/Prefabs/Environments/Nishishinjuku/Volume Profile.asset"
#     autowareFile = awsimProjectRootPath + "/Assets/AWSIM/Scenes/Main/AutowareSimulation.unity"

#     step1_time_list = []
#     step2_time_list = []
#     step3_time_list = []
#     single_score_array= [[[],[],[]] for _ in range(15)]
#     single_score_average_array = [[] for _ in range(4)]
#     score_array= [[] for _ in range(15)]
#     score_average_array = []
#     x_test_array = [[] for _ in range(15)]


#     for j in range(3):

#         if j == 0:
#             with open(volumeFile, "r") as file:
#                 lines = file.readlines()
#             with open(volumeFile, "w") as file:
#                 for k in range(len(lines)):
#                     if lines[k].strip() == "hdriSky:":
#                         if k + 1 < len(lines) and lines[k + 1].strip() == "m_OverrideState: 1":
#                             lines[k + 1] = "    m_OverrideState: 0\n"  # 書き換え
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "m_Value: {fileID: 8900000, guid: c9bc6cdb7a2e78b4c85ae74715b0a355, type: 3}":
#                             lines[k + 2] = "    m_Value: {fileID: 0}\n"  # 書き換え
#                     file.write(lines[k])
#             with open(autowareFile, "r") as file:
#                 lines = file.readlines()
#             with open(autowareFile, "w") as file:
#                 for k in range(len(lines)):
#                     if lines[k].strip() == "- target: {fileID: 9027057599863099216, guid: 58d73df60b244d146bdf5f5896f78355, type: 3}":
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "value: 23123.09":
#                             lines[k + 2] = "      value: 73123.09\n"  # 書き換え
#                     if lines[k].strip() == "- target: {fileID: 9027057599863099217, guid: 58d73df60b244d146bdf5f5896f78355, type: 3}":
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "value: 23123.09":
#                             lines[k + 2] = "      value: 73123.09\n"  # 書き換え
#                     file.write(lines[k])

#         elif j == 1:
#             with open(volumeFile, "r") as file:
#                 lines = file.readlines()
#             with open(volumeFile, "w") as file:
#                 for k in range(len(lines)):
#                     if lines[k].strip() == "meanFreePath:":
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "m_Value: 1000":
#                             lines[k + 2] = "    m_Value: 50\n"  # 書き換え
#                     file.write(lines[k])
#             with open(autowareFile, "r") as file:
#                 lines = file.readlines()
#             with open(autowareFile, "w") as file:
#                 for k in range(len(lines)):
#                     if lines[k].strip() == "- target: {fileID: 9027057599863099216, guid: 58d73df60b244d146bdf5f5896f78355, type: 3}":
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "value: 73123.09":
#                             lines[k + 2] = "      value: 33123.09\n"  # 書き換え
#                     if lines[k].strip() == "- target: {fileID: 9027057599863099217, guid: 58d73df60b244d146bdf5f5896f78355, type: 3}":
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "value: 73123.09":
#                             lines[k + 2] = "      value: 33123.09\n"  # 書き換え
#                     file.write(lines[k])
#             with open(autowareFile, "r") as file:
#                 lines = file.readlines()
#             with open(autowareFile, "w") as file:
#                 for k in range(len(lines)):
#                     if lines[k].strip() == "propertyPath: RainIntensity":
#                         if k + 1 < len(lines) and lines[k + 1].strip() == "value: 0":
#                             lines[k + 1] = "      value: 0.4\n"  # 書き換え
#                     file.write(lines[k])

#         elif j == 2:
#             with open(volumeFile, "r") as file:
#                 lines = file.readlines()
#             with open(volumeFile, "w") as file:
#                 for k in range(len(lines)):
#                     if lines[k].strip() == "meanFreePath:":
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "m_Value: 50":
#                             lines[k + 2] = "    m_Value: 1000\n"  # 書き換え
#                     file.write(lines[k])
#             with open(autowareFile, "r") as file:
#                 lines = file.readlines()
#             with open(autowareFile, "w") as file:
#                 for k in range(len(lines)):
#                     if lines[k].strip() == "propertyPath: RainIntensity":
#                         if k + 1 < len(lines) and lines[k + 1].strip() == "value: 0.4":
#                             lines[k + 1] = "      value: 0\n"  # 書き換え
#                     file.write(lines[k])
#             with open(autowareFile, "w") as file:
#                 for k in range(len(lines)):
#                     if lines[k].strip() == "- target: {fileID: 9027057599863099216, guid: 58d73df60b244d146bdf5f5896f78355, type: 3}":
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "value: 33123.09":
#                             lines[k + 2] = "      value: 23123.09\n"  # 書き換え
#                     if lines[k].strip() == "- target: {fileID: 9027057599863099217, guid: 58d73df60b244d146bdf5f5896f78355, type: 3}":
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "value: 33123.09":
#                             lines[k + 2] = "      value: 23123.09\n"  # 書き換え
#                     file.write(lines[k])
#             with open(volumeFile, "r") as file:
#                 lines = file.readlines()
#             with open(volumeFile, "w") as file:
#                 for k in range(len(lines)):
#                     if lines[k].strip() == "hdriSky:":
#                         if k + 1 < len(lines) and lines[k + 1].strip() == "m_OverrideState: 0":
#                             lines[k + 1] = "    m_OverrideState: 1\n"  # 書き換え
#                         if k + 2 < len(lines) and lines[k + 2].strip() == "m_Value: {fileID: 0}":
#                             lines[k + 2] = "    m_Value: {fileID: 8900000, guid: c9bc6cdb7a2e78b4c85ae74715b0a355, type: 3}\n"  # 書き換え
#                     file.write(lines[k])

#         # Build a player with the specified camera paraemters
#         # ref: https://docs.unity3d.com/ja/2018.4/Manual/CommandLineArguments.html
#         print("Building player...")
#         commandToBuildPlayer = [
#             unityBinaryPath,
#             "-quit",
#             #"-batchmode",
#             "-buildTarget",
#             "Linux64",
#             "-executeMethod",
#             "BuildPlayer.MyBuild",
#             "-projectPath",
#             awsimProjectRootPath
#             ]
#         subprocess.call(commandToBuildPlayer)
#         print("Done.")

#         start_time = time.time()
#         # Run the built player
#         print("Running player...")
#         # commandToRunPlayer =[awsimProjectRootPath+"/Linux/Player"]
#         commandToRunPlayer =["sudo", awsimProjectRootPath+"/Linux/Player"]
#         subprocess.call(commandToRunPlayer)
#         print("Done.")

#         path1 = awsimProjectRootPath + "/Assets/Addition/TrafficData0.txt"
#         path2 = awsimProjectRootPath + "/Assets/Addition/TrafficData1.txt"
#         path3 = awsimProjectRootPath + "/Assets/Addition/TrafficData2.txt"
#         path4 = awsimProjectRootPath + "/Assets/Addition/TrafficData3.txt"
#         path5 = awsimProjectRootPath + "/Assets/Addition/TrafficData4.txt"
#         path6 = awsimProjectRootPath + "/Assets/Addition/TrafficData5.txt"
#         path7 = awsimProjectRootPath + "/Assets/Addition/TrafficData6.txt"
#         path8 = awsimProjectRootPath + "/Assets/Addition/TrafficData7.txt"
#         path9 = awsimProjectRootPath + "/Assets/Addition/TrafficData8.txt"

#         path10 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages1"
#         path11 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages2"
#         path12 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages3"
#         path13 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages4"
#         path14 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages5"
#         path15 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages6"
#         path16 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages7"
#         path17 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages8"

#         path18 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_1.txt"
#         path19 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_2.txt"
#         path20 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_3.txt"
#         path21 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_4.txt"
#         path22 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_5.txt"
#         path23 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_6.txt"
#         path24 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_7.txt"
#         path25 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_8.txt"

#         path26 = awsimProjectRootPath + "/Assets/Addition/CapturedImages"
#         path27 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix"

#         pathes = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, path14, path15, path16, path17, path18, path19, path20, path21, path22, path23, path24, path25, path26, path27]
#         pathes2 = [path26, path27, path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, path14, path15, path16, path17, path18, path19, path20, path21, path22, path23, path24, path25]


#         for path in pathes2:
#             subprocess.call(["sudo", "chown", "fujii:fujii", path])

#         #video creation
#         print("Creating video...")
#         image_to_video.image_to_video()
#         print("Done.")

#         for path in pathes:
#             subprocess.call(["sudo", "rm", "-r", path])

#         step1_time = time.time() - start_time
#         step1_time_list.append(step1_time)
#         start_time = time.time()

#         args = parse_args("Express MTMC: run MOT on all cameras and then MTMC.")
#         cfg = get_cfg_defaults()

#         if args.config:
#             cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
#         cfg = expand_relative_paths(cfg)
#         if j==0:
#             cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, now, "epoch"+str(i), "sunny")
#         elif j==1:
#             cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, now, "epoch"+str(i), "rainy")
#         elif j==2:
#             cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, now, "epoch"+str(i), "evening")
#         cfg.freeze()

#         if not os.path.exists(cfg.OUTPUT_DIR):
#             os.makedirs(cfg.OUTPUT_DIR)
#         if not global_checks["OUTPUT_DIR"](cfg.OUTPUT_DIR):
#             log.error(
#                 "Invalid param value in: OUTPUT_DIR. Provide an absolute path to a directory, whose parent exists.")
#             sys.exit(2)

#         log.log_init(os.path.join(cfg.OUTPUT_DIR, args.log_filename),
#                     args.log_level, not args.no_log_stdout)

#         MTMC_OUTPUT_NAME = "mtmc"
#         """Run Express MTMC on a given config."""
#         mot_configs = []
#         cam_names, cam_dirs = [], []
#         for cam_idx, cam_info in enumerate(cfg.EXPRESS.CAMERAS):
#             cam_cfg = cfg.clone()
#             cam_cfg.defrost()
#             for key, val in cam_info.items():
#                 setattr(cam_cfg.MOT, key.upper(), val)

#             cam_video_name = os.path.split(cam_cfg.MOT.VIDEO)[1].split(".")[0]
#             cam_names.append(cam_video_name)

#             # set output dir of MOT to a unique folder under the root OUTPUT_DIR
#             cam_dir = os.path.join(cfg.OUTPUT_DIR, f"{cam_idx}_{cam_video_name}")
#             cam_dirs.append(cam_dir)
#             cam_cfg.OUTPUT_DIR = cam_dir
#             if len(cfg.EVAL.GROUND_TRUTHS) == len(cfg.EXPRESS.CAMERAS):
#                 cam_cfg.EVAL.GROUND_TRUTHS = [cfg.EVAL.GROUND_TRUTHS[cam_idx]]
#             cam_cfg.freeze()
#             mot_configs.append(cam_cfg)

#         single_score_list = []
#         # run MOT in all cameras
#         for mot_conf in mot_configs:
#             _, single_score = run_mot(mot_conf)
#             single_score_list.append(single_score)
#         print(f"single_score: {single_score_list}")
#         top_6_ids = sorted(range(len(single_score_list)), key=lambda j: single_score_list[j], reverse=True)[:6]
#         print(f"top_6_ids: {top_6_ids}")
#         pickle_paths = [os.path.join(path, f"{MOT_OUTPUT_NAME}.pkl") for path in cam_dirs]

#         step2_time = time.time() - start_time
#         step2_time_list.append(step2_time)
#         start_time = time.time()

#         log.info("Express: Running MOT on all cameras finished. Running MTMC...")

#         if j == 0:
#             combinations = list(itertools.combinations(top_6_ids, 4))
#         print(f"combinations: {combinations}")

#         for k, combo in enumerate(combinations):
#             #combo = tuple(random.sample(combo, len(combo)))
#             print(f"combo: {combo}")

#             combostr = "".join([str(j) for j in combo])

#             mtmc_cfg = cfg.clone()
#             mtmc_cfg.defrost()
#             mtmc_cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "combo"+ combostr)
#             mtmc_cfg.EXPRESS.CAMERAS = [cfg.EXPRESS.CAMERAS[j] for j in combo]
#             mtmc_cfg.EVAL.GROUND_TRUTHS = [cfg.EVAL.GROUND_TRUTHS[j] for j in combo]
#             mtmc_cfg.MTMC.PICKLED_TRACKLETS = [pickle_paths[j] for j in combo]
#             mtmc_cfg.freeze()

#             mot_configs_combo = [mot_configs[j] for j in combo]

#             all_ground_truths_empty = all(all(os.path.getsize(gt) == 0 for gt in mot_conf.EVAL.GROUND_TRUTHS) for mot_conf in mot_configs_combo)
#             all_predictions_empty = all(all(os.path.getsize(pred) == 0 for pred in mot_conf.EVAL.PREDICTIONS) for mot_conf in mot_configs_combo)
#             if all_ground_truths_empty or all_predictions_empty:
#                 score = 0
#                 score_array[k].append(score)
#                 print(f"score: {score}")
#                 print(f"score_array: {score_array}")
#             else:
#                 mtracks = run_mtmc(mtmc_cfg)
#                 log.info("Express: Running MTMC on all cameras finished. Saving final results ...")

#                 cam_combo_dirs = [os.path.join(mtmc_cfg.OUTPUT_DIR, f"{j}_{cam_names[j]}") for j in combo]
#                 for directory in cam_combo_dirs:
#                     os.makedirs(directory, exist_ok=True)

#                 # save single cam tracks
#                 final_pickle_paths = [os.path.join(d, f"{MTMC_OUTPUT_NAME}.pkl") for d in cam_combo_dirs]
#                 final_csv_paths = [os.path.join(d, f"{MTMC_OUTPUT_NAME}.csv") for d in cam_combo_dirs]
#                 final_txt_paths = [os.path.join(d, f"{MTMC_OUTPUT_NAME}.txt") for d in cam_combo_dirs]
#                 save_tracklets_per_cam(mtracks, final_pickle_paths)
#                 save_tracklets_txt_per_cam(mtracks, final_txt_paths)
#                 save_tracklets_csv_per_cam(mtracks, final_csv_paths)


#                 """
#                 if cfg.EXPRESS.FINAL_VIDEO_OUTPUT:
#                     for j, cam_dir in enumerate(cam_combo_dirs):
#                         video_in = mot_configs[combo[j]].MOT.VIDEO
#                         video_ext = video_in.split(".")[1]
#                         video_out = os.path.join(cam_dir, f"{MTMC_OUTPUT_NAME}.{video_ext}")
#                         annotate_video_mtmc(video_in, video_out, mtracks, j, font=cfg.FONT, fontsize=cfg.FONTSIZE)
#                         log.info(f"Express: video {j} saved.")
#                 """


#                 if len(cfg.EVAL.GROUND_TRUTHS) == 0:
#                     log.info("Ground truths are not provided for evaluation, terminating.")

#                 log.info("Ground truth annotations are provided, trying to evaluate MTMC ...")
#                 if len(cfg.EVAL.GROUND_TRUTHS) != len(cam_names):
#                     log.error(
#                         "Number of ground truth files != number of cameras, aborting evaluation ...")

#                 mtmc_cfg.defrost()
#                 mtmc_cfg.EVAL.PREDICTIONS = final_txt_paths
#                 mtmc_cfg.freeze()
#                 score = run_evaluation(mtmc_cfg)
#                 score_array[k].append(score)
#                 print(f"score: {score}")
#                 print(f"score_array: {score_array}")

#             single_score_array[k][j].append(single_score_list[combo[0]] if isinstance(single_score_list, list) else 0)
#             single_score_array[k][j].append(single_score_list[combo[1]] if isinstance(single_score_list, list) else 0)
#             single_score_array[k][j].append(single_score_list[combo[2]] if isinstance(single_score_list, list) else 0)
#             single_score_array[k][j].append(single_score_list[combo[3]] if isinstance(single_score_list, list) else 0)
#             print(f"single_score_array: {single_score_array}")

#             for l,m in enumerate(combo):
#                 if l==0:
#                     if m==0: x_test_array[k][:6] = x_suggest[-1][:6]
#                     elif m==1: x_test_array[k][:6] = x_suggest[-1][6:12]
#                     elif m==2: x_test_array[k][:6] = x_suggest[-1][12:18]
#                     elif m==3: x_test_array[k][:6] = x_suggest[-1][18:]
#                     elif m==4: x_test_array[k][:6] = x_suggest[-2][:6]
#                     elif m==5: x_test_array[k][:6] = x_suggest[-2][6:12]
#                     elif m==6: x_test_array[k][:6] = x_suggest[-2][12:18]
#                     elif m==7: x_test_array[k][:6] = x_suggest[-2][18:]
#                 elif l==1:
#                     if m==0: x_test_array[k][6:12] = x_suggest[-1][:6]
#                     elif m==1: x_test_array[k][6:12] = x_suggest[-1][6:12]
#                     elif m==2: x_test_array[k][6:12] = x_suggest[-1][12:18]
#                     elif m==3: x_test_array[k][6:12] = x_suggest[-1][18:]
#                     elif m==4: x_test_array[k][6:12] = x_suggest[-2][:6]
#                     elif m==5: x_test_array[k][6:12] = x_suggest[-2][6:12]
#                     elif m==6: x_test_array[k][6:12] = x_suggest[-2][12:18]
#                     elif m==7: x_test_array[k][6:12] = x_suggest[-2][18:]
#                 elif l==2:
#                     if m==0: x_test_array[k][12:18] = x_suggest[-1][:6]
#                     elif m==1: x_test_array[k][12:18] = x_suggest[-1][6:12]
#                     elif m==2: x_test_array[k][12:18] = x_suggest[-1][12:18]
#                     elif m==3: x_test_array[k][12:18] = x_suggest[-1][18:]
#                     elif m==4: x_test_array[k][12:18] = x_suggest[-2][:6]
#                     elif m==5: x_test_array[k][12:18] = x_suggest[-2][6:12]
#                     elif m==6: x_test_array[k][12:18] = x_suggest[-2][12:18]
#                     elif m==7: x_test_array[k][12:18] = x_suggest[-2][18:]
#                 elif l==3:
#                     if m==0: x_test_array[k][18:] = x_suggest[-1][:6]
#                     elif m==1: x_test_array[k][18:] = x_suggest[-1][6:12]
#                     elif m==2: x_test_array[k][18:] = x_suggest[-1][12:18]
#                     elif m==3: x_test_array[k][18:] = x_suggest[-1][18:]
#                     elif m==4: x_test_array[k][18:] = x_suggest[-2][:6]
#                     elif m==5: x_test_array[k][18:] = x_suggest[-2][6:12]
#                     elif m==6: x_test_array[k][18:] = x_suggest[-2][12:18]
#                     elif m==7: x_test_array[k][18:] = x_suggest[-2][18:]

#         step3_time = time.time() - start_time
#         step3_time_list.append(step3_time)


#     start_time = time.time()

#     for q, combo in enumerate(combinations):
#         with open(os.path.join(save_dir, now, "x_test.txt"), "a") as f:
#             x_test = x_test_array[q]
#             x_test_values = [x.item() if isinstance(x, torch.Tensor) else x for x in x_test]
#             f.write(str(i+1)+ "," + ",".join(map(str, combo)) + "," +",".join(map(str, x_test_values)))
#             f.write("\n")

#     train_x = torch.cat((train_x, torch.tensor(x_test_array)), dim=0) if train_x.size(1) > 0 else torch.tensor(x_test_array)
#     print(f"train_x.shape: {train_x.shape}")
#     print(f"train_x: {train_x}")


#     score_average_array = [
#         sum(scores) / len(scores) if len(scores) > 0 else 0
#         for scores in score_array]

#     for q, combo in enumerate(combinations):
#         with open(os.path.join(save_dir, now, "score.txt"), "a") as f:
#             score_list = score_array[q]
#             score_avarage = score_average_array[q]
#             f.write(str(i+1)+ "," + ",".join(map(str, combo)) + "," +",".join(map(str, score_list))+","+str(score_avarage))
#             f.write("\n")

#     train_y = torch.cat((train_y, torch.tensor(score_average_array).float().view(15, 1)), dim=0) if train_y.size(1) > 0 else torch.tensor(score_average_array).float().view(15, 1)
#     print(f"train_y.shape: {train_y.shape}")
#     print(f"train_y: {train_y}")


#     for q in range(15):
#         for p in range(4):
#             # 各 15 の要素の「3つの値」の平均を計算
#             avg = sum(single_score_array[q][r][p] for r in range(3)) / 3
#             single_score_average_array[p].append(avg)
#     print(f"single_score_average_array: {single_score_average_array}")


#     for q, combo in enumerate(combinations):
#         with open(os.path.join(save_dir, now, "single_score.txt"), "a") as f:
#             flattened_single_score_array = [item for sublist in single_score_array[q] for item in sublist]
#             print(f"flattened_single_score_array: {flattened_single_score_array}")
#             f.write(str(i+1)+ "," + ",".join(map(str, combo)) + "," +",".join(map(str, flattened_single_score_array)))
#             f.write("\n")
#         with open(os.path.join(save_dir, now, "single_score_average.txt"), "a") as f:
#             f.write(str(i+1)+ "," + ",".join(map(str, combo)) + ",")
#             row = [single_score_average_array[0][q],single_score_average_array[1][q],single_score_average_array[2][q],single_score_average_array[3][q]]
#             f.write(",".join(map(str, row)))
#             f.write("\n")

#     train_y_1 = torch.cat((train_y_1, torch.tensor(single_score_average_array[0]).float().view(15, 1)), dim=0) if train_y_1.size(1) > 0 else torch.tensor(single_score_average_array[0]).float().view(15, 1)
#     train_y_2 = torch.cat((train_y_2, torch.tensor(single_score_average_array[1]).float().view(15, 1)), dim=0) if train_y_2.size(1) > 0 else torch.tensor(single_score_average_array[1]).float().view(15, 1)
#     train_y_3 = torch.cat((train_y_3, torch.tensor(single_score_average_array[2]).float().view(15, 1)), dim=0) if train_y_3.size(1) > 0 else torch.tensor(single_score_average_array[2]).float().view(15, 1)
#     train_y_4 = torch.cat((train_y_4, torch.tensor(single_score_average_array[3]).float().view(15, 1)), dim=0) if train_y_4.size(1) > 0 else torch.tensor(single_score_average_array[3]).float().view(15, 1)
#     print(f"train_y_1.shape: {train_y_1.shape}")
#     print(f"train_y_2.shape: {train_y_2.shape}")
#     print(f"train_y_3.shape: {train_y_3.shape}")
#     print(f"train_y_4.shape: {train_y_4.shape}")
#     print(f"train_y_1: {train_y_1}")

#     #shutil.copytree(os.path.join(os.getcwd(), "datasets"), cfg.OUTPUT_DIR, dirs_exist_ok=True)

#     step3_time = time.time() - start_time
#     step3_time_list.append(step3_time)
#     start_time = time.time()


#     bounds_min = torch.tensor([-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0,-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0])
#     bounds_max = torch.tensor([60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0,60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0])
#     train_x_scaled = (train_x - bounds_min) / (bounds_max - bounds_min)
#     print(f"train_x_scaled.shape: {train_x_scaled.shape}")
#     print(f"train_x_scaled: {train_x_scaled}")

#     i0 = torch.zeros(train_x.shape[0],1)
#     i1 = torch.ones(train_x.shape[0],1)
#     i2 = torch.ones(train_x.shape[0],1)*2
#     i3 = torch.ones(train_x.shape[0],1)*3
#     i4 = torch.ones(train_x.shape[0],1)*4
#     print(f"i0.shape: {i0.shape}")

#     x_multi = torch.cat([
#         torch.cat([train_x_scaled, i0], dim=-1),
#         torch.cat([torch.cat([train_x_scaled[:, :6]]*4,dim=1), i1], dim=-1),
#         torch.cat([torch.cat([train_x_scaled[:, 6:12]]*4, dim=1), i2], dim=-1),
#         torch.cat([torch.cat([train_x_scaled[:, 12:18]]*4, dim=1), i3], dim=-1),
#         torch.cat([torch.cat([train_x_scaled[:, 18:]]*4, dim=1), i4], dim=-1),
#         ], dim=0)
#     y_multi = torch.cat([train_y, train_y_1, train_y_2, train_y_3, train_y_4])
#     print(f"x_multi.shape: {x_multi.shape}")
#     print(f"y_multi.shape: {y_multi.shape}")

#     with gpytorch.settings.cholesky_jitter(1e-3):
#         dim = x_multi.shape[-1] - 1  # 入力次元数（タスク情報を除く）
#         #print(f"dim: {dim}")
#         #print("x_multi shape:", x_multi.shape)  # 入力データの形状
#         #print("y_multi shape:", y_multi.shape)  # 出力データの形状
#         mu_0 = 0.0  # LogNormalPriorのmuパラメータ
#         sigma_0 = 1.0  # LogNormalPriorのsigmaパラメータ

#         model=MultiTaskGP(x_multi, y_multi, task_feature=-1)
#         model.likelihood.noise_covar.noise = torch.tensor(1e-3, requires_grad=True)
#         mll = ExactMarginalLogLikelihood(model.likelihood, model)
#         fit_gpytorch_model(mll)

#         # TODO 変更箇所② num_samplesの値
#         ref_point=torch.tensor([0,0,0,0,0])
#         pareto_y = torch.tensor([[0.5,0.15,0.15,0.15,0.15]])
#         partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_y)

#         weights = torch.tensor([0.4, 0.15, 0.15, 0.15, 0.15])
#         objective = WeightedMCMultiOutputObjective(weights=weights)
#         sampler = SobolQMCNormalSampler(num_samples=3000)

#         qEHVI = qExpectedHypervolumeImprovement(
#             model=model,
#             ref_point=ref_point,
#             objective=objective,
#             partitioning=partitioning,
#             sampler=sampler,
#         )

#         """
#         bounds = torch.tensor([
#             [-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0],
#             [60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0]
#         ], dtype=torch.float32)
#         """

#         bounds = torch.stack([torch.zeros(24, dtype=torch.float32), torch.ones(24, dtype=torch.float32)])

#         # TODO 変更箇所③ num_restartsの値
#         num_restarts  = 100
#         #scales = [120, 4, 120, 30, 360, 20, 120, 4, 120, 30, 360, 20, 120, 4, 120, 30, 360, 20, 120, 4, 120, 30, 360, 20]
#         callable_object_standard = CallableObject_standard()

#         batch_initial_conditions = torch.empty(0, 1, 24, requires_grad=True)

#         while batch_initial_conditions.size(0) < num_restarts:
#             batch_initial_condition = generate_initial_condition()
#             if callable_object_standard(batch_initial_condition) >= torch.tensor(0):
#                 # 条件を満たす場合、batch_initial_conditionsに追加
#                 if batch_initial_conditions.size(0) == 0:
#                     batch_initial_conditions = batch_initial_condition
#                 else:
#                     batch_initial_conditions = torch.cat((batch_initial_conditions, batch_initial_condition), dim=0)

#         """
#         # 初期条件の設定
#         initial_condition = train_x_scaled[-1].view(1, 1, 24).clone().detach().requires_grad_(True)
#         # バッチ初期条件の初期化
#         batch_initial_conditions = torch.empty(0, 1, 24, requires_grad=True)
#         # 条件を満たすまで繰り返す
#         while batch_initial_conditions.size(0) < num_restarts:
#             noise = torch.randn_like(initial_condition)
#             # TODO 変更箇所④ ratioの値
#             ratio=0.1
#             for d in range(24):
#                 #noise[:, :, d] = noise[:, :, d] * scales[d] * ratio
#                 noise[:, :, d] = noise[:, :, d] * ratio
#             noise.requires_grad_(True)
#             noisy_initial_condition = initial_condition + noise

#             if callable_object(noisy_initial_condition) >= torch.tensor(0):
#                 # 条件を満たす場合、batch_initial_conditionsに追加
#                 if batch_initial_conditions.size(0) == 0:
#                     batch_initial_conditions = noisy_initial_condition
#                     print(f'noisy_initial_condition: {noisy_initial_condition}')
#                 else:
#                     batch_initial_conditions = torch.cat((batch_initial_conditions, noisy_initial_condition), dim=0)
#         """
#         #batch_initial_conditions = (batch_initial_conditions - bounds_min) / (bounds_max - bounds_min)


#         x_next, acq_value = optimize_acqf(
#             qEHVI,
#             bounds=bounds,
#             q=1,
#             num_restarts=num_restarts,
#             nonlinear_inequality_constraints=[CallableObject_standard()],
#             batch_initial_conditions=batch_initial_conditions,
#         )

#     x_next = x_next * (bounds_max - bounds_min) + bounds_min

#     x_suggest = torch.cat((x_suggest, x_next))
#     print(f'次の候補点は{x_next}です。')
#     with open(os.path.join(save_dir, now, "x_next.txt"), "a") as f:
#         f.write(str(i+1)+",")
#         for x in x_next:
#             f.write(str(x)+",")
#         f.seek(f.tell()-1, os.SEEK_SET)
#         f.truncate()
#         f.write("\n")

#     torch.save(model.state_dict(), os.path.join(save_dir, now, "epoch"+str(i), f'model_checkpoint_{i}.pth'))
#     torch.save(x_suggest, os.path.join(save_dir, now, "epoch"+str(i), f'x_suggest_{i}.pth'))
#     torch.save(train_x, os.path.join(save_dir, now, "epoch"+str(i), f'train_x_{i}.pth'))
#     torch.save(train_y, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_{i}.pth'))
#     torch.save(train_y_1, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_1_{i}.pth'))
#     torch.save(train_y_2, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_2_{i}.pth'))
#     torch.save(train_y_3, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_3_{i}.pth'))
#     torch.save(train_y_4, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_4_{i}.pth'))

#     kwargs = {'camera1_pos_x':x_suggest[-1][0],'camera1_pos_y':x_suggest[-1][1],'camera1_pos_z':x_suggest[-1][2],'camera1_angle_x':x_suggest[-1][3],'camera1_angle_y':x_suggest[-1][4],'camera1_focalLength':x_suggest[-1][5],
#         'camera2_pos_x':x_suggest[-1][6],'camera2_pos_y':x_suggest[-1][7],'camera2_pos_z':x_suggest[-1][8],'camera2_angle_x':x_suggest[-1][9],'camera2_angle_y':x_suggest[-1][10],'camera2_focalLength':x_suggest[-1][11],
#         'camera3_pos_x':x_suggest[-1][12],'camera3_pos_y':x_suggest[-1][13],'camera3_pos_z':x_suggest[-1][14],'camera3_angle_x':x_suggest[-1][15],'camera3_angle_y':x_suggest[-1][16],'camera3_focalLength':x_suggest[-1][17],
#         'camera4_pos_x':x_suggest[-1][18],'camera4_pos_y':x_suggest[-1][19],'camera4_pos_z':x_suggest[-1][20],'camera4_angle_x':x_suggest[-1][21],'camera4_angle_y':x_suggest[-1][22],'camera4_focalLength':x_suggest[-1][23],
#         'camera5_pos_x':x_suggest[-2][0],'camera5_pos_y':x_suggest[-2][1],'camera5_pos_z':x_suggest[-2][2],'camera5_angle_x':x_suggest[-2][3],'camera5_angle_y':x_suggest[-2][4],'camera5_focalLength':x_suggest[-2][5],
#         'camera6_pos_x':x_suggest[-2][6],'camera6_pos_y':x_suggest[-2][7],'camera6_pos_z':x_suggest[-2][8],'camera6_angle_x':x_suggest[-2][9],'camera6_angle_y':x_suggest[-2][10],'camera6_focalLength':x_suggest[-2][11],
#         'camera7_pos_x':x_suggest[-2][12],'camera7_pos_y':x_suggest[-2][13],'camera7_pos_z':x_suggest[-2][14],'camera7_angle_x':x_suggest[-2][15],'camera7_angle_y':x_suggest[-2][16],'camera7_focalLength':x_suggest[-2][17],
#         'camera8_pos_x':x_suggest[-2][18],'camera8_pos_y':x_suggest[-2][19],'camera8_pos_z':x_suggest[-2][20],'camera8_angle_x':x_suggest[-2][21],'camera8_angle_y':x_suggest[-2][22],'camera8_focalLength':x_suggest[-2][23]
#         }

#     # Reset script to its original contents
#     print("Resetting scripts...")
#     bayes_util.ResetScripts(awsimProjectRootPath)
#     print("Done.")

#     step4_time = time.time() - start_time

#     with open(output_file, "a") as f:
#         for j in range(3):
#             f.write(str(step1_time_list[j])+","+str(step2_time_list[j])+","+str(step3_time_list[j])+",")
#         f.write(str(step3_time_list[3])+",")
#         f.write(str(step4_time)+"\n")



#ランダム配置
for i in range(300):
    # Set export directory
    # bayes_util.GenerateScriptToSetExportDirectory( awsimProjectRootPath, "trial_"+str(i))
    # Create a unity script that changes camera parameters (position/angle/focalLength)
    # You need to specify all the parameters.
    # print("Generating camera parameter script...")
    print(str(i)+"回目")

    bayes_util.GenerateScriptToChangeParameter(awsimProjectRootPath, **kwargs)
    print("Done.")

    volumeFile = awsimProjectRootPath + "/Assets/AWSIM/Prefabs/Environments/Nishishinjuku/Volume Profile.asset"
    autowareFile = awsimProjectRootPath + "/Assets/AWSIM/Scenes/Main/AutowareSimulation.unity"

    step1_time_list = []
    step2_time_list = []
    step3_time_list = []
    single_score_array= [[[]] for _ in range(15)]
    single_score_average_array = [[] for _ in range(4)]
    score_array= [[] for _ in range(15)]
    score_average_array = []
    x_test_array = [[] for _ in range(15)]

    for j in range(1):
        # Build a player with the specified camera paraemters
        # ref: https://docs.unity3d.com/ja/2018.4/Manual/CommandLineArguments.html
        print("Building player...")
        commandToBuildPlayer = [
            unityBinaryPath,
            "-quit",
            #"-batchmode",
            "-buildTarget",
            "Linux64",
            "-executeMethod",
            "BuildPlayer.MyBuild",
            "-projectPath",
            awsimProjectRootPath
            ]
        subprocess.call(commandToBuildPlayer)
        print("Done.")

        start_time = time.time()
        # Run the built player
        print("Running player...")
        # commandToRunPlayer =[awsimProjectRootPath+"/Linux/Player"]
        commandToRunPlayer =["sudo", awsimProjectRootPath+"/Linux/Player"]
        subprocess.call(commandToRunPlayer)
        print("Done.")

        path1 = awsimProjectRootPath + "/Assets/Addition/TrafficData0.txt"
        path2 = awsimProjectRootPath + "/Assets/Addition/TrafficData1.txt"
        path3 = awsimProjectRootPath + "/Assets/Addition/TrafficData2.txt"
        path4 = awsimProjectRootPath + "/Assets/Addition/TrafficData3.txt"
        path5 = awsimProjectRootPath + "/Assets/Addition/TrafficData4.txt"
        path6 = awsimProjectRootPath + "/Assets/Addition/TrafficData5.txt"
        path7 = awsimProjectRootPath + "/Assets/Addition/TrafficData6.txt"
        path8 = awsimProjectRootPath + "/Assets/Addition/TrafficData7.txt"
        path9 = awsimProjectRootPath + "/Assets/Addition/TrafficData8.txt"

        path10 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages1"
        path11 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages2"
        path12 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages3"
        path13 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages4"
        path14 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages5"
        path15 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages6"
        path16 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages7"
        path17 = awsimProjectRootPath + "/Assets/Addition/CapturedImages/CapturedImages8"

        path18 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_1.txt"
        path19 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_2.txt"
        path20 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_3.txt"
        path21 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_4.txt"
        path22 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_5.txt"
        path23 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_6.txt"
        path24 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_7.txt"
        path25 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix/camera_matrix_inverse_8.txt"

        path26 = awsimProjectRootPath + "/Assets/Addition/CapturedImages"
        path27 = awsimProjectRootPath + "/Assets/Addition/SavedMatrix"

        pathes = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, path14, path15, path16, path17, path18, path19, path20, path21, path22, path23, path24, path25, path26, path27]
        pathes2 = [path26, path27, path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, path14, path15, path16, path17, path18, path19, path20, path21, path22, path23, path24, path25]

        for path in pathes2:
            subprocess.call(["sudo", "chown", "fujii:fujii", path])

        #video creation
        print("Creating video...")
        image_to_video.image_to_video()
        print("Done.")

        for path in pathes:
            subprocess.call(["sudo", "rm", "-r", path])

        step1_time = time.time() - start_time
        step1_time_list.append(step1_time)
        start_time = time.time()

        args = parse_args("Express MTMC: run MOT on all cameras and then MTMC.")
        cfg = get_cfg_defaults()

        if args.config:
            cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
        cfg = expand_relative_paths(cfg)
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, now, "epoch"+str(i))
        cfg.freeze()

        if not os.path.exists(cfg.OUTPUT_DIR):
            os.makedirs(cfg.OUTPUT_DIR)
        if not global_checks["OUTPUT_DIR"](cfg.OUTPUT_DIR):
            log.error(
                "Invalid param value in: OUTPUT_DIR. Provide an absolute path to a directory, whose parent exists.")
            sys.exit(2)

        log.log_init(os.path.join(cfg.OUTPUT_DIR, args.log_filename),
                    args.log_level, not args.no_log_stdout)

        MTMC_OUTPUT_NAME = "mtmc"
        mot_configs = []
        cam_names, cam_dirs = [], []
        for cam_idx, cam_info in enumerate(cfg.EXPRESS.CAMERAS):
            cam_cfg = cfg.clone()
            cam_cfg.defrost()
            for key, val in cam_info.items():
                setattr(cam_cfg.MOT, key.upper(), val)

            cam_video_name = os.path.split(cam_cfg.MOT.VIDEO)[1].split(".")[0]
            cam_names.append(cam_video_name)

            # set output dir of MOT to a unique folder under the root OUTPUT_DIR
            cam_dir = os.path.join(cfg.OUTPUT_DIR, f"{cam_idx}_{cam_video_name}")
            cam_dirs.append(cam_dir)
            cam_cfg.OUTPUT_DIR = cam_dir
            if len(cfg.EVAL.GROUND_TRUTHS) == len(cfg.EXPRESS.CAMERAS):
                cam_cfg.EVAL.GROUND_TRUTHS = [cfg.EVAL.GROUND_TRUTHS[cam_idx]]
            cam_cfg.freeze()
            mot_configs.append(cam_cfg)

        single_score_list = []
        # run MOT in all cameras
        for mot_conf in mot_configs:
            _, single_score = run_mot(mot_conf)
            single_score_list.append(single_score)
        print(f"single_score: {single_score_list}")
        top_6_ids = sorted(range(len(single_score_list)), key=lambda j: single_score_list[j], reverse=True)[:6]
        print(f"top_6_ids: {top_6_ids}")
        pickle_paths = [os.path.join(path, f"{MOT_OUTPUT_NAME}.pkl") for path in cam_dirs]

        step2_time = time.time() - start_time
        step2_time_list.append(step2_time)
        start_time = time.time()

        log.info("Express: Running MOT on all cameras finished. Running MTMC...")

        if j == 0:
            combinations = list(itertools.combinations(top_6_ids, 4))
        print(f"combinations: {combinations}")

        for k, combo in enumerate(combinations):
            #combo = tuple(random.sample(combo, len(combo)))
            print(f"combo: {combo}")

            combostr = "".join([str(j) for j in combo])

            mtmc_cfg = cfg.clone()
            mtmc_cfg.defrost()
            mtmc_cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "combo"+ combostr)
            mtmc_cfg.EXPRESS.CAMERAS = [cfg.EXPRESS.CAMERAS[j] for j in combo]
            mtmc_cfg.EVAL.GROUND_TRUTHS = [cfg.EVAL.GROUND_TRUTHS[j] for j in combo]
            mtmc_cfg.MTMC.PICKLED_TRACKLETS = [pickle_paths[j] for j in combo]
            mtmc_cfg.freeze()

            mot_configs_combo = [mot_configs[j] for j in combo]

            all_ground_truths_empty = all(all(os.path.getsize(gt) == 0 for gt in mot_conf.EVAL.GROUND_TRUTHS) for mot_conf in mot_configs_combo)
            all_predictions_empty = all(all(os.path.getsize(pred) == 0 for pred in mot_conf.EVAL.PREDICTIONS) for mot_conf in mot_configs_combo)
            if all_ground_truths_empty or all_predictions_empty:
                score = 0
                score_array[k].append(score)
                print(f"score: {score}")
                print(f"score_array: {score_array}")
            else:
                mtracks = run_mtmc(mtmc_cfg)
                log.info("Express: Running MTMC on all cameras finished. Saving final results ...")

                cam_combo_dirs = [os.path.join(mtmc_cfg.OUTPUT_DIR, f"{j}_{cam_names[j]}") for j in combo]
                for directory in cam_combo_dirs:
                    os.makedirs(directory, exist_ok=True)

                # save single cam tracks
                final_pickle_paths = [os.path.join(d, f"{MTMC_OUTPUT_NAME}.pkl") for d in cam_combo_dirs]
                final_csv_paths = [os.path.join(d, f"{MTMC_OUTPUT_NAME}.csv") for d in cam_combo_dirs]
                final_txt_paths = [os.path.join(d, f"{MTMC_OUTPUT_NAME}.txt") for d in cam_combo_dirs]
                save_tracklets_per_cam(mtracks, final_pickle_paths)
                save_tracklets_txt_per_cam(mtracks, final_txt_paths)
                save_tracklets_csv_per_cam(mtracks, final_csv_paths)

                if len(cfg.EVAL.GROUND_TRUTHS) == 0:
                    log.info("Ground truths are not provided for evaluation, terminating.")

                log.info("Ground truth annotations are provided, trying to evaluate MTMC ...")
                if len(cfg.EVAL.GROUND_TRUTHS) != len(cam_names):
                    log.error(
                        "Number of ground truth files != number of cameras, aborting evaluation ...")

                mtmc_cfg.defrost()
                mtmc_cfg.EVAL.PREDICTIONS = final_txt_paths
                mtmc_cfg.freeze()
                score = run_evaluation(mtmc_cfg)
                score_array[k].append(score)
                print(f"score: {score}")
                print(f"score_array: {score_array}")

            single_score_array[k][j].append(single_score_list[combo[0]] if isinstance(single_score_list, list) else 0)
            single_score_array[k][j].append(single_score_list[combo[1]] if isinstance(single_score_list, list) else 0)
            single_score_array[k][j].append(single_score_list[combo[2]] if isinstance(single_score_list, list) else 0)
            single_score_array[k][j].append(single_score_list[combo[3]] if isinstance(single_score_list, list) else 0)
            print(f"single_score_array: {single_score_array}")

            for l,m in enumerate(combo):
                if l==0:
                    if m==0: x_test_array[k][:6] = x_suggest[-1][:6]
                    elif m==1: x_test_array[k][:6] = x_suggest[-1][6:12]
                    elif m==2: x_test_array[k][:6] = x_suggest[-1][12:18]
                    elif m==3: x_test_array[k][:6] = x_suggest[-1][18:]
                    elif m==4: x_test_array[k][:6] = x_suggest[-2][:6]
                    elif m==5: x_test_array[k][:6] = x_suggest[-2][6:12]
                    elif m==6: x_test_array[k][:6] = x_suggest[-2][12:18]
                    elif m==7: x_test_array[k][:6] = x_suggest[-2][18:]
                elif l==1:
                    if m==0: x_test_array[k][6:12] = x_suggest[-1][:6]
                    elif m==1: x_test_array[k][6:12] = x_suggest[-1][6:12]
                    elif m==2: x_test_array[k][6:12] = x_suggest[-1][12:18]
                    elif m==3: x_test_array[k][6:12] = x_suggest[-1][18:]
                    elif m==4: x_test_array[k][6:12] = x_suggest[-2][:6]
                    elif m==5: x_test_array[k][6:12] = x_suggest[-2][6:12]
                    elif m==6: x_test_array[k][6:12] = x_suggest[-2][12:18]
                    elif m==7: x_test_array[k][6:12] = x_suggest[-2][18:]
                elif l==2:
                    if m==0: x_test_array[k][12:18] = x_suggest[-1][:6]
                    elif m==1: x_test_array[k][12:18] = x_suggest[-1][6:12]
                    elif m==2: x_test_array[k][12:18] = x_suggest[-1][12:18]
                    elif m==3: x_test_array[k][12:18] = x_suggest[-1][18:]
                    elif m==4: x_test_array[k][12:18] = x_suggest[-2][:6]
                    elif m==5: x_test_array[k][12:18] = x_suggest[-2][6:12]
                    elif m==6: x_test_array[k][12:18] = x_suggest[-2][12:18]
                    elif m==7: x_test_array[k][12:18] = x_suggest[-2][18:]
                elif l==3:
                    if m==0: x_test_array[k][18:] = x_suggest[-1][:6]
                    elif m==1: x_test_array[k][18:] = x_suggest[-1][6:12]
                    elif m==2: x_test_array[k][18:] = x_suggest[-1][12:18]
                    elif m==3: x_test_array[k][18:] = x_suggest[-1][18:]
                    elif m==4: x_test_array[k][18:] = x_suggest[-2][:6]
                    elif m==5: x_test_array[k][18:] = x_suggest[-2][6:12]
                    elif m==6: x_test_array[k][18:] = x_suggest[-2][12:18]
                    elif m==7: x_test_array[k][18:] = x_suggest[-2][18:]

        step3_time = time.time() - start_time
        step3_time_list.append(step3_time)


    start_time = time.time()

    for q, combo in enumerate(combinations):
        with open(os.path.join(save_dir, now, "x_test.txt"), "a") as f:
            x_test = x_test_array[q]
            x_test_values = [x.item() if isinstance(x, torch.Tensor) else x for x in x_test]
            f.write(str(i+1)+ "," + ",".join(map(str, combo)) + "," +",".join(map(str, x_test_values)))
            f.write("\n")

    train_x = torch.cat((train_x, torch.tensor(x_test_array)), dim=0) if train_x.size(1) > 0 else torch.tensor(x_test_array)
    print(f"train_x.shape: {train_x.shape}")
    print(f"train_x: {train_x}")

    score_average_array = [
        sum(scores) / len(scores) if len(scores) > 0 else 0
        for scores in score_array]

    for q, combo in enumerate(combinations):
        with open(os.path.join(save_dir, now, "score.txt"), "a") as f:
            score_list = score_array[q]
            score_avarage = score_average_array[q]
            f.write(str(i+1)+ "," + ",".join(map(str, combo)) + "," +",".join(map(str, score_list))+","+str(score_avarage))
            f.write("\n")

    train_y = torch.cat((train_y, torch.tensor(score_average_array).float().view(15, 1)), dim=0) if train_y.size(1) > 0 else torch.tensor(score_average_array).float().view(15, 1)
    print(f"train_y.shape: {train_y.shape}")
    print(f"train_y: {train_y}")


    for q in range(15):
        for p in range(4):
            # 各 15 の要素の「3つの値」の平均を計算
            avg = sum(single_score_array[q][r][p] for r in range(1))
            single_score_average_array[p].append(avg)
    print(f"single_score_average_array: {single_score_average_array}")


    for q, combo in enumerate(combinations):
        with open(os.path.join(save_dir, now, "single_score.txt"), "a") as f:
            flattened_single_score_array = [item for sublist in single_score_array[q] for item in sublist]
            print(f"flattened_single_score_array: {flattened_single_score_array}")
            f.write(str(i+1)+ "," + ",".join(map(str, combo)) + "," +",".join(map(str, flattened_single_score_array)))
            f.write("\n")
        with open(os.path.join(save_dir, now, "single_score_average.txt"), "a") as f:
            f.write(str(i+1)+ "," + ",".join(map(str, combo)) + ",")
            row = [single_score_average_array[0][q],single_score_average_array[1][q],single_score_average_array[2][q],single_score_average_array[3][q]]
            f.write(",".join(map(str, row)))
            f.write("\n")

    train_y_1 = torch.cat((train_y_1, torch.tensor(single_score_average_array[0]).float().view(15, 1)), dim=0) if train_y_1.size(1) > 0 else torch.tensor(single_score_average_array[0]).float().view(15, 1)
    train_y_2 = torch.cat((train_y_2, torch.tensor(single_score_average_array[1]).float().view(15, 1)), dim=0) if train_y_2.size(1) > 0 else torch.tensor(single_score_average_array[1]).float().view(15, 1)
    train_y_3 = torch.cat((train_y_3, torch.tensor(single_score_average_array[2]).float().view(15, 1)), dim=0) if train_y_3.size(1) > 0 else torch.tensor(single_score_average_array[2]).float().view(15, 1)
    train_y_4 = torch.cat((train_y_4, torch.tensor(single_score_average_array[3]).float().view(15, 1)), dim=0) if train_y_4.size(1) > 0 else torch.tensor(single_score_average_array[3]).float().view(15, 1)
    print(f"train_y_1.shape: {train_y_1.shape}")
    print(f"train_y_2.shape: {train_y_2.shape}")
    print(f"train_y_3.shape: {train_y_3.shape}")
    print(f"train_y_4.shape: {train_y_4.shape}")
    print(f"train_y_1: {train_y_1}")

    #shutil.copytree(os.path.join(os.getcwd(), "datasets"), cfg.OUTPUT_DIR, dirs_exist_ok=True)

    step3_time = time.time() - start_time
    step3_time_list.append(step3_time)
    start_time = time.time()

    bounds = torch.tensor([
        [-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0,-60.0, -4.0, -60.0, 0.0, -180.0, 10.0, -60.0, -4.0, -60.0, 0.0, -180.0, 10.0],
        [60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0,60.0, 0.0, 60.0, 30.0, 180.0, 30.0, 60.0, 0.0, 60.0, 30.0, 180.0, 30.0]
        ], dtype=torch.float32)
    # 条件を満たすサンプルを生成
    callable_object = CallableObject()
    x_suggest_list = []
    # ランダムサンプリング
    while len(x_suggest_list) < 2:
        # 各次元ごとにランダムサンプリング
        x_random = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, bounds.shape[1])
        # callable_object の条件を満たすかをチェック
        if callable_object(x_random) >= 0:
            x_suggest_list.append(x_random)
    x_suggest_new = torch.cat(x_suggest_list, dim=0)
    x_suggest = torch.cat((x_suggest, x_suggest_new))

    torch.save(x_suggest, os.path.join(save_dir, now, "epoch"+str(i), f'x_suggest_{i}.pth'))
    torch.save(train_x, os.path.join(save_dir, now, "epoch"+str(i), f'train_x_{i}.pth'))
    torch.save(train_y, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_{i}.pth'))
    torch.save(train_y_1, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_1_{i}.pth'))
    torch.save(train_y_2, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_2_{i}.pth'))
    torch.save(train_y_3, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_3_{i}.pth'))
    torch.save(train_y_4, os.path.join(save_dir, now, "epoch"+str(i), f'train_y_4_{i}.pth'))

    kwargs = {'camera1_pos_x':x_suggest[-1][0],'camera1_pos_y':x_suggest[-1][1],'camera1_pos_z':x_suggest[-1][2],'camera1_angle_x':x_suggest[-1][3],'camera1_angle_y':x_suggest[-1][4],'camera1_focalLength':x_suggest[-1][5],
        'camera2_pos_x':x_suggest[-1][6],'camera2_pos_y':x_suggest[-1][7],'camera2_pos_z':x_suggest[-1][8],'camera2_angle_x':x_suggest[-1][9],'camera2_angle_y':x_suggest[-1][10],'camera2_focalLength':x_suggest[-1][11],
        'camera3_pos_x':x_suggest[-1][12],'camera3_pos_y':x_suggest[-1][13],'camera3_pos_z':x_suggest[-1][14],'camera3_angle_x':x_suggest[-1][15],'camera3_angle_y':x_suggest[-1][16],'camera3_focalLength':x_suggest[-1][17],
        'camera4_pos_x':x_suggest[-1][18],'camera4_pos_y':x_suggest[-1][19],'camera4_pos_z':x_suggest[-1][20],'camera4_angle_x':x_suggest[-1][21],'camera4_angle_y':x_suggest[-1][22],'camera4_focalLength':x_suggest[-1][23],
        'camera5_pos_x':x_suggest[-2][0],'camera5_pos_y':x_suggest[-2][1],'camera5_pos_z':x_suggest[-2][2],'camera5_angle_x':x_suggest[-2][3],'camera5_angle_y':x_suggest[-2][4],'camera5_focalLength':x_suggest[-2][5],
        'camera6_pos_x':x_suggest[-2][6],'camera6_pos_y':x_suggest[-2][7],'camera6_pos_z':x_suggest[-2][8],'camera6_angle_x':x_suggest[-2][9],'camera6_angle_y':x_suggest[-2][10],'camera6_focalLength':x_suggest[-2][11],
        'camera7_pos_x':x_suggest[-2][12],'camera7_pos_y':x_suggest[-2][13],'camera7_pos_z':x_suggest[-2][14],'camera7_angle_x':x_suggest[-2][15],'camera7_angle_y':x_suggest[-2][16],'camera7_focalLength':x_suggest[-2][17],
        'camera8_pos_x':x_suggest[-2][18],'camera8_pos_y':x_suggest[-2][19],'camera8_pos_z':x_suggest[-2][20],'camera8_angle_x':x_suggest[-2][21],'camera8_angle_y':x_suggest[-2][22],'camera8_focalLength':x_suggest[-2][23]
        }

    # Reset script to its original contents
    print("Resetting scripts...")
    bayes_util.ResetScripts(awsimProjectRootPath)
    print("Done.")

    step4_time = time.time() - start_time

    with open(output_file, "a") as f:
        for j in range(1):
            f.write(str(step1_time_list[j])+","+str(step2_time_list[j])+","+str(step3_time_list[j])+",")
        #f.write(str(step3_time_list[3])+",")
        f.write(str(step4_time)+"\n")

