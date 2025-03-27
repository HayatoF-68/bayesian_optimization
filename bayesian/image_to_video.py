import os 
import glob
import cv2 
import shutil

def image_to_video():

    folder_path = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition"

    image_path1= "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/CapturedImages/CapturedImages1"
    image_path2= "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/CapturedImages/CapturedImages2"
    image_path3= "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/CapturedImages/CapturedImages3"
    image_path4= "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/CapturedImages/CapturedImages4"
    image_path5= "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/CapturedImages/CapturedImages5"
    image_path6= "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/CapturedImages/CapturedImages6"
    image_path7= "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/CapturedImages/CapturedImages7"
    image_path8= "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/CapturedImages/CapturedImages8"
    image_paths = [image_path1, image_path2, image_path3, image_path4, image_path5, image_path6, image_path7, image_path8]

    video_path1 = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/camera1"
    video_path2 = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/camera2"
    video_path3 = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/camera3"
    video_path4 = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/camera4"
    video_path5 = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/camera5"
    video_path6 = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/camera6"
    video_path7 = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/camera7"
    video_path8 = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets/camera8"
    video_paths = [video_path1, video_path2, video_path3, video_path4, video_path5, video_path6, video_path7, video_path8]

    collected_data0_file = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/TrafficData0.txt"
    collected_data1_file = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/TrafficData1.txt"
    collected_data2_file = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/TrafficData2.txt"
    collected_data3_file = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/TrafficData3.txt"
    collected_data4_file = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/TrafficData4.txt"
    collected_data5_file = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/TrafficData5.txt"
    collected_data6_file = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/TrafficData6.txt"
    collected_data7_file = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/TrafficData7.txt"
    collected_data8_file = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/TrafficData8.txt"
    collected_data_files = [collected_data1_file, collected_data2_file, collected_data3_file, collected_data4_file, collected_data5_file, collected_data6_file, collected_data7_file, collected_data8_file]

    matrix_file1 = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/SavedMatrix/camera_matrix_inverse_1.txt"
    matrix_file2 = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/SavedMatrix/camera_matrix_inverse_2.txt"
    matrix_file3 = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/SavedMatrix/camera_matrix_inverse_3.txt"
    matrix_file4 = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/SavedMatrix/camera_matrix_inverse_4.txt"
    matrix_file5 = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/SavedMatrix/camera_matrix_inverse_5.txt"
    matrix_file6 = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/SavedMatrix/camera_matrix_inverse_6.txt"
    matrix_file7 = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/SavedMatrix/camera_matrix_inverse_7.txt"
    matrix_file8 = "/home/fujii/Documents/Projects/bayes-arrangement/Assets/Addition/SavedMatrix/camera_matrix_inverse_8.txt"
    matrix_files = [matrix_file1, matrix_file2, matrix_file3, matrix_file4, matrix_file5, matrix_file6, matrix_file7, matrix_file8]

    gt_folder = "/home/fujii/Documents/Projects/bayesian_optimization-master/bayesian_optimization/datasets"

    shutil.copy(collected_data0_file, os.path.join(gt_folder, "gt.txt"))

    for image_path, video_path, collected_data_file, matrix_file in zip(image_paths, video_paths, collected_data_files, matrix_files):
        # *.pngファイルを検索
        a = sorted(glob.glob(os.path.join(image_path, "*.png")))

        # aが空リストの場合のエラーチェックを追加
        if not a:
            print("指定したディレクトリに.pngファイルが見つかりません。")
            exit()  # または適切なエラー処理

        # 最初の画像を読み込んで基本情報を取得
        #img = cv2.imread(a[0])
        #Y, X, channels = img.shape[:3]
        Y, X = 720, 1280
        frame_rate = 10

        # 出力ファイルパスの指定
        file = os.path.join(video_path, 'video.avi')
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D') 
        video = cv2.VideoWriter(file, fourcc, frame_rate, (X, Y))

        # 各画像ファイルに対して処理
        for file in a:
            img = cv2.imread(file)
            if img is None: 
                print("error!")
            else: 
                img = cv2.resize(img, (X, Y))
                video.write(img)

        # ビデオ書き込みを終了
        video.release()

        #collected_data_fileをvideo_pathにコピー"
        shutil.copy(collected_data_file, os.path.join(video_path, "gt.txt"))
        shutil.copy(matrix_file, os.path.join(video_path, "calibration.txt"))