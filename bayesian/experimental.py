import GPy
import GPyOpt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

from scipy.interpolate import Rbf
from scipy import ndimage
#%matplotlib inline

from scipy.interpolate import griddata

import torch

# 要件を満たすランダムな batch_initial_condition を生成
def generate_initial_condition():
    # 初期値のテンソルを作成（requires_grad=True）
    batch_initial_condition = torch.rand((1, 1, 24), dtype=torch.float32, requires_grad=True)

    # 指定された要素の組 (0,2), (6,8), (12,14), (18,20)
    element_pairs = [(0, 2), (6, 8), (12, 14), (18, 20)]

    # detach して一時的に編集可能なテンソルを作成
    detached_condition = batch_initial_condition.detach().clone()

    for first_idx, second_idx in element_pairs:
        # 2つ目の要素をランダムに生成
        second_value = torch.rand(1).item()
        detached_condition[0, 0, second_idx] = second_value

        # 2つ目の要素に基づいて1つ目の要素を生成
        if 0 <= second_value < 0.3333:
            # 1つ目の要素は 0.3333~0.4167 または 0.5833~0.6667
            first_value = torch.rand(1).item()
            if first_value < 0.5:
                detached_condition[0, 0, first_idx] = 0.3333 + (0.4167 - 0.3333) * first_value * 2
            else:
                detached_condition[0, 0, first_idx] = 0.5833 + (0.6667 - 0.5833) * (first_value - 0.5) * 2

        elif 0.3333 <= second_value < 0.4167:
            # 1つ目の要素は 0~0.4167 または 0.5833~1
            first_value = torch.rand(1).item()
            if first_value < 0.5:
                detached_condition[0, 0, first_idx] = 0.4167 * first_value * 2
            else:
                detached_condition[0, 0, first_idx] = 0.5833 + (1.0 - 0.5833) * (first_value - 0.5) * 2

        elif 0.4167 <= second_value < 0.6667:
            # 2つ目の要素を再生成
            second_value = torch.rand(1).item()
            detached_condition[0, 0, second_idx] = second_value

        elif 0.6667 <= second_value < 0.75:
            # 1つ目の要素は 0~0.4167 または 0.5833~1
            first_value = torch.rand(1).item()
            if first_value < 0.5:
                detached_condition[0, 0, first_idx] = 0.4167 * first_value * 2
            else:
                detached_condition[0, 0, first_idx] = 0.5833 + (1.0 - 0.5833) * (first_value - 0.5) * 2

        elif 0.75 <= second_value <= 1:
            # 1つ目の要素は 0.3333~0.4167 または 0.5833~0.6667
            first_value = torch.rand(1).item()
            if first_value < 0.5:
                detached_condition[0, 0, first_idx] = 0.3333 + (0.4167 - 0.3333) * first_value * 2
            else:
                detached_condition[0, 0, first_idx] = 0.5833 + (0.6667 - 0.5833) * (first_value - 0.5) * 2

    # detached_conditionを元のテンソルに戻す
    batch_initial_condition = detached_condition.clone().detach().requires_grad_(True)

    return batch_initial_condition


class CallableObject_standard:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def relu_approximation(x, lower_bound, upper_bound):
            return torch.relu(x - lower_bound) * torch.relu(upper_bound - x)

        def check_conditions(x, indices):
            i, j = indices
            conditions = [
                relu_approximation(x[i], 0, 0.3333) * (relu_approximation(x[j], 0.3333, 0.4167) + relu_approximation(x[j], 0.5833, 0.6667)),
                relu_approximation(x[i], 0.3333, 0.4167) * (relu_approximation(x[j], 0, 0.4167) + relu_approximation(x[j], 0.5833, 1)),
                relu_approximation(x[i], 0.6667, 0.75) * (relu_approximation(x[j], 0, 0.4167) + relu_approximation(x[j], 0.5833, 1)),
                relu_approximation(x[i], 0.75, 1) * (relu_approximation(x[j], 0.3333, 0.4167) + relu_approximation(x[j], 0.5833, 0.6667)),
            ]
            return torch.stack(conditions).max()

        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x[0, 0, :].unsqueeze(0).unsqueeze(0)
        bool_tensors = torch.stack([check_conditions(x.flatten(), indices) for indices in indices_list])
        result = bool_tensors.min() - 0.0000001

        return result


class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def relu_approximation(x, lower_bound, upper_bound):
            return torch.relu(x - lower_bound) * torch.relu(upper_bound - x)

        def check_conditions(x, indices):
            i, j = indices
            conditions = [
                relu_approximation(x[i], -60, -20) * (relu_approximation(x[j], -20, -10) + relu_approximation(x[j], 10, 20)),
                relu_approximation(x[i], -20, -10) * (relu_approximation(x[j], -60, -10) + relu_approximation(x[j], 10, 60)),
                relu_approximation(x[i], 20, 30) * (relu_approximation(x[j], -60, -10) + relu_approximation(x[j], 10, 60)),
                relu_approximation(x[i], 30, 60) * (relu_approximation(x[j], -20, -10) + relu_approximation(x[j], 10, 20)),
            ]
            return torch.stack(conditions).max()

        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x[0, 0, :].unsqueeze(0).unsqueeze(0)
        bool_tensors = torch.stack([check_conditions(x.flatten(), indices) for indices in indices_list])
        result = bool_tensors.min() - 1

        return result









"""
class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def check_conditions(x, indices):
            i, j = indices
            def generate_region(constraints_func, x_range=(-60, 60), y_range=(-60, 60), resolution=121):
                # x, yの範囲と解像度を設定
                x = np.linspace(x_range[0], x_range[1], resolution)
                y = np.linspace(y_range[0], y_range[1], resolution)
                X, Y = np.meshgrid(x, y)
                # 結果の配列を初期化
                result = np.full((resolution, resolution), -1)
                # 各点が制約を満たすかどうかチェック
                for i in range(resolution):
                    for j in range(resolution):
                        if constraints_func(X[i, j], Y[i, j]):
                            result[i, j] = 1
                # 境界を設定
                boundary = np.abs(np.gradient(result.astype(float)))
                result = np.where((boundary[0] + boundary[1]) > 0, 0, result)
                return X, Y, result
            # 制約をif文で定義
            def check_constraints(x, y):
                # 条件式をif文で定義
                if ((x<=-10) | (10<x)) & ((y<=-10) | (20<y)) & (not ((x<=-20) | (20<x)) & ((y<=-20) | (30<y))) :
                    return True
                else:
                    return False
            # 領域を生成
            X, Y, tsdf = generate_region(check_constraints)
            # ゼロ等位面の距離変換を計算
            dist_to_zero = ndimage.distance_transform_edt(tsdf != 0)
            # 元の配列のコピーを作成して、距離で置き換え
            sdf = tsdf.copy()
            sdf[tsdf != 0] = dist_to_zero[tsdf != 0]
            # 負の領域を設定
            sdf[tsdf == 1] *= -1
            # 正負を逆転
            sdf = sdf * -1

            def smooth_function(region, x_range=(-60, 60), y_range=(-60, 60), smooth=5):
                size = region.shape[0]
                x = np.linspace(x_range[0], x_range[1], size)
                y = np.linspace(y_range[0], y_range[1], size)
                xx, yy = np.meshgrid(x, y)
                values = region.flatten()
                xx_flat = xx.flatten()
                yy_flat = yy.flatten()
                print("a")
                rbf = Rbf(xx_flat, yy_flat, values, function='multiquadric', smooth=smooth)
                print("b")
                def smooth_f(x, y):
                    return rbf(x, y)
                print("c")
                return smooth_f

            def smooth_function(region, x_range=(-60, 60), y_range=(-60, 60)):
                size = region.shape[0]
                x = np.linspace(x_range[0], x_range[1], size)
                y = np.linspace(y_range[0], y_range[1], size)
                xx, yy = np.meshgrid(x, y)
                points = np.c_[xx.ravel(), yy.ravel()]
                values = region.ravel()
                def smooth_f(x, y):
                    return griddata(points, values, (x, y), method='cubic')
                return smooth_f
            # スムージングパラメータを設定して連続かつ滑らかな関数の生成
            #smooth_f = smooth_function(sdf, smooth=500)
            smooth_f = smooth_function(sdf)

            x = x.detach().numpy()
            z = smooth_f(x[j], x[i])
            return torch.tensor(z, dtype=torch.float32)



        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x[0, 0, :].unsqueeze(0).unsqueeze(0)
        bool_tensors = torch.stack([check_conditions(x.flatten(), indices) for indices in indices_list])
        result = bool_tensors.min() + 20

        return result
"""

"""
class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def leaky_relu_approximation(x, lower_bound, upper_bound, negative_slope=0.01):
            return torch.nn.functional.leaky_relu(x - lower_bound, negative_slope) * torch.nn.functional.leaky_relu(upper_bound - x, negative_slope)

        def check_conditions(x, indices):
            i, j = indices
            conditions = [
                leaky_relu_approximation(x[i], -60, -20) * (leaky_relu_approximation(x[j], -20, -10) + leaky_relu_approximation(x[j], 10, 20)),
                leaky_relu_approximation(x[i], -20, -10) * (leaky_relu_approximation(x[j], -60, -10) + leaky_relu_approximation(x[j], 10, 60)),
                leaky_relu_approximation(x[i], -10, 20),
                leaky_relu_approximation(x[i], 20, 30) * (leaky_relu_approximation(x[j], -60, -10) + leaky_relu_approximation(x[j], 10, 60)),
                leaky_relu_approximation(x[i], 30, 60) * (leaky_relu_approximation(x[j], -20, -10) + leaky_relu_approximation(x[j], 10, 20)),
            ]
            return torch.stack(conditions).max()

        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]

        # テンソルを統一的に扱うために次元を変形
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # 次元を増やす（[C] -> [1, 1, C]）
        elif x.dim() == 3:
            x = x[0, 0, :].unsqueeze(0).unsqueeze(0)  # 次元を変形（[1, 1, C] -> [1, 1, C]）

        bool_tensors = torch.stack([check_conditions(x.flatten(), indices) for indices in indices_list])

        # 0.5を基準にした条件分岐をLeaky ReLU関数で近似
        leaky_relu_result = torch.nn.functional.leaky_relu(bool_tensors.sum() - 0.5, 0.01)
        result = leaky_relu_result * 2 - 1  # 出力を-1から1の範囲にスケール

        return result
"""
"""
class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def leaky_relu_approximation(x, lower_bound, upper_bound, negative_slope=0.01):
            return torch.nn.functional.leaky_relu(x - lower_bound, negative_slope) * torch.nn.functional.leaky_relu(upper_bound - x, negative_slope)

        def check_conditions(x, indices):
            i, j = indices
            conditions = [
                leaky_relu_approximation(x[i], -60, -20) * (leaky_relu_approximation(x[j], -20, -10) + leaky_relu_approximation(x[j], 10, 20)),
                leaky_relu_approximation(x[i], -20, -10) * (leaky_relu_approximation(x[j], -60, -10) + leaky_relu_approximation(x[j], 10, 60)),
                leaky_relu_approximation(x[i], -10, 20),
                leaky_relu_approximation(x[i], 20, 30) * (leaky_relu_approximation(x[j], -60, -10) + leaky_relu_approximation(x[j], 10, 60)),
                leaky_relu_approximation(x[i], 30, 60) * (leaky_relu_approximation(x[j], -20, -10) + leaky_relu_approximation(x[j], 10, 20)),
            ]
            return torch.stack(conditions).max()

        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]

        if x.dim() == 1:
            bool_tensors = torch.stack([check_conditions(x, indices) for indices in indices_list])
        elif x.dim() == 3:
            bool_tensors = torch.stack([check_conditions(x[0, 0, :], indices) for indices in indices_list])

        # 0.5を基準にした条件分岐をLeaky ReLU関数で近似
        leaky_relu_result = torch.nn.functional.leaky_relu(bool_tensors.sum() - 0.5, 0.01)
        result = leaky_relu_result * 2 - 1  # 出力を-1から1の範囲にスケール

        return result


class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def relu_approximation(x, lower_bound, upper_bound):
            return torch.relu(x - lower_bound) * torch.relu(upper_bound - x)

        def check_conditions(x, indices):
            i, j = indices
            conditions = [
                relu_approximation(x[i], -60, -20) * (relu_approximation(x[j], -20, -10) + relu_approximation(x[j], 10, 20)),
                relu_approximation(x[i], -20, -10) * (relu_approximation(x[j], -60, -10) + relu_approximation(x[j], 10, 60)),
                relu_approximation(x[i], 20, 30) * (relu_approximation(x[j], -60, -10) + relu_approximation(x[j], 10, 60)),
                relu_approximation(x[i], 30, 60) * (relu_approximation(x[j], -20, -10) + relu_approximation(x[j], 10, 20)),
            ]
            return torch.stack(conditions).max()

        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x[0, 0, :].unsqueeze(0).unsqueeze(0)
        bool_tensors = torch.stack([check_conditions(x.flatten(), indices) for indices in indices_list])
        result = bool_tensors.min() - 1

        return result


class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def sigmoid_approximation(x, lower_bound, upper_bound):
            return torch.sigmoid(1e-5 * (x - lower_bound)) * torch.sigmoid(1e-5 * (upper_bound - x))

        def check_conditions(x, indices):
            i, j = indices
            conditions = [
                sigmoid_approximation(x[i], -60, -20) * (sigmoid_approximation(x[j], -20, -10) + sigmoid_approximation(x[j], 10, 20)),
                sigmoid_approximation(x[i], -20, -10) * (sigmoid_approximation(x[j], -60, -10) + sigmoid_approximation(x[j], 10, 60)),
                sigmoid_approximation(x[i], -10, 20),
                sigmoid_approximation(x[i], 20, 30) * (sigmoid_approximation(x[j], -60, -10) + sigmoid_approximation(x[j], 10, 60)),
                sigmoid_approximation(x[i], 30, 60) * (sigmoid_approximation(x[j], -20, -10) + sigmoid_approximation(x[j], 10, 20)),
            ]
            return torch.stack(conditions).max()

        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]

        if x.dim() == 1:
            bool_tensors = torch.stack([check_conditions(x, indices) for indices in indices_list])
        elif x.dim() == 3:
            bool_tensors = torch.stack([check_conditions(x[0, 0, :], indices) for indices in indices_list])

        # 0.5を基準にした条件分岐をシグモイド関数で近似
        sigmoid_result = torch.sigmoid(1e-5 * (bool_tensors.sum() - 0.5))
        result = sigmoid_result * 2 - 1  # 出力を-1から1の範囲にスケール

        return result
"""

"""
class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def sigmoid_approximation(x, lower_bound, upper_bound):
            return torch.sigmoid(10 * (x - lower_bound)) * torch.sigmoid(10 * (upper_bound - x))

        def check_conditions(x, indices):
            i, j = indices
            conditions = [
                sigmoid_approximation(x[i], -60, -20) * (sigmoid_approximation(x[j], -20, -10) + sigmoid_approximation(x[j], 10, 20)),
                sigmoid_approximation(x[i], -20, -10) * (sigmoid_approximation(x[j], -60, -10) + sigmoid_approximation(x[j], 10, 60)),
                sigmoid_approximation(x[i], -10, 20),
                sigmoid_approximation(x[i], 20, 30) * (sigmoid_approximation(x[j], -60, -10) + sigmoid_approximation(x[j], 10, 60)),
                sigmoid_approximation(x[i], 30, 60) * (sigmoid_approximation(x[j], -20, -10) + sigmoid_approximation(x[j], 10, 20)),
            ]
            return torch.stack(conditions).max()

        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]

        if x.dim() == 1:
            bool_tensors = torch.stack([check_conditions(x, indices) for indices in indices_list])
        elif x.dim() == 3:
            bool_tensors = torch.stack([check_conditions(x[0, 0, :], indices) for indices in indices_list])

        result = torch.tensor(1.0, requires_grad=True) if bool_tensors.all() > 0.5 else torch.tensor(-1.0, requires_grad=True)
        return result


class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def check_conditions(x, indices):
            i, j = indices
            conditions = [
                ((-60 <= x[i]) & (x[i] < -20) & ((-20 <= x[j]) & (x[j] < -10) | (10 <= x[j]) & (x[j] < 20))).float(),
                ((-20 <= x[i]) & (x[i] < -10) & ((-60 <= x[j]) & (x[j] < -10) | (10 <= x[j]) & (x[j] < 60))).float(),
                ((-10 <= x[i]) & (x[i] < 20)).float(),
                ((20 <= x[i]) & (x[i] < 30) & ((-60 <= x[j]) & (x[j] < -10) | (10 <= x[j]) & (x[j] < 60))).float(),
                ((30 <= x[i]) & (x[i] < 60) & ((-20 <= x[j]) & (x[j] < -10) | (10 <= x[j]) & (x[j] < 20))).float(),
            ]
            return torch.stack(conditions).max()

        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]

        if x.dim() == 1:
            bool_tensors = torch.stack([check_conditions(x, indices) for indices in indices_list])
        elif x.dim() == 3:
            bool_tensors = torch.stack([check_conditions(x[0, 0, :], indices) for indices in indices_list])

        result = torch.tensor(1.0, requires_grad=True) if bool_tensors.all() else torch.tensor(-1.0, requires_grad=True)
        return result


class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def check_conditions(x, indices):
            i, j = indices
            conditions = [
                (-60 <= x[i]) & (x[i] < -20) & (((-20 <= x[j]) & (x[j] < -10)) | ((10 <= x[j]) & (x[j] < 20))),
                (-20 <= x[i]) & (x[i] < -10) & (((-60 <= x[j]) & (x[j] < -10)) | ((10 <= x[j]) & (x[j] < 60))),
                (-10 <= x[i]) & (x[i] < 20),
                (20 <= x[i]) & (x[i] < 30) & (((-60 <= x[j]) & (x[j] < -10)) | ((10 <= x[j]) & (x[j] < 60))),
                (30 <= x[i]) & (x[i] < 60) & (((-20 <= x[j]) & (x[j] < -10)) | ((10 <= x[j]) & (x[j] < 20))),
            ]
            return torch.any(torch.stack(conditions))

        indices_list = [(2, 0), (8, 6), (14, 12), (20, 18)]

        if x.dim() == 1:
            bool_tensors = torch.stack([check_conditions(x, indices) for indices in indices_list])
        elif x.dim() == 3:
            bool_tensors = torch.stack([check_conditions(x[0, 0, :], indices) for indices in indices_list])

        result = torch.tensor(1.0, requires_grad=True) if bool_tensors.all() else torch.tensor(-1.0, requires_grad=True)
        return result
"""


"""
class CallableObject:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        print(f"x: {x}, len(x): {len(x)}, x.shape: {x.shape}")
        # 入力の形状によって異なるアクセス方法を使う
        if x.dim() == 1:
            print(f"x[0]: {x[0]}, x[1]: {x[1]}, x[2]: {x[2]}")
        elif x.dim() == 3:
            print(f"x[0]: {x[0, 0, 0]}, x[1]: {x[0, 0, 1]}, x[2]: {x[0, 0, 2]}")
        else:
            print("Unexpected tensor dimension")


        boolList = []

        if x.dim() == 1:
            if -60 <= x[2] < -20:
                boolList.append(True) if -20 <= x[0] < -10 or 10 <= x[0] < 20 else boolList.append(False)
            elif -20 <= x[2] < -10:
                boolList.append(True) if -60 <= x[0] < -10 or 10 <= x[0] < 60 else boolList.append(False)
            elif -10 <= x[2] < 20:
                boolList.append(False)
            elif 20 <= x[2] < 30:
                boolList.append(True) if -60 <= x[0] < -10 or 10 <= x[0] < 60 else boolList.append(False)
            elif 30 <= x[2] < 60:
                boolList.append(True) if -20 <= x[0] < -10 or 10 <= x[0] < 20 else boolList.append(False)

            if -60 <= x[8] < -20:
                boolList.append(True) if -20 <= x[6] < -10 or 10 <= x[6] < 20 else boolList.append(False)
            elif -20 <= x[8] < -10:
                boolList.append(True) if -60 <= x[6] < -10 or 10 <= x[6] < 60 else boolList.append(False)
            elif -10 <= x[8] < 20:
                boolList.append(False)
            elif 20 <= x[8] < 30:
                boolList.append(True) if -60 <= x[6] < -10 or 10 <= x[6] < 60 else boolList.append(False)
            elif 30 <= x[8] < 60:
                boolList.append(True) if -20 <= x[6] < -10 or 10 <= x[6] < 20 else boolList.append(False)

            if -60 <= x[14] < -20:
                boolList.append(True) if -20 <= x[12] < -10 or 10 <= x[12] < 20 else boolList.append(False)
            elif -20 <= x[14] < -10:
                boolList.append(True) if -60 <= x[12] < -10 or 10 <= x[12] < 60 else boolList.append(False)
            elif -10 <= x[14] < 20:
                boolList.append(False)
            elif 20 <= x[14] < 30:
                boolList.append(True) if -60 <= x[12] < -10 or 10 <= x[12] < 60 else boolList.append(False)
            elif 30 <= x[14] < 60:
                boolList.append(True) if -20 <= x[12] < -10 or 10 <= x[12] < 20 else boolList.append(False)

            if -60 <= x[20] < -20:
                boolList.append(True) if -20 <= x[18] < -10 or 10 <= x[18] < 20 else boolList.append(False)
            elif -20 <= x[20] < -10:
                boolList.append(True) if -60 <= x[18] < -10 or 10 <= x[18] < 60 else boolList.append(False)
            elif -10 <= x[20] < 20:
                boolList.append(False)
            elif 20 <= x[20] < 30:
                boolList.append(True) if -60 <= x[18] < -10 or 10 <= x[18] < 60 else boolList.append(False)
            elif 30 <= x[20] < 60:
                boolList.append(True) if -20 <= x[18] < -10 or 10 <= x[18] < 20 else boolList.append(False)
            result = torch.tensor(1) if torch.tensor(boolList).all() else torch.tensor(-1)

        if x.dim() == 3:
            if -60 <= x[0, 0, 2] < -20:
                boolList.append(True) if -20 <= x[0, 0, 0] < -10 or 10 <= x[0, 0, 0] < 20 else boolList.append(False)
            elif -20 <= x[0, 0, 2] < -10:
                boolList.append(True) if -60 <= x[0, 0, 0] < -10 or 10 <= x[0, 0, 0] < 60 else boolList.append(False)
            elif -10 <= x[0, 0, 2] < 20:
                boolList.append(False)
            elif 20 <= x[0, 0, 2] < 30:
                boolList.append(True) if -60 <= x[0, 0, 0] < -10 or 10 <= x[0, 0, 0] < 60 else boolList.append(False)
            elif 30 <= x[0, 0, 2] < 60:
                boolList.append(True) if -20 <= x[0, 0, 0] < -10 or 10 <= x[0, 0, 0] < 20 else boolList.append(False)

            if -60 <= x[0, 0, 8] < -20:
                boolList.append(True) if -20 <= x[0, 0, 6] < -10 or 10 <= x[0, 0, 6] < 20 else boolList.append(False)
            elif -20 <= x[0, 0, 8] < -10:
                boolList.append(True) if -60 <= x[0, 0, 6] < -10 or 10 <= x[0, 0, 6] < 60 else boolList.append(False)
            elif -10 <= x[0, 0, 8] < 20:
                boolList.append(False)
            elif 20 <= x[0, 0, 8] < 30:
                boolList.append(True) if -60 <= x[0, 0, 6] < -10 or 10 <= x[0, 0, 6] < 60 else boolList.append(False)
            elif 30 <= x[0, 0, 8] < 60:
                boolList.append(True) if -20 <= x[0, 0, 6] < -10 or 10 <= x[0, 0, 6] < 20 else boolList.append(False)

            if -60 <= x[0, 0, 14] < -20:
                boolList.append(True) if -20 <= x[0, 0, 12] < -10 or 10 <= x[0, 0, 12] < 20 else boolList.append(False)
            elif -20 <= x[0, 0, 14] < -10:
                boolList.append(True) if -60 <= x[0, 0, 12] < -10 or 10 <= x[0, 0, 12] < 60 else boolList.append(False)
            elif -10 <= x[0, 0, 14] < 20:
                boolList.append(False)
            elif 20 <= x[0, 0, 14] < 30:
                boolList.append(True) if -60 <= x[0, 0, 12] < -10 or 10 <= x[0, 0, 12] < 60 else boolList.append(False)
            elif 30 <= x[0, 0, 14] < 60:
                boolList.append(True) if -20 <= x[0, 0, 12] < -10 or 10 <= x[0, 0, 12] < 20 else boolList.append(False)

            if -60 <= x[0, 0, 20] < -20:
                boolList.append(True) if -20 <= x[0, 0, 18] < -10 or 10 <= x[0, 0, 18] < 20 else boolList.append(False)
            elif -20 <= x[0, 0, 20] < -10:
                boolList.append(True) if -60 <= x[0, 0, 18] < -10 or 10 <= x[0, 0, 18] < 60 else boolList.append(False)
            elif -10 <= x[0, 0, 20] < 20:
                boolList.append(False)
            elif 20 <= x[0, 0, 20] < 30:
                boolList.append(True) if -60 <= x[0, 0, 18] < -10 or 10 <= x[0, 0, 18] < 60 else boolList.append(False)
            elif 30 <= x[0, 0, 20] < 60:
                boolList.append(True) if -20 <= x[0, 0, 18] < -10 or 10 <= x[0, 0, 18] < 20 else boolList.append(False)
            result = torch.tensor(1) if torch.tensor(boolList).all() else torch.tensor(-1)

        return result
"""


def bayesian_optimization(x_array, y_array):
  bounds = [{'name': 'camera1_pos_x', 'type': 'continuous', 'domain': (-60, 60)},
            {'name': 'camera1_pos_y', 'type': 'continuous', 'domain': (-4, 0)},
            {'name': 'camera1_pos_z', 'type': 'continuous', 'domain': (-60, 60)},
            {'name': 'camera1_angle_x', 'type': 'continuous', 'domain': (0, 45)},
            {'name': 'camera1_angle_y', 'type': 'continuous', 'domain': (-180, 180)},
            {'name': 'camera1_focalLength', 'type': 'continuous', 'domain': (10, 30)},
            {'name': 'camera2_pos_x', 'type': 'continuous', 'domain': (-60, 60)},
            {'name': 'camera2_pos_y', 'type': 'continuous', 'domain': (-4, 0)},
            {'name': 'camera2_pos_z', 'type': 'continuous', 'domain': (-60, 60)},
            {'name': 'camera2_angle_x', 'type': 'continuous', 'domain': (0, 45)},
            {'name': 'camera2_angle_y', 'type': 'continuous', 'domain': (-180, 180)},
            {'name': 'camera2_focalLength', 'type': 'continuous', 'domain': (10, 30)},
            {'name': 'camera3_pos_x', 'type': 'continuous', 'domain': (-60, 60)},
            {'name': 'camera3_pos_y', 'type': 'continuous', 'domain': (-4, 0)},
            {'name': 'camera3_pos_z', 'type': 'continuous', 'domain': (-60, 60)},
            {'name': 'camera3_angle_x', 'type': 'continuous', 'domain': (0, 45)},
            {'name': 'camera3_angle_y', 'type': 'continuous', 'domain': (-180, 180)},
            {'name': 'camera3_focalLength', 'type': 'continuous', 'domain': (10, 30)},
            {'name': 'camera4_pos_x', 'type': 'continuous', 'domain': (-60, 60)},
            {'name': 'camera4_pos_y', 'type': 'continuous', 'domain': (-4, 0)},
            {'name': 'camera4_pos_z', 'type': 'continuous', 'domain': (-60, 60)},
            {'name': 'camera4_angle_x', 'type': 'continuous', 'domain': (0, 45)},
            {'name': 'camera4_angle_y', 'type': 'continuous', 'domain': (-180, 180)},
            {'name': 'camera4_focalLength', 'type': 'continuous', 'domain': (10, 30)}
            ]

  params = {'acquisition_type':'PI',#獲得関数としてEIを指定    
            'f':None, #最適化する関数の設定（実験結果は分からないので設定しない）．
            'domain':bounds, #パラメータの探索範囲の指定
            'model_type':'GP', #標準的なガウシアンプロセスを指定．
            'X':x_array, #既知データの説明変数（インプットするx）ここが変更
            'Y':y_array, #既知データの目的変数（インプットするy）ここが変更
            'de_duplication':True, #重複したデータをサンプルしないように設定．
            "normalize_Y":True, #defaltはTrue
            "exact_feval":False, #defaltはFalse
            'maximize':True
            } 
  bo = GPyOpt.methods.BayesianOptimization(**params)
  x_suggest = bo.suggest_next_locations(ignored_X=x_array) 
  y_predict = bo.model.model.predict(x_suggest) 
  y_mean =y_predict[0]
  y_variance=y_predict[1]
  x_array = np.vstack((x_array, x_suggest)) # np.stack(())が二重になる点注意。
  #bo.plot_acquisition()
  print(f'つぎは{x_suggest}、yの予測平均は{y_mean}、分散は{y_variance}')

  kwargs = {'camera1_pos_x': x_suggest[-1][0],'camera1_pos_y': x_suggest[-1][1],'camera1_pos_z': x_suggest[-1][2],'camera1_angle_x': x_suggest[-1][3],'camera1_angle_y': x_suggest[-1][4],'camera1_focalLength': x_suggest[-1][5],
            'camera2_pos_x': x_suggest[-1][6],'camera2_pos_y': x_suggest[-1][7],'camera2_pos_z': x_suggest[-1][8],'camera2_angle_x': x_suggest[-1][9],'camera2_angle_y': x_suggest[-1][10],'camera2_focalLength': x_suggest[-1][11],
            'camera3_pos_x': x_suggest[-1][12],'camera3_pos_y': x_suggest[-1][13],'camera3_pos_z': x_suggest[-1][14],'camera3_angle_x': x_suggest[-1][15],'camera3_angle_y': x_suggest[-1][16],'camera3_focalLength': x_suggest[-1][17],
            'camera4_pos_x': x_suggest[-1][18],'camera4_pos_y': x_suggest[-1][19],'camera4_pos_z': x_suggest[-1][20],'camera4_angle_x': x_suggest[-1][21],'camera4_angle_y': x_suggest[-1][22],'camera4_focalLength': x_suggest[-1][23]
            }

  return kwargs, x_array

