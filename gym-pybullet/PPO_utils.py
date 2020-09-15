import torch
from torch import nn

class PPOActor_pos(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # 状態を受け取り，ガウス分布の平均を出力するネットワークを構築します
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0]),#行動の次元の数だけガウス分布を出力
        )

        # ガウス分布の標準偏差の対数を表す，学習するパラメータを作成します．
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        # [演習] 決定論的な行動を計算し，返します．
        # return ...
        return torch.tanh(self.net(states))#ガウス分布の平均にtanhを適用したのが最適手法
    #def sample(self, states):
        # [演習] ガウス分布の平均と標準偏差から確率論的な行動と確率密度の対数を計算し，返します．
        # (例)
        # actions, log_pis = reparameterize(...)
        # return actions, log_pis
        return reparameterize(self.net(states), self.log_stds)

    #def evaluate_log_pi(self, states, actions):
        # 現在の方策における行動 actions の確率密度の対数を計算し，返します．
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class PPOActor_camera(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # from IPython.core.debugger import Pdb; Pdb().set_trace()                        #ブレークポイント
        self.kernel_size1 = 8
        self.stride1 = 4
        self.kernel_size2 = 4
        self.stride2 = 2
        self.kernel_size3 = 3
        self.stride3 = 1

        self.block1 = nn.Sequential(
            nn.Conv2d(state_shape[0][2], 32, kernel_size=self.kernel_size1, stride=self.stride1),
            # 3x84x84 -> 32x20x20　 #state_shape[0][2]はcamera_shapeの3(RGB)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size2, stride=self.stride2),  # 32x20x20 -> 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=self.kernel_size3, stride=self.stride3),  # 64x9x9 -> 64x7x7
            nn.ReLU()
        )
        """
        output_size = calculate_cnn_size(state_shape[0][2], self.kernel_size1, self.stride1)
        output_size = calculate_cnn_size(output_size, self.kernel_size2, self.stride2)
        output_size = calculate_cnn_size(output_size, self.kernel_size3, self.stride3)
        output_size = int(output_size)
        """

        self.full_connection = nn.Sequential(
            nn.Linear(in_features=3138, out_features=64),  # =64*output_size*output_size+state_shape[1][0]→3138
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0])
        )  # 行動の次元の数だけガウス分布を出力

        # ガウス分布の標準偏差の対数を表す，学習するパラメータを作成します．
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, camera_state, pos_state):
        x = self.block1(camera_state)  # カメラ画像を畳み込み
        x = x.view(x.size(0), -1)  # Flatten. 64x7x7　-> 3136 #(batch_size,3136)

        x = torch.cat([x, pos_state], dim=1)  # cameraの特徴量とゴールまでの相対座標を統合

        # from IPython.core.debugger import Pdb; Pdb().set_trace()#######################
        # x.shape = torch.Size([1, 3138]

        x = self.full_connection(x)  # (batch_num,action_shape)####################################################問題あり

        return torch.tanh(x)  # ガウス分布の平均にtanhを適用したのが最適手法

    def forward2(self, camera_state, pos_state):
        x = self.block1(camera_state)  # カメラ画像を畳み込み
        x = x.view(x.size(0), -1)  # Flatten. 64x7x7　-> 3136 #(batch_size,3136)

        x = torch.cat([x, pos_state], dim=1)  # cameraの特徴量とゴールまでの相対座標を統合

        # from IPython.core.debugger import Pdb; Pdb().set_trace()#######################
        # x.shape = torch.Size([1, 3138]

        x = self.full_connection(x)  # (batch_num,action_shape)####################################################問題あり

        return x

    """
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # 状態を受け取り，ガウス分布の平均を出力するネットワークを構築します
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0]),#行動の次元の数だけガウス分布を出力
        )

        # ガウス分布の標準偏差の対数を表す，学習するパラメータを作成します．
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        # [演習] 決定論的な行動を計算し，返します．
        # return ...
        return torch.tanh(self.net(states))#ガウス分布の平均にtanhを適用したのが最適手法
      """

    def sample(self, states):
        # [演習] ガウス分布の平均と標準偏差から確率論的な行動と確率密度の対数を計算し，返します．
        # (例)
        # actions, log_pis = reparameterize(...)
        # return actions, log_pis
        means = self.forward2(states[0], states[1])
        return reparameterize(means, self.log_stds)

    def evaluate_log_pi(self, cam_states, pos_states, actions):
        # 現在の方策における行動 actions の確率密度の対数を計算し，返します．
        return evaluate_lop_pi(self.forward2(cam_states, pos_states), self.log_stds, actions)


class PPOActor_pos2(nn.Module):

    def __init__(self, state_shape, action_shape):
        super().__init__()

        # 状態を受け取り，ガウス分布の平均を出力するネットワークを構築します
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, action_shape[0]),#行動の次元の数だけガウス分布を出力
        )

        # ガウス分布の標準偏差の対数を表す，学習するパラメータを作成します．
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        # [演習] 決定論的な行動を計算し，返します．
        # return ...
        return torch.tanh(self.net(states))#ガウス分布の平均にtanhを適用したのが最適手法
    def sample(self, states):
        # [演習] ガウス分布の平均と標準偏差から確率論的な行動と確率密度の対数を計算し，返します．
        # (例)
        # actions, log_pis = reparameterize(...)
        # return actions, log_pis
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        # 現在の方策における行動 actions の確率密度の対数を計算し，返します．
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class PPOActor_camera2(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # from IPython.core.debugger import Pdb; Pdb().set_trace()                        #ブレークポイント
        self.kernel_size1 = 8
        self.stride1 = 4
        self.kernel_size2 = 4
        self.stride2 = 2
        self.kernel_size3 = 4
        self.stride3 = 2
        self.kernel_size4 = 3
        self.stride4 = 1

        self.block1 = nn.Sequential(
            nn.Conv2d(state_shape[0][2], 32, kernel_size=self.kernel_size1, stride=self.stride1),
            # 3x84x84 -> 32x20x20　 #state_shape[0][2]はcamera_shapeの3(RGB)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size2, stride=self.stride2),  # 32x20x20 -> 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=self.kernel_size3, stride=self.stride3),  # 64x9x9 -> 64x7x7
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=self.kernel_size4, stride=self.stride4),  # 64x9x9 -> 64x6x6
            nn.ReLU()
        )
        """
        output_size = calculate_cnn_size(state_shape[0][2], self.kernel_size1, self.stride1)
        output_size = calculate_cnn_size(output_size, self.kernel_size2, self.stride2)
        output_size = calculate_cnn_size(output_size, self.kernel_size3, self.stride3)
        output_size = int(output_size)
        """

        self.full_connection = nn.Sequential(
            nn.Linear(in_features=2306, out_features=128),
            # =64*output_size*output_size+state_shape[1][0]→3138
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_shape[0])
        )  # 行動の次元の数だけガウス分布を出力

        # ガウス分布の標準偏差の対数を表す，学習するパラメータを作成します．
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, camera_state, pos_state):
        # from IPython.core.debugger import Pdb; Pdb().set_trace()

        x = self.block1(camera_state)  # カメラ画像を畳み込み
        x = x.view(x.size(0), -1)  # Flatten. 64x7x7　-> 3136 #(batch_size,3136)

        x = torch.cat([x, pos_state], dim=1)  # cameraの特徴量とゴールまでの相対座標を統合

        # from IPython.core.debugger import Pdb; Pdb().set_trace()#######################
        # x.shape = torch.Size([1, 3138]

        x = self.full_connection(x)  # (batch_num,action_shape)####################################################問題あり

        return torch.tanh(x)  # ガウス分布の平均にtanhを適用したのが最適手法

    def forward2(self, camera_state, pos_state):
        x = self.block1(camera_state)  # カメラ画像を畳み込み
        x = x.view(x.size(0), -1)  # Flatten. 64x7x7　-> 3136 #(batch_size,3136)

        x = torch.cat([x, pos_state], dim=1)  # cameraの特徴量とゴールまでの相対座標を統合

        # from IPython.core.debugger import Pdb; Pdb().set_trace()#######################
        # x.shape = torch.Size([1, 3138]

        x = self.full_connection(x)  # (batch_num,action_shape)####################################################問題あり

        return x

    """
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # 状態を受け取り，ガウス分布の平均を出力するネットワークを構築します
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0]),#行動の次元の数だけガウス分布を出力
        )

        # ガウス分布の標準偏差の対数を表す，学習するパラメータを作成します．
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        # [演習] 決定論的な行動を計算し，返します．
        # return ...
        return torch.tanh(self.net(states))#ガウス分布の平均にtanhを適用したのが最適手法
      """

    def sample(self, states):
        # [演習] ガウス分布の平均と標準偏差から確率論的な行動と確率密度の対数を計算し，返します．
        # (例)
        # actions, log_pis = reparameterize(...)
        # return actions, log_pis
        means = self.forward2(states[0], states[1])
        return reparameterize(means, self.log_stds)

    def evaluate_log_pi(self, cam_states, pos_states, actions):
        # 現在の方策における行動 actions の確率密度の対数を計算し，返します．
        return evaluate_lop_pi(self.forward2(cam_states, pos_states), self.log_stds, actions)