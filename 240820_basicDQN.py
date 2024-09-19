# 기본 DQN - noisynet, dueling, per 제거
import numpy as np
import cupy as cp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import datetime
import math

import torch.cuda as cuda
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
import gc

from UAV_env import UAV_env
from Nstep_Buffer import n_step_buffer
from collections import deque

load = False
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./best_model/{date_time}.pkl"
load_path = f"./best_model/obstacle.pkl"

project_exe = "project_240912.exe"

# load_path = f"./best_model/{date_time} + .pkl"
# start_event = Event(enable_timing=True)
# end_event = Event(enable_timing=True)

class Q_network(nn.Module):
    def __init__(self, num_actions):
        super(Q_network, self).__init__()
        self.num_actions = num_actions
        self.image_cnn = nn.Sequential(
            # nn.Conv2d(입력채널 수, 출력 채널수, kernel_size=필터 크기, stride=필터 이동간격)
            # 64x64x4 -> 30x30x32
            nn.Conv2d(4, 32, kernel_size=6, stride=2, groups=1, bias=True),
            nn.GELU(),
            # 30x30x32  -> 13x13x64
            nn.Conv2d(32, 64, kernel_size=6, stride=2, groups=1, bias=True),
            nn.GELU(),
            # 13x13x64 -> 10x10x64
            nn.Conv2d(64, 64, kernel_size=4, stride=1, groups=1, bias=True),
            nn.GELU(),
            # 10x10x64 -> 7x7x64
            nn.Conv2d(64, 64, kernel_size=4, stride=1, groups=1, bias=True),
            nn.GELU(),
            # 7x7x64 -> 5x5x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=1, bias=True)
            # 1600
        )

        self.ray_fc = nn.Sequential(
            nn.Linear(13, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 32)
            # 32
        )

        # 8 ray +

        # 어드벤티지 레이어 => 특정 상태에서 특정 행동을 선택하는게 임의로 뽑는 거 보다 얼마나 좋은지
        self.fc_connected = nn.Sequential(
            nn.Linear(1632, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_actions)
        )

        self.init_weights(self.image_cnn)
        self.init_weights(self.ray_fc)
        self.init_weights(self.fc_connected)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, camera, ray, signal):
        batch = camera.size(0)  # 0의 크기를 반환하기 때문에 batch는 1이 됨
        image_fcinput = self.image_cnn(camera).view(-1, batch)
        combined_input = torch.cat([ray, signal], dim=2)
        ray_fcinput = self.ray_fc(combined_input).view(-1, batch)

        # x에 ray 성분 추가
        x = torch.cat([image_fcinput, ray_fcinput], dim=0)
        x = x.view(batch, -1)

        Q_values = self.fc_connected(x)

        # end_event.record()
        # torch.cuda.synchronize()
        # print("train time: ", start_event.elapsed_time(end_event))
        # print("Q_values:", Q_values)

        # 순수 연산시간 :
        # train 한번 소요시간 :
        return Q_values


class Agent:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.learning_rate = 0.00002  # 0.00002
        self.batch_size = 8
        self.gamma = 0.95  # 0.95
        self.n_step = 1  # 2
        self.num_actions = 9

        self.epsilon = 1
        self.initial_epsilon = 1.0
        self.epsilon_decay_rate = 0.8
        self.final_epsilon = 0.1  # 최종 값
        self.epsilon_decay_period = 200000  # 100000 #(231126)
        self.epsilon_cnt = 0
        self.epsilon_max_cnt = 1

        # self.epsilon_decay = 0.000006 #0.000006
        self.soft_update_rate = 0.005  # 0.01
        self.rate_update_frequency = 150000
        self.max_rate = 0.04

        self.data_buffer = deque(maxlen=15000)
        self.nstep_memory = n_step_buffer(n_step=self.n_step)

        self.policy_net = Q_network(self.num_actions).to(self.device)
        self.Q_target_net = Q_network(self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.Q_target_net.load_state_dict(self.policy_net.state_dict())
        self.Q_target_net.eval()

        # self.data_buffer = deque(maxlen=20000)
        # self.y_max_Q_avg = list()
        # self.y_loss = list()
        self.x_epoch = list()

        if load == True:
            print("Load trained model..")
            self.load_model()

    def update_epsilon(self, current_step):
        if self.epsilon_cnt == self.epsilon_max_cnt:
            self.epsilon = self.final_epsilon
        else:
            if current_step % self.epsilon_decay_period == 0:
                self.epsilon_cnt += 1
                self.initial_epsilon = self.initial_epsilon * self.epsilon_decay_rate
                self.epsilon = max(self.initial_epsilon, self.final_epsilon)
            else:
                cos_decay = 0.5 * (1 + math.cos(
                    math.pi * (current_step % self.epsilon_decay_period) / self.epsilon_decay_period))
                self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * cos_decay

    def epsilon_greedy(self, Q_values):
        # 난수 생성
        if np.random.random() < self.epsilon:
            # action을 random하게 선택
            action = random.randrange(self.num_actions)
            return action
        else:
            # 학습된 Q value 값중 가장 큰 action 선택
            return Q_values.argmax().item()

    # model 저장
    def save_model(self):
        torch.save({
            'state': self.policy_net.state_dict(),
            'optim': self.optimizer.state_dict()},
            save_path)
        return None

    # model 불러오기
    def load_model(self):
        checkpoint = torch.load(load_path)
        self.policy_net.load_state_dict(checkpoint['state'])
        self.Q_target_net.load_state_dict(checkpoint['state'])
        self.optimizer.load_state_dict(checkpoint['optim'])
        return None

    def store_trajectory(self, traj):
        self.data_buffer.append(traj)

    # 1. resizing : 64 * 64, gray scale로
    def re_scale_frame(self, obs):
        obs = cp.array(obs)
        obs = cp.asnumpy(obs)
        obs = np.transpose(obs, (1, 2, 0))
        obs = resize(rgb2gray(obs), (64, 64))
        return obs

    # 2. image 4개씩 쌓기
    def init_image_obs(self, obs):
        obs = self.re_scale_frame(obs)
        frame_obs = [obs for _ in range(4)]
        frame_obs = np.stack(frame_obs, axis=0)
        frame_obs = cp.array(frame_obs)  # cupy 배열로 변환
        return frame_obs

    # 3. 4장 쌓인 Image return
    def init_obs(self, obs):
        return self.init_image_obs(obs)

    def camera_obs(self, obs):
        camera_obs = cp.array(obs)  # cupy 배열로 변환
        # print(obs.shape) # 4 64 64 3
        camera_obs = cp.expand_dims(camera_obs, axis=0)
        camera_obs = torch.from_numpy(cp.asnumpy(camera_obs)).to(self.device)  # GPU로 전송
        return camera_obs

    def ray_obs(self, obs):
        ray_obs = cp.array(obs)  # cupy 배열로 변환
        ray_obs = cp.expand_dims(ray_obs, axis=0)
        ray_obs = torch.from_numpy(cp.asnumpy(ray_obs)).unsqueeze(0).to(self.device)  # GPU로 전송
        return ray_obs

    # numpy 변환은 cpu 연산으로 한 결과에만 적용 가능?

    def ray_obs_cpu(self, obs):
        obs_gpu = cp.asarray(obs)
        obs_gpu = cp.reshape(obs_gpu, (1, -1))
        return cp.asnumpy(obs_gpu)
    # FIFO, 4개씩 쌓기

    def accumulated_image_obs(self, obs, new_frame):
        temp_obs = obs[1:, :, :]  # 4x3x64x64에서 제일 오래된 이미지 제거 => 3x3x64x64
        new_frame = self.re_scale_frame(new_frame)  # 3x64x64
        temp_obs = cp.array(temp_obs)  # cupy 배열로 변환
        new_frame = cp.array(new_frame)  # cupy 배열로 변환
        new_frame = cp.expand_dims(new_frame, axis=0)  # 1x3x64x64
        frame_obs = cp.concatenate((temp_obs, new_frame), axis=0)  # 4x3x64x64
        frame_obs = cp.asnumpy(frame_obs)  # 다시 numpy 배열로 변환
        return frame_obs

    def accumlated_all_obs(self, obs, next_obs):
        return self.accumulated_image_obs(obs, next_obs)

    def convert_action(self, action):
        return action

    # action 선택, discrete action 15개 존재
    # obs shape : torch.Size([1, 4, 64, 64])
    def train_policy(self, obs_camera, obs_ray, signal_data):
        Q_values = self.policy_net(obs_camera, obs_ray, signal_data)
        max_q = Q_values.max()
        action = self.epsilon_greedy(Q_values)
        # print("action:",action)
        # print("action: ", action)
        # print("q_value[0][action] :", Q_values[0][action])
        # print("Q_value[action]",Q_values[0][action])
        return action, Q_values[0][action], max_q
#todo 여기부터

    def batch_torch_obs(self, obs):
        obs = [cp.asarray(ob) for ob in obs]  # obs의 모든 요소를 cupy 배열로 변환
        obs = cp.stack(obs, axis=0)  # obs를 축 0을 기준으로 스택
        obs = cp.squeeze(obs, axis=0) if obs.shape[0] == 1 else obs  # 첫 번째 축 제거
        obs = cp.asnumpy(obs)  # 다시 numpy 배열로 변환
        obs = torch.from_numpy(obs).to(self.device)  # torch tensor로 변환
        return obs

    def batch_ray_obs(self, obs):
        obs = cp.asarray(obs)  # cupy 배열로 변환
        # obs = cp.expand_dims(obs, axis=0)  # 새로운 축 추가
        obs = torch.from_numpy(cp.asnumpy(obs)).to(self.device)  # torch tensor로 변환
        return obs

    def batch_signal_obs(self, obs):
        obs = cp.asarray(obs)  # cupy 배열로 변환
        # obs = cp.expand_dims(obs, axis=0)  # 새로운 축 추가
        obs = torch.from_numpy(cp.asnumpy(obs)).to(self.device)  # torch tensor로 변환
        return obs

    # update target network
    # Q-Network의 파라미터를 target network 복사
    def update_target(self, step):
        if step % self.rate_update_frequency == 0:
            self.soft_update_rate += 0.001

        self.soft_update_rate = min(self.soft_update_rate, self.max_rate)
        # print("soft_rate: ", self.soft_update_rate)

        policy_dict = self.policy_net.state_dict()
        target_dict = self.Q_target_net.state_dict()

        # 소프트 업데이트 수행
        for name in target_dict:
            target_dict[name] = (1.0 - self.soft_update_rate) * target_dict[name] + self.soft_update_rate * policy_dict[
                name]

        # 업데이트된 가중치를 타겟 네트워크에 설정
        self.Q_target_net.load_state_dict(target_dict)
        # self.Q_target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, step, update_target):

        # mini_batch, idxs, IS_weights = self.memory.sample(self.batch_size)
        random_mini_batch = random.sample(self.data_buffer, self.batch_size)
        # #epsilon decaying
        # self.epsilon -= self.epsilon_decay
        # #min of epsilon : 0.05
        # self.epsilon = max(self.epsilon, 0.1) # 약 200000step 이후 최솟값
        # #print("epsilon: ", self.epsilon)

        self.obs_camera_list, self.obs_ray_list, self.signal_list, self.action_list, self.reward_list, self.next_obs_camera_list, self.next_obs_ray_list, self.next_signal_list, self.mask_list = zip(
            *random_mini_batch)

        # tensor
        obses_camera = self.batch_torch_obs(self.obs_camera_list)
        obses_ray = self.batch_ray_obs(self.obs_ray_list)
        # print("camera:",obses_camera.shape)
        # print("ray:",obses_ray.shape)

        actions = torch.LongTensor(self.action_list).unsqueeze(1).to(self.device)

        rewards = torch.Tensor(self.reward_list).to(self.device)
        next_obses_camera = self.batch_torch_obs(self.next_obs_camera_list)
        next_obses_ray = self.batch_ray_obs(self.next_obs_ray_list)

        masks = torch.Tensor(self.mask_list).to(self.device)

        obs_signal = self.batch_signal_obs(self.signal_list)
        next_obs_signal = self.batch_signal_obs(self.next_signal_list)
        # print("signal:",obs_signal.shape)

        # get Q-valueF
        Q_values = self.policy_net(obses_camera, obses_ray, obs_signal)
        # print(Q_values.shape)

        # 추정값
        q_value = Q_values.gather(1, actions).view(-1)
        # print(q_value)

        # get target, y(타겟값) 구하기 위한 다음 state에서의 max Q value
        # target network에서 next state에서의 max Q value -> 상수값
        with torch.no_grad():
            target_q_value = self.Q_target_net(next_obses_camera, next_obses_ray, next_obs_signal).max(1)[0]

        Y = (rewards + masks * (self.gamma ** self.n_step) * target_q_value).clone().detach()

        MSE = nn.MSELoss()
        #           input,  target
        loss = MSE(q_value, Y.detach())
        # errors = F.mse_loss(q_value, Y, reduction='none')

        # 우선순위 업데이트
        # for i in range(self.batch_size):
        #     tree_idx = idxs[i]
        #     self.memory.batch_update(tree_idx, errors[i])

        self.optimizer.zero_grad()

        # loss 정의 (importance sampling)
        # loss =  (torch.FloatTensor(IS_weights).to(self.device) * errors).mean()
        # 10,000번의 episode동안 몇 번의 target network update가 있는지
        # target network update 마다 max Q-value / loss function 분포

        self.x_epoch.append(step)

        # # tensor -> list
        # # max Q-value 분포
        # tensor_to_list_q_value = target_q_value.tolist()
        # # max_Q 값들(batch size : 32개)의 평균 값
        # list_q_value_avg = sum(tensor_to_list_q_value)/len(tensor_to_list_q_value)
        # self.y_max_Q_avg.append(list_q_value_avg)

        # # loss 평균 분포(reduction = mean)
        # loss_in_list = loss.tolist()
        # self.y_loss.append(loss_in_list)

        # backward 시작

        loss.backward()
        self.optimizer.step()

        # --------------------------------------------------------------------


def main():
    # engine_configuration_channel = EngineConfigurationChannel()
    # 파일 위치 지정 중요!
    env = UAV_env(time_scale=1.0, filename=project_exe, port=11300)

    cudnn.enabled = True
    cudnn.benchmark = True

    score = 0
    # episode당 step
    episode_step = 0
    # 전체 누적 step
    step = 0
    update_target = 4000  # 4000 (231126 3000)
    initial_exploration = 10000  # 10000

    # episode당 이동한 거리
    episode_distance = 0
    # total_epi_dis = 0
    # y_epi_dis = list()

    x_epi = list()
    x_episode = list()
    total_step = 0
    total_reward = 0
    y_epi_avg = list()
    y_epsilon = list()
    y_reward = list()
    y_init_qvalue = list()

    # x축 : 학습 과정에서 epoch 수, y축 : maxQ 값의 변화
    # x축 : step 수                y축 : loss 값
    # x축 : episode 수,            y축 : episode당 step수(주행시간)

    agent = Agent()  # 에이전트 인스턴스

    if load:
        agent.load_model()

    for epi in range(5001):
        x_epi.append(epi)
        y_epsilon.append(agent.epsilon)

        obs = env.reset()
        obs_camera = obs[0]

        obs_camera = torch.Tensor(obs_camera)
        # torch.Size([3, 84, 84]

        obs_camera = torch.Tensor(obs_camera).squeeze(dim=0)
        # (84, 84, 3) -> (64, 64, 1) -> 4장씩 쌓아 (64, 64, 4)
        # 같은 Image 4장 쌓기 -> 이후 action에 따라 환경이 바뀌고, 다른 Image data 쌓임
        obs_camera = agent.init_obs(obs_camera)

        obs_signal = obs[1]
        obs_signal = torch.Tensor(obs_signal)

        # ray sensor

        obs_ray = obs[2]
        # c#에서 받아온 obs
        #[0.         1.         1.         0.         1.         1.
        # 0.         1.         1.         0.         1.         1.
        # 1.         0.         0.17635795 0.         1.         1.
        # 0.         1.         1.         0.         1.         1.
        # 0.         1.         1.]

        idx_list = [2,5,8,11,14,17,20,23,26]
        obs_ray_tensor = [obs_ray[i] for i in range(27) if i in idx_list]
        obs_ray_tensor = torch.Tensor(obs_ray_tensor)


        #("obs_signal shape: ", obs_signal.shape)
        #print(obs_signal)

        while True:

            # action 선택
            dis_action, estimate_Q, max_est_Q = agent.train_policy(agent.camera_obs(obs_camera),
                                                                   agent.ray_obs(obs_ray_tensor),
                                                                   agent.ray_obs(obs_signal))

            if episode_step == 0:
                print("Max Q-value: ", max_est_Q.cpu().item())
                print("Epsilon:", agent.epsilon)
                y_init_qvalue.append(max_est_Q.cpu().item())

            # print(obs_signal)
            # action에 따른 step()
            # next step, reward, done 여부
            next_obs, reward, done = env.step(dis_action)
            # print('reward : %f'% reward)

            # action당 이동 거리 측정 -> 한 Episode당 이동한 거리 측정
            # if not done:
            #     if dis_action == 0 or dis_action == 3 or dis_action == 4:
            #         episode_distance += 4.0
            #         speed_count_1 += 1
            #     elif dis_action == 1 or dis_action == 5 or dis_action == 6:
            #         episode_distance += 7.0
            #         speed_count_2 += 1
            #     elif dis_action == 2 or dis_action == 7 or dis_action == 8:
            #         episode_distance += 10.0
            #         speed_count_3 += 1
            #     elif dis_action == 9:
            #         speed_count_4 += 1
            #     elif dis_action == 10:
            #         episode_distance += obs_signal[-1]
            #         speed_count_5 += 1

            # print(episode_distance)
            # state는 camera sensor로 얻은 Image만
            next_obs_camera = next_obs[0]
            next_obs_signal = next_obs[1]
            next_obs_ray = next_obs[2]

            # print(next_obs_ray.shape)
            next_obs_ray_tensor = [next_obs_ray[i] for i in range(27) if i in idx_list]
            next_obs_ray_tensor = torch.Tensor(next_obs_ray_tensor)

            next_obs_camera = torch.Tensor(next_obs_camera).squeeze(dim=0)
            # step이 증가함에 따라 4장 중 1장씩 밀기(FIFO)
            next_obs_camera = agent.accumlated_all_obs(obs_camera, next_obs_camera)
            next_obs_signal = torch.Tensor(next_obs_signal)

            mask = 0 if done else 1
            # print("%d번째 step에서의 reward : %f, action speed : %f"%(step, reward, action_speed))
            score += reward

            agent.store_trajectory(
                [obs_camera, agent.ray_obs_cpu(obs_ray_tensor), agent.ray_obs_cpu(obs_signal), dis_action, reward,
                 next_obs_camera, agent.ray_obs_cpu(next_obs_ray_tensor), agent.ray_obs_cpu(next_obs_signal), mask])
            # return_trajectory = agent.nstep_memory.append(cur_sample) # 멀티 스텝 학습을 위해 리턴값과 다음 상태 값 반환

            # if return_trajectory is not None:
            #     n_step_rewards, next_cam, next_ray, next_sig, last_mask = return_trajectory
            #     # 샘플 수정
            #     return_sample = (obs_camera, agent.ray_obs_cpu(obs_ray_tensor), agent.ray_obs_cpu(obs_signal), dis_action,
            #                      n_step_rewards, next_cam, agent.ray_obs_cpu(next_ray), agent.ray_obs_cpu(next_sig), last_mask)

            #     # TD-error를 위한 max Target Q-value 계산
            #     with torch.no_grad():
            #         target_Q = agent.Q_target_net(agent.camera_obs(next_cam), agent.ray_obs(next_ray), agent.ray_obs(next_sig)).max(1)[0]
            #         target_value = torch.tensor((n_step_rewards + last_mask * (agent.gamma ** agent.n_step)* target_Q).item()).to("cuda:0")

            #     # print("e:",  estimate_Q) # --> 스칼라 텐서
            #     # print("t: ", torch.tensor(target_value)) #--> 스칼라

            #     # 우선 순위 계산을 위한 TD-error 계산
            #     td_error = F.mse_loss(estimate_Q,  target_value.detach())

            #     # 우선 순위 리플레이버퍼에 저장
            #     agent.memory.store(td_error, return_sample)

            obs_camera = next_obs_camera
            obs_ray_tensor = next_obs_ray_tensor
            obs_signal = next_obs_signal

            # SumTree 노드 수가 배치 사이즈 이상 되면 학습
            if step > agent.batch_size:
                # if agent.memory.tree.n_entries > agent.n_step:
                agent.train(step, update_target)

                # 1000 step마다 모델 저장
                if step % 2000 == 0:
                    agent.save_model()

                # 타겟 네트워크 업데이트
                if step % update_target == 0:
                    agent.update_target(step)

            episode_step += 1
            step += 1
            agent.update_epsilon(step)

            if done:
                cuda.empty_cache()
                gc.collect()
                break

        if (epi + 1) % 1 == 0:
            print('%d 번째 episode의 총 step: %d' % (epi + 1, episode_step))
            print('True_score: %f' % score)
            print('Total step: %d\n' % step)

        # 100 episode까지의 step 전체
        total_step = total_step + episode_step
        total_reward = total_reward + score
        # 100 episode 마다
        if epi % 50 == 0:
            x_episode.append(epi)
            # second의 평균
            y_epi_avg.append(total_step // 50)
            y_reward.append(total_reward // 50)
            total_step = 0
            total_reward = 0

        score = 0
        # reward_score = 0
        episode_step = 0
        episode_distance = 0

    results_df = pd.DataFrame({'Episode': x_episode, 'Time': y_epi_avg})
    results_df.to_csv('./results/time_results.csv', index=False)

    results_df = pd.DataFrame({'Episode': x_episode, 'Reward sum': y_reward})
    results_df.to_csv('./results/reward_results.csv', index=False)

    results_df = pd.DataFrame({'Episode': x_epi, 'init Max Q-value': y_init_qvalue})
    results_df.to_csv('./results/initQvalue_results.csv', index=False)

    results_df = pd.DataFrame({'Episode': x_epi, 'Epsilon': y_epsilon})
    results_df.to_csv('./results/Epsilon_results.csv', index=False)
    # results_df = pd.DataFrame({'Episode' : x_episode, 'Move distance' : y_epi_dis})
    # results_df.to_csv('./results/distance_results.csv', index=False)

    # results_df = pd.DataFrame({'Target Update' : agent.x_epoch, 'Max Q value' : agent.y_max_Q_avg})
    # results_df.to_csv('./results/Q_value_results.csv', index=False)

    # results_df = pd.DataFrame({'Target Update' : agent.x_epoch, 'Loss' : agent.y_loss})
    # results_df.to_csv('./results/Loss_results.csv', index=False)

    # 100 episode 마다 episode 종료시까지의 평균 주행시간(second)
    plt.figure(1)
    plt.plot(x_episode, y_epi_avg)
    plt.xlabel('episode')
    plt.ylabel('driving time(sec)')

    plt.figure(2)
    plt.plot(x_episode, y_reward)
    plt.xlabel('episode')
    plt.ylabel('reward sum')

    plt.figure(3)
    plt.plot(x_epi, y_init_qvalue)
    plt.xlabel('episode')
    plt.ylabel('init Max-Qvalue')
    plt.show()

    # # target update(2000 step) 마다 maxQ 값 변화
    # plt.figure(2)
    # plt.plot(agent.x_epoch, agent.y_max_Q_avg)
    # plt.xlabel('target update')
    # plt.ylabel('max Q value')

    # # target update(2000 step) 마다 loss 값 변화
    # plt.figure(3)
    # plt.plot(agent.x_epoch, agent.y_loss)
    # plt.xlabel('target update')
    # plt.ylabel('loss')

    # # 100 episode마다 평균 주행시간
    # plt.figure(4)
    # plt.plot(x_episode, y_epi_dis)
    # plt.xlabel('episode')
    # plt.ylabel('move distance')
    # plt.show()


if __name__ == '__main__':
    main()
