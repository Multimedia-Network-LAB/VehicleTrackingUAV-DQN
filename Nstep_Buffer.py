from collections import deque
import numpy as np

# n-step 까비의 보상을 누적해서 첫 스텝의 보상으로 대체
# n-step의 다음 상태 값을 첫 스텝의 다음 상태 값으로 대체 
# cur_sample = (obs_camera, agent.ray_obs_cpu(obs_ray_tensor), agent.ray_obs_cpu(obs_signal), dis_action, reward, 
# next_obs_camera, agent.ray_obs_cpu(next_obs_ray_tensor), agent.ray_obs_cpu(next_obs_signal), mask)
class n_step_buffer:
    def __init__(self, n_step):
        self.n_step = n_step
        self.buffer = deque()

    def append(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.n_step:
            return_transition = self._get_n_step_return()
            self.buffer.popleft()
            return return_transition
        return None

    def _get_n_step_return(self):
        n_step_rewards = [self.buffer[i][4] for i in range(self.n_step)]
        n_step_return = np.sum([np.power(self.n_step, i) * r for i, r in enumerate(n_step_rewards)])

        # 마지막 스텝의 다음 상태 값을 첫 번째 스텝의 다음 상태 값으로 대체
        last_next_camera = self.buffer[-1][5]
        last_next_ray = self.buffer[-1][6]
        last_next_signal = self.buffer[-1][7]
        last_mask = self.buffer[-1][-1]

        # n-step 리턴과 첫 번째 스텝의 다음 상태 값을 반환
        return (n_step_return, last_next_camera, last_next_ray, last_next_signal, last_mask)