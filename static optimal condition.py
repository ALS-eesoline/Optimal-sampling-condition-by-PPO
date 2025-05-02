import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]  
        self.current_episode_reward += reward
        # print(self.locals['dones'][0]==False)
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return True


# 自定义环境
class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()
        self.observation_space = spaces.Box(low=-1000, high=10000, shape=(15,1), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,1), dtype=np.float32)
        # self.dwr_state = np.array([1., 10., 1.5]).reshape(-1,1)  # d,w,r初始状态 d=1m, w=10deg, r=1.5m
        self.target_coor = np.array([3,4,5]).reshape(-1,1)
        self.F_inv = np.eye(3)*0.001
        self.theta = np.ones((3,1))
        self.counter = 0
        self.time = 0
        self.UAV_position = np.random.rand(3,1)
        self.last_samplePoint = np.random.rand(3,1)
        self.done = False
        self.obs_last_position_list = np.random.rand(15,1) #之前的五次采样的bearing信息
        self.thetaPhi = np.zeros((2,1))


    def reset(self, seed=None):
        # self.dwr_state = np.array([1., 10., 1.5]).reshape(-1,1)  # d,w,r初始状态 d=1m, w=10deg, r=1.5m
        self.target_coor = np.array([3,4,5]).reshape(-1,1)
        self.F_inv = np.eye(3) * 0.001
        self.theta = np.ones((3, 1))
        self.counter = 0
        self.time = 0
        self.UAV_position = np.random.rand(3, 1)
        self.last_samplePoint = np.random.rand(3, 1)
        self.done = False
        self.obs_last_position_list = np.random.rand(15, 1)
        self.thetaPhi = np.zeros((2,1))


        obs = self.obs_last_position_list

        return obs, {}

    def step(self, action):

        # 找到采样点
        # self.dwr_state = self.dwr_state + action
        # print(self.dwr_state)

        thetaPhi = np.clip(self.thetaPhi + action, np.array([0, 0]).reshape(-1,1), np.array([np.pi, np.pi]).reshape(-1,1))

        # 因为目标只有静态的，所以theta就是对目标的估计
        self.F_inv, self.theta = estimation_by_bearing(self.target_coor, self.UAV_position, self.F_inv, self.theta)

        #计算下一个采样点的位置       
        next_UAV_position = calculate_UAV_position(self.target_coor, thetaPhi)

        # 计算reward
        action_penalty = np.linalg.norm(action)**2
        esti_error = np.linalg.norm(self.target_coor - self.theta)

        reward = -esti_error * 100 - action_penalty * 20
        # 组装前面的采样变化
        self.obs_last_position_list = update_content(self.obs_last_position_list, action,2)

        observation = self.obs_last_position_list

        self.last_samplePoint = self.UAV_position
        self.UAV_position = next_UAV_position
        self.thetaPhi = thetaPhi

        self.counter += 1
        self.time = self.counter * 0.1

        if self.counter == 12:
            self.done = True

        return observation, reward, self.done, False, {}



def calculate_UAV_position(target,thetaPhi):
    theta = thetaPhi[0]
    phi = thetaPhi[1]

    UAV_position = target + np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]).reshape(-1,1)

    return UAV_position


def update_content(obj,new,num):
    if num == 2:
        obj[2:,:] = obj[0:-2,:]
        obj[0:2,:] = new
    else:
        obj[1:,:] = obj[0:-1,:]
        obj[0,:] = new
    return obj





def estimation_by_bearing(target,samplePoint, F_inv, theta):
    # print('target:',target,'samplePoint:',samplePoint)
    b = target - samplePoint
    # print('b:',b)
    # b = b/np.linalg.norm(b)
    b = b/np.linalg.norm(b) + np.random.normal(0,0.01,(3,1))
    M = np.eye(3) - np.outer(b,b)
    y = M@samplePoint
    # Tmatrix = generate_Tmatrix(t)
    # Tmatrix_next = generate_Tmatrix(t+0.1)
    for i in range(3):
        input = M[:,i].reshape(-1,1)
        output = y[i]
        F_inv = F_inv + input@input.T
        theta = theta + np.linalg.inv(F_inv) @ input * (output - input.T  @ theta)
    return F_inv, theta


def main():
    env = UAVEnv()
    reward_callback = RewardCallback()

    model = PPO("MlpPolicy", env, verbose=1,  device='cpu')

    # 同样可以不断调整参数重复训练多次
    # model.load("ppo_uav_static_7")

    model.learn(total_timesteps=100000, callback = reward_callback)

    model.save("ppo_uav_static_6")

    plt.plot(reward_callback.episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # 测试模型
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(f"Observation: {obs}, Action: {action}, Reward: {reward}")





if __name__ == "__main__":
    main()
