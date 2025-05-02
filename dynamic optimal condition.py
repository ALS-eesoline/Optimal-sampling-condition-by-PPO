import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
import random
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

# 从训练的信息中抓出来reward，方便画图
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


# 这个类中实现了采样点的更新，虽然一些额外的代码被写到了类的外面
class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()
        self.observation_space = spaces.Box(low=-1000, high=10000, shape=(25,1), dtype=np.float32)
        self.action_space = spaces.Box(low=-3, high=3, shape=(3,1), dtype=np.float32)
        self.dwr_state = np.array([1., 10., 1.5]).reshape(-1,1)  # d,w,r初始状态 d=1m, w=10deg, r=1.5m
        self.target_coor = target_traj('cont_v')  #训练时使用了固定的轨迹
        self.F_inv = np.eye(3*3)*0.001
        self.theta = np.ones((3*3,1))
        self.counter = 0
        self.time = 0
        self.UAV_position = np.random.rand(3,1)
        self.last_samplePoint = np.random.rand(3,1)
        self.done = False
        self.esti_target_next = np.random.rand(3,1)
        self.esti_target_current = np.random.rand(3,1)
        self.obs_last_error_list = np.random.rand(5,1)  #前五次的预测误差，即真实点与预测点之间的坐标
        self.obs_current_error_list = np.random.rand(5,1)  #这个是算法内部 (output - input.T @ Tmatrix @ theta) 部分的误差累计
        self.obs_last_position_list = np.random.rand(15,1)  #是前五次的 UAV相对于 目标的bearing信息，一次采样生成三个点，五次采样组成15的长度


    def reset(self, seed=None):
        self.dwr_state = np.array([1., 10., 1.5]).reshape(-1,1)  
        self.target_coor = target_traj('cont_v')
        self.F_inv = np.eye(3*3) * 0.001
        self.theta = np.ones((3*3, 1))
        self.counter = 0
        self.time = 0
        self.UAV_position = np.random.rand(3, 1)
        self.last_samplePoint = np.random.rand(3, 1)
        self.done = False
        self.esti_target_next = np.random.rand(3, 1)
        self.esti_target_current = np.random.rand(3, 1)
        self.obs_last_error_list = np.random.rand(5, 1)  
        self.obs_current_error_list = np.random.rand(5, 1) 
        self.obs_last_position_list = np.random.rand(15, 1)
        obs = np.concatenate((self.obs_last_error_list, self.obs_current_error_list, self.obs_last_position_list))

        return obs, {}

    def step(self, action):

        # 找到采样点
        # self.dwr_state = self.dwr_state + action
        # print(self.dwr_state)

        action = np.clip(action, np.array([0, -10, 0]).reshape(-1,1), np.array([1.0, 10, 1.0]).reshape(-1,1)) #人为防止采样点变化过大
        self.dwr_state = np.clip(action+self.dwr_state, np.array([1, -10, 1]).reshape(-1,1), np.array([3, 10, 3]).reshape(-1,1))
        # print(self.dwr_state)
        target_current = self.target_coor[:, self.counter].reshape(-1,1)

        # esti_1为对target_current的估计量, esti_2为对target_next的估计量
        self.F_inv, self.theta, esti_1, esti_2, err  = estimation_by_bearing(target_current, self.UAV_position, self.time, self.F_inv, self.theta)


        # print(f'esti1 {esti_1}, esti2 {esti_2}')esti_target_current
        self.esti_target_current = esti_1
        self.esti_target_next = esti_2
        # 计算下一个采样点位置
        next_UAV_position = calculate_UAV_position(self.dwr_state, self.esti_target_next, self.esti_target_current, last_samplePoint=self.UAV_position, last_last_samplePoint=self.last_samplePoint)

        # 计算reward
        action_penalty = 10 * action[0]**2 + 0.1 * action[1]**2 + 10 * action[2]**2
        esti_error = np.linalg.norm(target_current - esti_1)
        energy_consumed = np.linalg.norm(self.last_samplePoint - self.UAV_position)**2

        distance_uav_target = np.linalg.norm(self.UAV_position - target_current)
        distance_penalty = 0
        #这里的penalty是为了保证无人机在目标一定范围内飞行
        if distance_uav_target > 2 and distance_uav_target < 6 :
            distance_penalty = 0
        else:
            distance_penalty = -300

        reward = -esti_error * 10000 - energy_consumed*10 - action_penalty*100 + distance_penalty
        # print(f'reward: {reward}, esti_error: {esti_error*factor}, energy_consumed: {energy_consumed}')

        # 组装error
        self.obs_last_error_list = update_content(self.obs_last_error_list,esti_error,1)
        self.obs_last_error_list = self.obs_last_error_list.reshape(-1,1)

        self.obs_current_error_list = update_content(self.obs_current_error_list,err,1)
        self.obs_current_error_list = self.obs_current_error_list.reshape(-1,1)

        # 组装前面的采样变化
        self.obs_last_position_list = update_content(self.obs_last_position_list,self.UAV_position - self.last_samplePoint,3)

        observation = np.concatenate((self.obs_last_error_list,self.obs_current_error_list,self.obs_last_position_list))

        self.last_samplePoint = self.UAV_position
        self.UAV_position = next_UAV_position

        self.counter += 1
        self.time = self.counter * 0.1

        if self.counter == 100:
            self.done = True

        return observation, reward, self.done, False, {}


# target_current 预测的下一个点，target_former 预测的当前点,last_samplePoint 当前采样点，last_last_samplePoint 上个采样点
def calculate_UAV_position(dwr,target_current,target_former,last_samplePoint,last_last_samplePoint):
    # 虽然是写的是target_current,former,但是这里使用的都是算法的估计值
    d_direction = - (target_current - target_former)/np.linalg.norm(target_current - target_former)
    d_direction = d_direction.reshape(-1,1)

    u = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)
    u -= np.dot(u.T, d_direction) * d_direction
    u /= np.linalg.norm(u)
    v = np.cross(d_direction.T, u.T).T

    # d_direction,u,v 为正交基，且单位列向量 （如何找到正交基？列出矩阵方程解方程吧！）
    matrix = np.concatenate((d_direction,u,v),axis=1)
    solution = np.linalg.inv(matrix) @ ((last_samplePoint - target_current).reshape(-1,1))
    center = target_current + solution[0] * d_direction

    # 判断往哪个方向转
    AB = last_samplePoint - center
    AC = last_last_samplePoint - center
    DC = AC - np.dot(AC.T,d_direction)*d_direction
    rotate_direction = np.cross(DC.T,AC.T).T
    rotate_direction = np.sign(np.dot(rotate_direction.T,d_direction))

    first_basis = AB / np.linalg.norm(AB)
    second_basis = np.cross(AB.T,-d_direction.T).T
    second_basis = second_basis/np.linalg.norm(second_basis) * rotate_direction

    angle = np.deg2rad(dwr[1])
    UAV_position = target_current + dwr[0] * d_direction + (np.cos(angle) * first_basis + np.sin(angle) * second_basis) * dwr[2]
    return UAV_position


def update_content(obj,new,num):
    if num == 3:
        obj[3:,:] = obj[0:-3,:]
        obj[0:3,:] = new
    else:
        obj[1:,:] = obj[0:-1,:]
        obj[0,:] = new
    return obj


def target_traj(mode):
    if mode == 'cont_v':
        v = np.array([-0.5, 0.3,0.7])
        p0 = np.array([0, 0, 0])
        a = np.array([0.3, 0.4, -0.1])*0.1
        traj = np.zeros((3,100)) 
        for i in range(100):
            t = i * 0.1
            p = p0 + t*v + 0.5*a*t**2
            traj[:,i] = p
    else:
        traj = np.zeros((3,100))
    return traj

def generate_Tmatrix(t):
    expand_number = 3
    T = np.zeros((3,3*expand_number))
    for j in range(1,expand_number+1):
        if j==1:
            T[:,0:3] = np.eye(3)
        else:
            T[:,3*(j-1):3*j] = np.eye(3)*t**(j-1)/math.factorial(j-1)
    return T

# 迭代最小二乘方法
def estimation_by_bearing(target, samplePoint,t, F_inv, theta):
    # print('target:',target,'samplePoint:',samplePoint)
    b = target - samplePoint
    # print('b:',b)
    # b = b/np.linalg.norm(b)
    b = b/np.linalg.norm(b) + np.random.normal(0,0.01,(3,1))
    b = b/np.linalg.norm(b)
    M = np.eye(3) - np.outer(b,b)
    y = M@samplePoint
    Tmatrix = generate_Tmatrix(t)
    Tmatrix_next = generate_Tmatrix(t+0.1)
    err = 0
    for i in range(3):
        input = M[:,i].reshape(-1,1)
        output = y[i]
        F_inv = F_inv + Tmatrix.T @input@input.T@Tmatrix
        theta = theta + np.linalg.inv(F_inv) @ Tmatrix.T @ input * (output - input.T @ Tmatrix @ theta)
        err += (output - input.T @ Tmatrix @ theta)**2
    return F_inv, theta, Tmatrix@theta, Tmatrix_next@theta, err


def main():
    env = UAVEnv()
    reward_callback = RewardCallback()

    # 全连接层，都是用默认
    model = PPO("MlpPolicy", env, verbose=1,  device='cpu')

    # 你可以不断地调整训练参数，即保存某个模型，在load这个模型开始新的训练
    # model.load("ppo_uav_17")

    # 训练模型
    model.learn(total_timesteps=250000, callback = reward_callback)

    # model.save("ppo_uav_19")

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
