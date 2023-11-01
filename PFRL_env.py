#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:50:50 2023

@author: yirong_cheng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:13:40 2023

@author: yirong_cheng
"""
import gym
import numpy as np
from scipy.stats import unitary_group
from gym import spaces
import h5py

class Scheduler_Env(gym.Env):
    ''' The customized RL environment for a single-cell MIMO scheduler '''
    def __init__(self, N_antennas = 8, N_users = 4):
        super(Scheduler_Env,self).__init__()
        self.N_antennas = N_antennas
        self.N_users = N_users      
        self.group_id = 2   #User location setup (Different environment)
        self.episode = 0    #Time idx (TTI)
        self.beta = 0.5     #Weights of sum_SE and fairness
        self.tau = 20       #Length of pilot signals  
        self.Npower = 0.1   #For normalized symbol power, Npower = 0.2 for bad channel; 0.1 for good/fair channel; 0.05 for bad channel
        
        self.mod_order = 16
        
        #Action space as a binary sequence indicating whether each user is selected
        self.action_space = spaces.MultiBinary(self.N_users)
        
        #Observation space (State space) as CSI matrices + cumulative data rate
        #The CSI matrix is vectorized by users (h_1_real,h_2_real,..., h_1_img,h_2_img,...)
        self.observation_space = spaces.Tuple((spaces.Box(low=-100, high=100, shape=(2*self.N_antennas*self.N_users,)),
                                               spaces.Box(low=0,high=10**3,shape=(self.N_users,))))
        
        # Load the CSI data during initialization:
        if self.group_id == 1:
            # The 2-cluster case
            H_file = h5py.File('./test_CSI_cluster2.hdf5','r')
        else:
            # The random uniform case
            H_file = h5py.File('./test_CSI_uniform.hdf5','r')
        H_r = np.squeeze(np.array(H_file.get('H_r'))[:,100,:,:])
        H_i = np.squeeze(np.array(H_file.get('H_i'))[:,100,:,:])
        self.CSI = np.array(H_r + 1j*H_i)
        idx = np.random.randint(self.CSI.shape[0])
        # initialize the channel information as a random instance stored in the channel data
        self.H = self.CSI[idx,:,:]
    
    def _get_state(self):
        return self._state
    
    def _get_fairness(self,se):
        return sum(se)**2 / (len(se) * np.linalg.norm(se)**2)
        
    def _get_info(self):
        return {"Dummy Info": 0}
                        
    def _get_channel_update(self):
        # Update the channel over time by selecting another random instance stored in the channel data
        idx = np.random.randint(self.CSI.shape[0])
        H_new = self.CSI[idx,:,:]
        return H_new
    
    def _get_channel_est(self):
        # First we assume \tau = 20 symbols as the pilot symbols per TTI
        # Generate the orthonormal pilot matrix P:
        Unitary_mat = unitary_group.rvs(self.tau)
        P = Unitary_mat[:self.N_users,:]
        #Normalize the data signal power for each TTI to 1
        P = np.sqrt(self.tau) * P
        #print("P shape: ", P.shape)
        # Generate noise matrix
        N = np.random.normal(0,self.Npower/2, size = (self.N_antennas, self.tau)) + 1j*np.random.normal(0,self.Npower/2, size = (self.N_antennas, self.tau))
        # Generate the received pilot information
        Y = np.matmul(self.H, P) + N
        # Implement the LS channel estimator
        H_hat = np.matmul(Y, 1/(self.tau) * np.conj(np.transpose(P))) 
        return H_hat
        
    def ZF_combiner(self,selected_user):
        #Having only the estimated channel info
        H_hat = self._get_channel_est()[:,selected_user]
        #Having the ideal channel info
        #H_hat = self.H[:,selected_user]
        inverse = np.matmul(np.conj(np.transpose(H_hat)), H_hat)
        # Return the ZF combiner with dimension: N_user_selected x N_antenna
        if inverse.size == 1:
            # When only a single user selected, the combiner is a vector (cannot use np.matmul)
            return 1/inverse * np.conj(np.transpose(H_hat))
        else:
            return np.matmul(np.linalg.pinv(inverse), np.conj(np.transpose(H_hat)))
    
    def _get_SE(self,idx,selected_user):
        # First we consider the case with perfect beamforming (known CSI at BS):
        combiner = self.ZF_combiner(selected_user)
        
        if selected_user.size == 1:
            desired_P = np.linalg.norm(np.dot(combiner, self.H[:,selected_user]))
            noise_vec = np.random.normal(0,self.Npower/2,self.N_antennas) + 1j*np.random.normal(0,self.Npower/2,self.N_antennas)
            noise_P = np.linalg.norm(np.dot(combiner, noise_vec))
            return np.log2(1 + desired_P/noise_P)
        else:
            desired_P = np.linalg.norm(np.dot(combiner[idx,:], self.H[:,selected_user[idx]]))
            interfered_P = 0
            #print("Desired_P: ")
            #print(desired_P)
            for i in range(0,selected_user.size):
                if i != idx:
                    interfered_P = interfered_P + np.linalg.norm(np.dot(combiner[idx,:], self.H[:,selected_user[i]]))
                    #print("Interfered_P: ")
                    #print(np.linalg.norm(np.dot(combiner[idx,:], self.H[:,selected_user[i]])))
            noise_vec = np.random.normal(0,self.Npower/2,self.N_antennas) + 1j*np.random.normal(0,self.Npower/2,self.N_antennas)
            noise_P = np.linalg.norm(np.dot(combiner[idx,:], noise_vec))
            #print("Noise_P")
            #print(noise_P)
            return np.log2( 1 + desired_P/(interfered_P + noise_P))
    
    #Generate a random symbol (with unit power) and modulate it with complex I/Q signals
    def mod(self):
        modvec_qpsk   =  (1/np.sqrt(2))  * np.array([-1, 1])
        modvec_16qam  =  (1/np.sqrt(10)) * np.array([-3, -1, +3, +1])
        sym_idx = np.random.randint(0,self.mod_order)
        if self.mod_order == 16:
            return complex(modvec_16qam[sym_idx>>2],modvec_16qam[np.mod(sym_idx,4)])
        else: #mod_order = 4
            return complex(modvec_qpsk[sym_idx>>1],modvec_qpsk[np.mod(sym_idx,2)])
        
    def demod(self, sym_received):
        if self.mod_order == 16:
            return float((8*(np.real(sym_received)>0)) + (4*(abs(np.real(sym_received))<0.6325)) + (2*(np.imag(sym_received)>0)) + (1*(abs(np.imag(sym_received))<0.6325)))
        else: #mod_order = 4
            return float(2*(np.real(sym_received)>0) + 1*(np.imag(sym_received)>0))
    
    def action2user(self,action): #Convert the multi-binary vector of action to list of selected user indices
        return np.squeeze(np.nonzero(action))
    
    def reset(self):
        # First define the distribution of CSI (correlated Gaussian)
        self.episode = 0
        self.H = self._get_channel_update()
        H_state = np.hstack((self.H.flatten().real, self.H.flatten().imag))
        self._state = np.hstack((H_state,np.zeros(self.N_users)))
        state = self._get_state()
        info = self._get_info()
        
        return state,info        
                
    
    def step(self, action):
        # To performance the user selection, we need to assume that all users keep sending orthongoal pilots every TTI
        # Hence, the BS has the (estimated) CSI as the input of the policy network
        # Given the action, the BS:
        #   1. Compute the ZF uplink combiner (with perfect CSI now)
        #   2. Compute the SE based on selected user
        #   3. Update the state by updating CSI & updating the cumulative data rate
        
        selected_user = self.action2user(action)
        se = np.zeros(self.N_users)
        if selected_user.size == 1:
            se[selected_user] = self._get_SE(0,selected_user)
        else:
            for i in range(0,selected_user.size):
                se[selected_user[i]] = self._get_SE(i,selected_user)
        avg_rate = self._get_state()[-self.N_users:]
        avg_rate = avg_rate*self.episode/(self.episode+1) + se/(self.episode+1)
        
        reward = self.beta * sum(se) + (1 - self.beta) * self._get_fairness(se)
        #print("Reward: ")
        print("steps: ", self.episode)
        print("se: ", se)
        #print("Fairness: ", (1 - self.beta) * self._get_fairness(se))
        
        self.H = self._get_channel_update()
        H_state = np.hstack((self.H.flatten().real, self.H.flatten().imag))
        self._state = np.hstack((H_state, avg_rate))
        state_new = self._get_state()        
        
        info = self._get_info()
        
        self.episode = self.episode + 1
        return state_new, reward, (self.episode > 100), False, info

if __name__ == "__main__":
    
    
    env = Scheduler_Env()
    state,_ = env.reset()
    
    for i in range(2000):
        # env.action_space.sample() produces randomly sampled action
        #action = env.action_space.sample()
        action = [1,1,0,0]
        if np.count_nonzero(action) != 0:
            observation, reward, terminated, truncated, _ = env.step(action)
            #print("step", i, action, reward)
    
    '''
    H_file = h5py.File('./test_CSI_cluster2_close.hdf5','r')
    a_group_key = list(H_file.keys())
    print(np.array(H_file.get('H_r')).shape)
    H_r = np.squeeze(np.array(H_file.get('H_r'))[:,100,:,:])
    H_i = np.squeeze(np.array(H_file.get('H_i'))[:,100,:,:])
    

    H = np.array(H_r + 1j*H_i)
    # H size: TTI x Antenna x User
    print("H shape is:", H.shape[0])
    Cov_avg = np.zeros((H.shape[2],H.shape[2]))
    for i in range(0,H.shape[0]):
        cov = np.matmul(np.conj(np.transpose(H[i,:,:])), H[i,:,:])
        Cov_avg = Cov_avg + cov
    Cov_avg = Cov_avg / H.shape[0]
    print(Cov_avg)
    '''