# Test for apply beamformer to CSV
import numpy as np
import time
import pyroomacoustics as pra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from function_off import *
# elif (DOAest_block_len > 1) and (t % DOAest_block_len == 0):
# DOAcurrent = []
# for j in range(n_src):
#     DOAdesired = estDOA(Y_pb_block[:, :, j, max(0, DOAest_block_len - t - 1):], DOAest, d, fs=fs, f=f, n_fft=n_fft,
#                         estimatedDOApast=estimatedDOApast, sv_virtual=sv_virtual, a=W_inv[:, :, j], smooth_len=0)
#     DOAcurrent = np.concatenate([DOAcurrent, [DOAdesired]])
# DOAcurrent = np.concatenate([DOApath[:, t], [np.max(DOAcurrent)]])
# estimatedDOApast[:, max(0, t + 1 - DOAest_block_len):t + 1] = np.tile(np.array(DOAcurrent)[:, None],
# Y_pb_block[:, :, j, max(0, DOAest_block_len - t - 1):]这段代码实现了一种窗口效应，其中窗口的大小是 DOAest_block_len，窗口随时间步 t 向前移动。
# (1, min(t + 1, DOAest_block_len)))


from bssSubFunctions import *

eps = 1e-12 # epsilon



def estDOAb(d, fs, n_fft, X, DOA_target):
 d1 = d
 fs1 = fs
 n_fft1 = n_fft
 r1 = np.array(1.5)
 X = X
 num_src = 1
 freq_range = [0.0, 8000.0]
 azimuth1 =  DOA_target* np.pi / 180
 algo_names = sorted(pra.doa.algorithms.keys())

 # 字典用于储存每个算法的结果
 azimuth_dict = {}

 for algo_name in algo_names:
  # Construct the new DOA object
  # the max_four parameter is necessary for FRIDA only
  doa = pra.doa.algorithms[algo_name](L=d1, fs=fs1, nfft=n_fft1,  num_src=1, azimuth=azimuth1)

  # this call here perform localization on the frames in X
  doa.locate_sources(X, freq_bins=freq_range)
  # if algo_name == 'MUSIC':
  #     doa.polar_plt_dirac()
  #     plt.title(algo_name)

  # doa.azimuth_recon contains the reconstructed location of the source
  azimuth = doa.azimuth_recon / np.pi * 180.0
  print(algo_name)
  print("  Recovered azimuth:", azimuth, "degrees")

  # 储存每个算法的结果到字典中
  azimuth_dict[algo_name] = [azimuth, num_src]

 # plt.show()

 return azimuth_dict

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np

def plot_dict_list(dict_list, A, B, C, time1, time2):
    # 初始化一个新的字典来存储每个键对应的所有值
    result_dict = {'MUSIC': [], 'NormMUSIC': [], 'SRP': []}

    # 遍历字典列表，收集每个键对应的所有值
    for d in dict_list:
        for key in result_dict.keys():
            # 将每个值加入列表
            result_dict[key].append(d[key][0][0])
    for key in result_dict.keys():
        result_dict[key] = [(360 - value) if value > 180 else value for value in result_dict[key]]

    # 创建真实值列表
    true_values_A_to_B = np.linspace(A, B, time1)
    true_values_B_to_C = np.linspace(B, C, time2)
    true_values = np.concatenate((true_values_A_to_B, true_values_B_to_C))

    # 创建一个等间隔的时间轴，使得每个数据点在其对应的时间中点出现
    time_stamps = np.linspace(0, 20, len(result_dict['MUSIC']))

    # 使用matplotlib来绘制图形
    fig, ax = plt.subplots(figsize=(8, 6))  # 调整图形尺寸

    # 绘制数据线
    for key, values in result_dict.items():
        ax.plot(time_stamps, values, label=key, linewidth=2)  # 调整线条粗细

    # 绘制真实值
    true_time_stamps = np.linspace(0, 20, len(true_values))
    ax.plot(true_time_stamps, true_values, label='True values', linestyle='--', color='black')

    # 添加标题和标签
    ax.set_title('Comparison of Values over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

    # 添加图例说明
    ax.legend(loc='upper right')

    plt.tight_layout()  # 调整图形布局
    plt.show()
    plt.close()


def beamformtest1(X, DOA1, DOA2, d, DOAb_target, omega, ff):
    # beamformtest2(XBlock[:, :, :, Nb], array_DOAb_interf[Nb],
    #                             array_DOAb_target[Nb], d, array_DOAb_target[Nb],f2omega(f), f,Nb)  # h (L,F,M)
    time_list = []
    # X input shape is (M, F, L)
    # X should be of shape (L, F, M)
    # sv should be of shape (F, M, n_sv)
    # theta = DOA
    # DOAb_target = 130

    c = 340
    delta = 0.02
    tau0 = delta / c
    M = 6
    steerVec = generate_sv(d, DOAb_target, omega)
    # steerVec = sv
    
    # return np.exp(1.j * omega * d * np.cos(ang2rad(theta)) / c)
    # theta_d = DOA* np.pi / 180
    
    # theta_1 = (DOA1 - 40) * np.pi / 180
    # theta_2 = (DOA1 + 40) * np.pi / 180
    # theta_3 = (DOA2 - 20) * np.pi / 180
    # theta_4 = (DOA2 + 20) * np.pi / 180
    theta_1 = (DOA1 - 20) * np.pi / 180
    theta_2 = (DOA1 + 10) * np.pi / 180
    theta_3 = (DOA2 - 10) * np.pi / 180
    theta_4 = (DOA2 + 40) * np.pi / 180
    m_mat, n_mat = np.meshgrid(np.arange(M), np.arange(M))
    diff_mat = m_mat - n_mat
    zero_diff_mask = diff_mat == 0
    
    GammaV1 = np.zeros_like(diff_mat, dtype=np.complex)
    GammaV2 = np.zeros_like(diff_mat, dtype=np.complex)
    Gamma_target = np.zeros((1025, 6, 6), dtype=complex)
    Gamma_interf = np.zeros((1025, 6, 6), dtype=complex)
    h_SC = np.squeeze(steerVec)
    Y = np.zeros_like(X)  # Output signal
    ff[0] = ff[1]
    time_st = time.time()
    for f in range(X.shape[1]):
        f_value = ff[f]  # You need to adjust this line based on your actual frequency range
        # f_value = 7200  # You need to adjust this line based on your actual frequency range
        # steerVec[f,:] = np.exp(1j * ome[f] * tau0 * np.arange(M) * np.cos(DOAb_target)).reshape(-1, 1)
        
        omega = f_value * 2 * np.pi
        
        # only perform the calculation where diff_mat is not zero
        nonzero_indices = np.nonzero(~zero_diff_mask)
        
        GammaV1[nonzero_indices] = (np.cos(theta_1) - np.cos(theta_2)) * (
                np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_1)) -
                np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_2))) / (
                                           1j * omega * tau0 * diff_mat[nonzero_indices])
        GammaV2[nonzero_indices] = (np.cos(theta_3) - np.cos(theta_4)) * (
                np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_3)) -
                np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_4))) / (
                                           1j * omega * tau0 * diff_mat[nonzero_indices])
        Gamma_target[f] = GammaV1
        Gamma_interf[f] = GammaV2
        np.fill_diagonal(GammaV1, np.cos(theta_1) - np.cos(theta_2))
        np.fill_diagonal(GammaV2, np.cos(theta_3) - np.cos(theta_4))
        
        h_SC[f, :] = supercardioid(steerVec[f, :], GammaV1, GammaV2)
        # p = np.squeeze(h_SC[f,:])
        # # Apply the beamformer on the frequency bin
        # Y[:, f, :] = X[:, f, :]*p
        # for m in range(M):
        #     Y[:, f, m] = np.dot(X[:, f, m], h_SC[f, m])
        h_SC_expanded = np.expand_dims(h_SC, axis=0)  # h_SC_expanded.shape -> (1, F, M)
        # x 314 1025 6
        Y = X * h_SC_expanded  # X_weighted.shape -> (L, F, M)
        
        # 对所有麦克风进行求和，得到波束形成的结果
        # Y = np.sum(Y, axis=2)  # beamformed.shape -> (L, F)
    time_ed = time.time() - time_st
    time_list.append(time_ed)
    phi = np.arange(-180, 181) * np.pi / 180
    phi_mat, m2_mat = np.meshgrid(phi, np.arange(M))
    f_index = 1000
    D_phi = np.exp(1j * ff[f_index] * 2 * np.pi * tau0 * m2_mat * np.cos(phi_mat))
    B = np.matmul(D_phi.T, h_SC[f_index])
    Ba = np.abs(B)
    plt.figure(figsize=(7, 7))
    plt.polar(phi, Ba)
    plt.savefig('FBR_Beam_parttern.png')
    return Y, h_SC, time_list, Gamma_target, Gamma_interf

def beamformtest2(X, DOA1, DOA2, d, DOAb_target, omega, ff):
    # beamformtest2(XBlock[:, :, :, Nb], array_DOAb_interf[Nb],
    #                             array_DOAb_target[Nb], d, array_DOAb_target[Nb],f2omega(f), f,Nb)  # h (L,F,M)
    time_list = []
    # X input shape is (M, F, L)
    # X should be of shape (L, F, M)
    # sv should be of shape (F, M, n_sv)
    #theta = DOA
    # DOAb_target = 130
    X = X.transpose(2, 1, 0)
    c = 340
    delta = 0.02
    tau0 = delta / c
    M = 6
    steerVec = generate_sv(d, DOAb_target, omega)
    # steerVec = sv

    #return np.exp(1.j * omega * d * np.cos(ang2rad(theta)) / c)
    # theta_d = DOA* np.pi / 180
    
    # theta_1 = (DOA1 - 40) * np.pi / 180
    # theta_2 = (DOA1 + 40) * np.pi / 180
    # theta_3 = (DOA2 - 20) * np.pi / 180
    # theta_4 = (DOA2 + 20) * np.pi / 180
    theta_1 = (DOA1 - 10) * np.pi / 180
    theta_2 = (DOA1 + 10) * np.pi / 180
    theta_3 = (DOA2 - 30) * np.pi / 180
    theta_4 = (DOA2 + 30 ) * np.pi / 180
    m_mat, n_mat = np.meshgrid(np.arange(M), np.arange(M))
    diff_mat = m_mat - n_mat
    zero_diff_mask = diff_mat == 0

    GammaV1 = np.zeros_like(diff_mat, dtype=np.complex)
    GammaV2 = np.zeros_like(diff_mat, dtype=np.complex)
    Gamma_target = np.zeros((1025, 6, 6), dtype=complex)
    Gamma_interf = np.zeros((1025, 6, 6), dtype=complex)
    h_SC = np.squeeze(steerVec)
    Y = np.zeros_like(X)  # Output signal
    ff[0] = ff[1]
    time_st = time.time()
    for f in range(X.shape[1]):
        f_value = ff[f]  # You need to adjust this line based on your actual frequency range
        # f_value = 7200  # You need to adjust this line based on your actual frequency range
        # steerVec[f,:] = np.exp(1j * ome[f] * tau0 * np.arange(M) * np.cos(DOAb_target)).reshape(-1, 1)

        
        omega = f_value * 2 * np.pi

        # only perform the calculation where diff_mat is not zero
        nonzero_indices = np.nonzero(~zero_diff_mask)

        GammaV1[nonzero_indices] = (np.cos(theta_1) - np.cos(theta_2)) * (
                    np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_1)) -
                    np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_2))) / (
                                               1j * omega * tau0 * diff_mat[nonzero_indices])
        GammaV2[nonzero_indices] = (np.cos(theta_3) - np.cos(theta_4)) * (
                    np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_3)) -
                    np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_4))) / (
                                               1j * omega * tau0 * diff_mat[nonzero_indices])
        Gamma_target[f] = GammaV1
        Gamma_interf[f] = GammaV2
        np.fill_diagonal(GammaV1, np.cos(theta_1) - np.cos(theta_2))
        np.fill_diagonal(GammaV2, np.cos(theta_3) - np.cos(theta_4))
        
        h_SC[f,:] = supercardioid(steerVec[f,:], GammaV1, GammaV2)
        # p = np.squeeze(h_SC[f,:])
        # # Apply the beamformer on the frequency bin
        # Y[:, f, :] = X[:, f, :]*p
        # for m in range(M):
        #     Y[:, f, m] = np.dot(X[:, f, m], h_SC[f, m])
        h_SC_expanded = np.expand_dims(h_SC, axis=0)  # h_SC_expanded.shape -> (1, F, M)
        #x 314 1025 6
        Y = X * h_SC_expanded  # X_weighted.shape -> (L, F, M)
        
        # 对所有麦克风进行求和，得到波束形成的结果
        # Y = np.sum(Y, axis=2)  # beamformed.shape -> (L, F)
    time_ed = time.time() - time_st
    time_list.append(time_ed)
    phi = np.arange(-180, 181) * np.pi / 180
    phi_mat, m2_mat = np.meshgrid(phi, np.arange(M))
    f_index = 1000
    D_phi = np.exp(1j * ff[f_index] * 2 * np.pi * tau0 * m2_mat * np.cos(phi_mat))
    B = np.matmul(D_phi.T, h_SC[f_index])
    Ba = np.abs(B)
    plt.figure(figsize=(7, 7))
    plt.polar(phi, Ba)
    plt.savefig('FBR_Beam_parttern.png')
    return Y,h_SC,time_list,Gamma_target,Gamma_interf

def supercardioid(steerVec, backGamma, foreGamma):
    epsilon = 1e-8
    # backGamma = backGamma + epsilon * np.eye(backGamma.shape[0])
    Mat_D = np.linalg.solve(foreGamma, backGamma)
    eigval, eigvec = np.linalg.eig(Mat_D)
    position = np.argmax(eigval)
    t = eigvec[:, position]
    h_SC = t / np.dot(steerVec.conj().T, t)
    return h_SC
    # return  np.conj(h_SC)

def beamformtest3(X, DOA1, DOA2, d, DOAb_target, ome, ff):
    # X should be of shape (L, F, M)
    # sv should be of shape (F, M, n_sv)
    # h_SC should be of shape (F, M)
    c = 343
    delta = 0.02
    tau0 = delta / c
    M = 6

    steerVec = generate_sv(d, DOAb_target, ome)
    theta_1 = (DOA1 - 40) * np.pi / 180
    theta_2 = (DOA1 + 40) * np.pi / 180
    theta_3 = (DOA2 - 20) * np.pi / 180
    theta_4 = (DOA2 + 20) * np.pi / 180

    m_mat, n_mat = np.meshgrid(np.arange(M), np.arange(M))
    diff_mat = m_mat - n_mat
    zero_diff_mask = diff_mat == 0

    GammaV1 = np.zeros_like(diff_mat, dtype=np.complex)
    GammaV2 = np.zeros_like(diff_mat, dtype=np.complex)
    h_SC = np.squeeze(steerVec)
    Y = np.zeros_like(X)  # Output signal
    ff[0] = ff[1]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    line, = ax.plot([], [])

    def init():
        line.set_data([], [])
        return line,

    def animate(f):
        f_value = ff[f]  # You need to adjust this line based on your actual frequency range
        # f_value = 3000
        omega = f_value * 2 * np.pi

        nonzero_indices = np.nonzero(~zero_diff_mask)

        GammaV1[nonzero_indices] = (np.cos(theta_1) - np.cos(theta_2)) * (
                np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_1)) -
                np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_2))) / (
                                           1j * omega * tau0 * diff_mat[nonzero_indices])
        GammaV2[nonzero_indices] = (np.cos(theta_3) - np.cos(theta_4)) * (
                np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_3)) -
                np.exp(1j * omega * tau0 * diff_mat[nonzero_indices] * np.cos(theta_4))) / (
                                           1j * omega * tau0 * diff_mat[nonzero_indices])

        np.fill_diagonal(GammaV1, np.cos(theta_1) - np.cos(theta_2))
        np.fill_diagonal(GammaV2, np.cos(theta_3) - np.cos(theta_4))
        
        h_SC[f,:] = supercardioid(steerVec[f,:], GammaV1, GammaV2)

        for m in range(M):
            Y[:, f, m] = np.dot(X[:, f, m], h_SC[f, m])

        phi = np.arange(-180, 181) * np.pi / 180
        phi_mat, m2_mat = np.meshgrid(phi, np.arange(M))
        D_phi = np.exp(-1j * omega * tau0 * m2_mat * np.cos(phi_mat))
        B = np.matmul(D_phi.T, h_SC[f])
        Ba = np.abs(B)
        ax.set_ylim(0, np.max(Ba))
        line.set_data(phi, Ba)
        ax.set_title('Frequency: {} Hz'.format(f_value))
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=range(X.shape[1]), init_func=init, blit=True)
    ani.save('FBRBFbeamparttern.mp4')
    print('-------------- finish FBRBFbeamparttern.mp4-------------- ')
    
    
    return Y,h_SC






def estimate_DOA(d, fs, n_fft, XBlock, DOA_range, DOAestalg, plot=False):

    
    M1, F1, BFl1, Nb1 = XBlock.shape
    # Apply BF in each block
    DOAb = []  # Initialize as empty list
    for Nb in range(Nb1):
        # Get DOA estimation for each block in many ways
        azimuth_dict = estDOAb(d=d, fs=fs, n_fft=n_fft, X=XBlock[:, :, :, Nb],DOA_target=DOA_range)
        DOAb.append(azimuth_dict)
    
    if plot:
        if np.all(DOA_range < 90):
            plot_dict_list(DOAb, 0, 80, 0, 100, 100)
        elif np.all(DOA_range >= 90):
            plot_dict_list(DOAb, 180, 100, 180, 100, 100)
    
    # Trans dict to array
    n_sv = len(DOAb)
    methods = DOAb[0].keys()
    
    # Then, we can use list compre5hension and Numpy to convert the list of dictionaries to a 2D array
    array_DOAb = np.array([[p[method][0][0] for p in DOAb] for method in methods])
    
    if DOAestalg == "MUSIC":
        array_DOAb = array_DOAb[0, :]
    elif DOAestalg == "NormMUSIC":
        array_DOAb = array_DOAb[1, :]
    else:
        array_DOAb = array_DOAb[2, :]
    
    return array_DOAb


def estDOAb(d, fs, n_fft, X, DOA_target):
 d1 = d
 fs1 = fs
 n_fft1 = n_fft
 X = X
 num_src = 1
 freq_range = [0.0, 1000.0]
 azimuth1 =  DOA_target* np.pi / 180
 algo_names = sorted(pra.doa.algorithms.keys())

 # 字典用于储存每个算法的结果
 azimuth_dict = {}

 for algo_name in algo_names:
  # Construct the new DOA object
  # the max_four parameter is necessary for FRIDA only
  doa = pra.doa.algorithms[algo_name](L=d1, fs=fs1, nfft=n_fft1,  num_src=1, azimuth=azimuth1)

  # this call here perform localization on the frames in X
  doa.locate_sources(X, freq_bins=freq_range)
  # if algo_name == 'MUSIC':
  #     doa.polar_plt_dirac()
  #     plt.title(algo_name)

  # doa.azimuth_recon contains the reconstructed location of the source
  azimuth = doa.azimuth_recon / np.pi * 180.0
  print(algo_name)
  print("  Recovered azimuth:", azimuth, "degrees")

  # 储存每个算法的结果到字典中
  azimuth_dict[algo_name] = [azimuth, num_src]

 # plt.show()

 return azimuth_dict

def GCCSVAuxIVE(X,sv,lamb_gc_ur=0.2,lamb_gc_null=0,maxIter=50,Lb=100): #
 # (input)
 # X: microphone signal (K(=Frequency) x M(=Microphone) x L(=Time))
 # p: # of DOAs
 # lamb_gc_ur: UR constraint weight
 # lamb_gc_null: null constraint weight
 # maxIter: 50 (scalar)
 # Lb: blockLength (scalar)
 # (output)
 # Z: estimated signal (K x N(=1) x L)
 # u: w itertation
 # (Algorithm)

 time_list = []
 eps = 1e-12
 X = X.transpose(1,0,2)
 K,M,L = X.shape
 # get steering vector and parameters of gc #
 svur = sv
 svnull = None
 c = np.eye(1)
 w = np.tile(np.eye(M,dtype='c16'),(K,1,1,1))[...,:1] # (K x 1 x M x 1)
 u = np.zeros((maxIter,K,M,1),'c16') # (it x K x M x 1)
 XBlock = makeXBlock(X,Lb) # (K x T x M x Lb)
 for it in range(maxIter):
  time_st = time.time()
  ZBlock = Ht(w)@XBlock # (K x T x 1 x Lb)
  C = XBlock@Ht(XBlock)/Lb # (K x T x M x M)
  Cw = C@ w # (K x T x M x 1)
  sgm2 = np.real(Ht(w)@Cw) # (K x T x 1 x 1) (1025 X 3)
  a = Cw/sgm2 # (K x T x M x 1)
  r = np.sqrt(np.sum((np.abs(ZBlock)**2)/sgm2,axis=0))[None] # (1 x T x 1 x Lb)
  V = (XBlock/r@Ht(XBlock)/Lb) # (K x T x M x M)
  urlambddH = np.zeros([K,M,M],'c16') # K x M x M
  lambcd = np.zeros([K,M,1],'c16') # K x M x 1
  for i in range(sv.shape[2]):
    urlambddH += lamb_gc_ur * svur[:,:,i,None] @ Ht(svur[:,:,i,None])# K x M x M
    lambcd += lamb_gc_ur * c * svur[:,:,i,None]  # K x M x 1
 
  tmpP = np.sum(V/sgm2,axis=1,keepdims=True)/(2*L//Lb) + urlambddH[:,None,:] # (K x 1 x M x M)
  tmpQ = np.sum(Ht(w)@V@w*(a/sgm2),axis=1,keepdims=True)/(2*L//Lb) + lambcd[:,None,:] # (K x 1 x M x 1)
  tmpR = np.sum(V,axis=1,keepdims=True) # (K x 1 x M x M))
  P_inv = np.linalg.inv(tmpP) if all(np.linalg.det(tmpP) != 0) else np.linalg.inv(tmpP + eps*np.eye(M))
  w = P_inv@tmpQ # (K x 1 x M x 1)
  w = w/(np.sqrt(Ht(w)@tmpR@w)) # (K x 1 x M x 1)
  u[it] = w[:, 0]
  time_ed = time.time() - time_st
  time_list.append(time_ed)
 Z = projectionBackXY(X, mul(Ht(w[:, 0]), X))
 
 return Z,u,time_list


def FBRGCCSVAuxIVE(X, sv, Gamma_target, Gamma_interfer, lamb_gc_ur=0.2, lamb_gc_null=0, maxIter=50, Lb=5):
    #use gc UPdate rule
    # (input)
    # X: microphone signal (K(=Frequency) x M(=Microphone) x L(=Time))
    # p: # of DOAs
    # lamb_gc_ur: UR constraint weight
    # lamb_gc_null: null constraint weight
    # maxIter: 50 (scalar)
    # Lb: blockLength (scalar)
    # (output)
    # Z: estimated signal (K x N(=1) x L)
    # Gamma_target/interference : K x M x M
    # u: w itertation
    # (Algorithm)
    Lb = 50
    lamb_target = 10
    lamb_interfer = 100
    lamb_s = 1e-6
    time_list = []
    eps = 1e-12
    X = X.transpose(1, 0, 2)
    K, M, L = X.shape
    # get steering vector and parameters of gc #
    svur = sv
    svnull = None
    c = np.eye(1)
    w = np.tile(np.eye(M, dtype='c16'), (K, 1, 1, 1))[..., :1]  # (K x 1 x M x 1)
    u = np.zeros((maxIter, K, M, 1), 'c16')  # (it x K x M x 1)
    XBlock = makeXBlock(X, Lb)  # (K x T x M x Lb)
    for it in range(maxIter):
        time_st = time.time()
        ZBlock = Ht(w) @ XBlock  # (K x T x 1 x Lb)
        C = XBlock @ Ht(XBlock) / Lb  # (K x T x M x M)
        # Ck,t = ˆ E [ xk,t xH k,t ] is the sample-based covariance matrix of xk,t.
        Cw = C @ w  # (K x T x M x 1)
        sgm2 = np.real(Ht(w) @ Cw)  # (K x T x 1 x 1)# σ2k,t denotes the variance of sk,t.
        a = Cw / sgm2  # (K x T x M x 1)
        # The orthogonal constraint(OGC) can be imposed by making ak,t fully dependent on wk t
        r = np.sqrt(np.sum((np.abs(ZBlock) ** 2) / sgm2, axis=0))[None]  # (1 x T x 1 x Lb)
        # rt and V is an auxiliary variabl
        V = (XBlock / r @ Ht(XBlock) / Lb)  # (K x T x M x M)
        urlambddH = np.zeros([K, M, M], 'c16')  # K x M x M
        # lambcd = np.zeros([K, M, 1], 'c16')  # K x M x 1
        # for i in range(sv.shape[2]):
        #     urlambddH += lamb_gc_ur * svur[:, :, i, None] @ Ht(svur[:, :, i, None])  # K x M x M
        #     lambcd += lamb_gc_ur * c * svur[:, :, i, None]  # K x M x 1
        
        tmpP = np.sum(V / sgm2, axis=1, keepdims=True) / (2 * L // Lb)  \
        + lamb_interfer *  Gamma_interfer[:, None, :] - lamb_target * Gamma_target[:, None, :]+ lamb_s * np.eye(M)
               # + urlambddH[:, None, :] # (K x 1 x M x M)
        tmpQ = np.sum(Ht(w) @ V @ w * (a / sgm2), axis=1, keepdims=True) / (2 * L // Lb) \
               # + lambcd[:, None,:]  # (K x 1 x M x 1)
        tmpR = np.sum(V, axis=1, keepdims=True)  # (K x 1 x M x M))
        P_inv = np.linalg.inv(tmpP) if np.all(np.linalg.det(tmpP) != 0) else np.linalg.inv(tmpP + eps * np.eye(M))

        w = P_inv @ tmpQ  # (K x 1 x M x 1)
        w = w / (np.sqrt(Ht(w) @ tmpR @ w))  # (K x 1 x M x 1)
        u[it] = w[:, 0]
        time_ed = time.time() - time_st
        time_list.append(time_ed)
    Z = projectionBackXY(X, mul(Ht(w[:, 0]), X))
    
    return Z, u, time_list


def FBRGCCSVAuxIVE2(X, sv, Gamma_target, Gamma_interfer, lamb_gc_ur=0.2, lamb_gc_null=0, maxIter=50, Lb=5):
    #Add sonstraint in variable vector
    # (input)
    # X: microphone signal (K(=Frequency) x M(=Microphone) x L(=Time))
    # p: # of DOAs
    # lamb_gc_ur: UR constraint weight
    # lamb_gc_null: null constraint weight
    # maxIter: 50 (scalar)
    # Lb: blockLength (scalar)
    # (output)
    # Z: estimated signal (K x N(=1) x L)
    # Gamma_target/interference : K x M x M
    # u: w itertation
    # (Algorithm)
    Lb = 50
    lamb_target = 10
    lamb_interfer = 100
    lamb_s = 1e-6
    time_list = []
    eps = 1e-12
    X = X.transpose(1, 0, 2)
    K, M, L = X.shape
    # get steering vector and parameters of gc #
    svur = sv
    svnull = None
    c = np.eye(1)
    w = np.tile(np.eye(M, dtype='c16'), (K, 1, 1, 1))[..., :1]  # (K x 1 x M x 1)
    u = np.zeros((maxIter, K, M, 1), 'c16')  # (it x K x M x 1)
    XBlock = makeXBlock(X, Lb)  # (K x T x M x Lb)
    for it in range(maxIter):
        time_st = time.time()
        ZBlock = Ht(w) @ XBlock  # (K x T x 1 x Lb)
        C = XBlock @ Ht(XBlock) / Lb  # (K x T x M x M)
        # Ck,t = ˆ E [ xk,t xH k,t ] is the sample-based covariance matrix of xk,t.
        Cw = C @ w  # (K x T x M x 1)
        sgm2 = np.real(Ht(w) @ Cw)  # (K x T x 1 x 1)# σ2k,t denotes the variance of sk,t.
        a = Cw / sgm2  # (K x T x M x 1)
        # The orthogonal constraint(OGC) can be imposed by making ak,t fully dependent on wk t
        r = np.sqrt(np.sum((np.abs(ZBlock) ** 2) / sgm2, axis=0))[None]  # (1 x T x 1 x Lb)
        # rt and V is an auxiliary variabl
        V = (XBlock / r @ Ht(XBlock) / Lb) + + lamb_interfer * Gamma_interfer[:, None, :] - lamb_target * Gamma_target[:, None, :] + lamb_s * np.eye(
            M)  # (K x T x M x M)
        # urlambddH = np.zeros([K, M, M], 'c16')  # K x M x M
        # lambcd = np.zeros([K, M, 1], 'c16')  # K x M x 1
        # for i in range(sv.shape[2]):
        #     urlambddH += lamb_gc_ur * svur[:, :, i, None] @ Ht(svur[:, :, i, None])  # K x M x M
        #     lambcd += lamb_gc_ur * c * svur[:, :, i, None]  # K x M x 1
        
        tmpP = np.sum(V / sgm2, axis=1, keepdims=True) / (2 * L // Lb) \
            #    + lamb_interfer * Gamma_interfer[:, None, :] - lamb_target * Gamma_target[:, None, :] + lamb_s * np.eye(
            # M)
        # + urlambddH[:, None, :] # (K x 1 x M x M)
        tmpQ = np.sum(Ht(w) @ V @ w * (a / sgm2), axis=1, keepdims=True) / (2 * L // Lb) \
            # + lambcd[:, None,:]  # (K x 1 x M x 1)
        tmpR = np.sum(V, axis=1, keepdims=True)  # (K x 1 x M x M))
        P_inv = np.linalg.inv(tmpP) if np.all(np.linalg.det(tmpP) != 0) else np.linalg.inv(tmpP + eps * np.eye(M))
        
        w = P_inv @ tmpQ  # (K x 1 x M x 1)
        w = w / (np.sqrt(Ht(w) @ tmpR @ w))  # (K x 1 x M x 1)
        u[it] = w[:, 0]
        time_ed = time.time() - time_st
        time_list.append(time_ed)
    Z = projectionBackXY(X, mul(Ht(w[:, 0]), X))
    
    return Z, u, time_list


def FBRGCCSVAuxIVE3(X,  filterFBR,  maxIter=50, Lb=5):
    # Sp
    # (input)
    # X: microphone signal (K(=Frequency) x M(=Microphone) x L(=Time))
    # p: # of DOAs
    # lamb_gc_ur: UR constraint weight
    # lamb_gc_null: null constraint weight
    # maxIter: 50 (scalar)
    # Lb: blockLength (scalar)
    # (output)
    # Z: estimated signal (K x N(=1) x L)
    # Gamma_target/interference : K x M x M
    # u: w itertation
    # (Algorithm)
    filterFBR = filterFBR[:, np.newaxis,:, np.newaxis]
    Lb = 50
    lamb_interfer = 100000
    lamb_s = 1e-6
    time_list = []
    eps = 1e-12
    X = X.transpose(1, 0, 2)
    K, M, L = X.shape
    # get steering vector and parameters of gc #
    svnull = None
    c = np.eye(1)
    w = np.tile(np.eye(M, dtype='c16'), (K, 1, 1, 1))[..., :1]  # (K x 1 x M x 1)
    u = np.zeros((maxIter, K, M, 1), 'c16')  # (it x K x M x 1)
    XBlock = makeXBlock(X, Lb)  # (K x T x M x Lb)
    for it in range(maxIter):
        time_st = time.time()
        ZBlock = Ht(w) @ XBlock  # (K x T x 1 x Lb)
        C = XBlock @ Ht(XBlock) / Lb  # (K x T x M x M)
        # Ck,t = ˆ E [ xk,t xH k,t ] is the sample-based covariance matrix of xk,t.
        Cw = C @ w  # (K x T x M x 1)
        sgm2 = np.real(Ht(w) @ Cw)  # (K x T x 1 x 1)# σ2k,t denotes the variance of sk,t.
        a = Cw / sgm2  # (K x T x M x 1)
        # The orthogonal constraint(OGC) can be imposed by making ak,t fully dependent on wk t
        r = np.sqrt(np.sum((np.abs(ZBlock) ** 2) / sgm2, axis=0))[None]  # (1 x T x 1 x Lb)
        # rt and V is an auxiliary variabl
        V = (XBlock / r @ Ht(XBlock) / Lb)  # (K x T x M x M)
        # urlambddH = np.zeros([K, M, M], 'c16')  # K x M x M
        # lambcd = np.zeros([K, M, 1], 'c16')  # K x M x 1
        # for i in range(sv.shape[2]):
        #     urlambddH += lamb_gc_ur * svur[:, :, i, None] @ Ht(svur[:, :, i, None])  # K x M x M
        #     lambcd += lamb_gc_ur * c * svur[:, :, i, None]  # K x M x 1
        
        tmpP = np.sum(V / sgm2, axis=1, keepdims=True) / (2 * L // Lb)
        + 1e-6*np.eye(M) # (K x 1 x M x M)
        tmpQ = np.sum(Ht(w) @ V @ w * (a / sgm2), axis=1, keepdims=True) / (2 * L // Lb) \
            + lamb_interfer*filterFBR  # (K x 1 x M x 1)
        tmpR = np.sum(V, axis=1, keepdims=True)  # (K x 1 x M x M))
        P_inv = np.linalg.inv(tmpP) if np.all(np.linalg.det(tmpP) != 0) else np.linalg.inv(tmpP + eps * np.eye(M))
        
        w = P_inv @ tmpQ  # (K x 1 x M x 1)
        w = w / (np.sqrt(Ht(w) @ tmpR @ w))  # (K x 1 x M x 1)
        u[it] = w[:, 0]
        time_ed = time.time() - time_st
        time_list.append(time_ed)
    Z = projectionBackXY(X, mul(Ht(w[:, 0]), X))
    
    return Z, u, time_list
def FBRGCCSVAuxIVEBL(X, sv, Gamma_target, Gamma_interfer, lamb_gc_ur=0.2, lamb_gc_null=0, maxIter=50, Lb=5):  #
    # (input)
    # X: microphone signal (K(=Frequency) x M(=Microphone) x L(=Time))
    # p: # of DOAs
    # lamb_gc_ur: UR constraint weight
    # lamb_gc_null: null constraint weight
    # maxIter: 50 (scalar)
    # Lb: blockLength (scalar)
    # (output)
    # Z: estimated signal (K x N(=1) x L)
    # Gamma_target/interference : K x M x M
    # u: w itertation
    # (Algorithm)
    lamb_target = 10
    lamb_interfer = 10000
    lamb_s = 1e-6
    time_list = []
    eps = 1e-12
    X = X.transpose(1, 0, 2)
    Gamma_target = Gamma_target.transpose(0, 3,1, 2)
    Gamma_interfer = Gamma_interfer.transpose(0, 3, 1, 2)
    K, M, L = X.shape
    # get steering vector and parameters of gc #
    svur = sv
    svnull = None
    c = np.eye(1)
    w = np.tile(np.eye(M, dtype='c16'), (K, 1, 1, 1))[..., :1]  # (K x 1 x M x 1)
    u = np.zeros((maxIter, K, M, 1), 'c16')  # (it x K x M x 1)
    XBlock = makeXBlock(X, Lb)  # (K x T x M x Lb)
    for it in range(maxIter):
        time_st = time.time()
        ZBlock = Ht(w) @ XBlock  # (K x T x 1 x Lb)
        C = XBlock @ Ht(XBlock) / Lb  # (K x T x M x M)
        # Ck,t = ˆ E [ xk,t xH k,t ] is the sample-based covariance matrix of xk,t.
        Cw = C @ w  # (K x T x M x 1)
        sgm2 = np.real(Ht(w) @ Cw)  # (K x T x 1 x 1)# σ2k,t denotes the variance of sk,t.
        a = Cw / sgm2  # (K x T x M x 1)
        # The orthogonal constraint(OGC) can be imposed by making ak,t fully dependent on wk t
        r = np.sqrt(np.sum((np.abs(ZBlock) ** 2) / sgm2, axis=0))[None]  # (1 x T x 1 x Lb)
        # rt and V is an auxiliary variabl
        V = (XBlock / r @ Ht(XBlock) / Lb)  # (K x T x M x M)
        # urlambddH = np.zeros([K, M, M], 'c16')  # K x M x M
        # lambcd = np.zeros([K, M, 1], 'c16')  # K x M x 1
        # for i in range(sv.shape[2]):
        #     urlambddH += lamb_gc_ur * svur[:, :, i, None] @ Ht(svur[:, :, i, None])  # K x M x M
        #     lambcd += lamb_gc_ur * c * svur[:, :, i, None]  # K x M x 1
        
        tmpP = np.sum(V / sgm2, axis=1, keepdims=True) / (2 * L // Lb) \
               - lamb_interfer * Gamma_interfer + lamb_target * Gamma_target + lamb_s * \
               np.eye(M)
        # + urlambddH[:, None, :] # (K x 1 x M x M)
        tmpQ = np.sum(Ht(w) @ V @ w * (a / sgm2), axis=1, keepdims=True) / (2 * L // Lb) \
            # + lambcd[:, None,:]  # (K x 1 x M x 1)
        tmpR = np.sum(V, axis=1, keepdims=True)  # (K x 1 x M x M))
        P_inv = np.linalg.inv(tmpP) if np.all(np.linalg.det(tmpP) != 0) else np.linalg.inv(tmpP + eps * np.eye(M))
        
        w = P_inv @ tmpQ  # (K x 1 x M x 1)
        w = w / (np.sqrt(Ht(w) @ tmpR @ w))  # (K x 1 x M x 1)
        u[it] = w[:, 0]
        time_ed = time.time() - time_st
        time_list.append(time_ed)
    Z = projectionBackXY(X, mul(Ht(w[:, 0]), X))
    
    return Z, u, time_list

def CSVWPEIVE(X,I=4,D=1,maxIter=100,Lb=10,dpi=5,finewpe=True): # [Ueda+EUSIPCO2023]
 # (input)
 # X: microphone signal (K(=Frequency) x M(=Microphone) x L(=Time))
 # maxIter: (scalar)
 # Lb: blockLength (scalar)
 # L: dereverberation filter length [flames]
 # D: prediction delay [flames]
 # (output)
 # Y: estimated signal (K x N(=1) x L)
 time_list = []
 eps = 1e-12
 X = X.transpose(1,0,2)
 K,M,L = X.shape
 w = np.tile(np.eye(M,dtype='c16'),(K,1,1,1))[...,:1] # (K x 1 x M x 1)
 XBlock = makeXBlock(X,Lb)             # (K x T x M  x Lb)
 G = np.zeros((K,M*I,M),'c16')         # (K x     x ML x M)
 if I>0:
  XBar = makeXBar(X,I,D)               # (K     x MI x L)
  XBarBlock = makeXBlock(XBar,Lb)      # (K x T x MI x Lb)
 for _ in range(maxIter):
  time_st = time.time()
  YBlock = XBlock if I == 0 else XBlock-Ht(G[:,None])@XBarBlock
  ZBlock = Ht(w)@YBlock # (K x T x 1 x Lb)
  C = YBlock@Ht(YBlock)/Lb # (K x T x M x M)
  Cw = C@w # (K x T x M x 1)
  sgm2 = np.real(Ht(w)@Cw) # (K x T x 1 x 1)
  a = Cw/sgm2 # (K x T x M x 1)
  r = np.maximum((np.abs(ZBlock)**2)/sgm2,eps) # (F x T x 1 x Lb)
  rSumInFrequency = np.sum(r,axis=0,keepdims=False) # (1 x T x 1 x Lb)
  rWPE = np.sqrt(rSumInFrequency)
  rIVA = np.sqrt(rSumInFrequency)
  if finewpe:
   rWPE = np.sqrt(r) # (F x T x 1 x Lb)
  if _%dpi == 0 and I>0:
   P = np.mean((XBarBlock/(rWPE*sgm2))@Ht(XBlock)/Lb,axis=1)
   invR = np.linalg.inv(np.mean((XBarBlock/(rWPE*sgm2))@Ht(XBarBlock)/Lb,axis=1))
   G = invR@P # (K x MI x M)
  V = (YBlock/rIVA)@Ht(YBlock)/Lb # (K x T x M x M)
  tmpP = np.sum(V/sgm2,axis=1,keepdims=True) # (K x 1 x M x M)
  tmpQ = np.sum(Ht(w)@V@w*(a/sgm2),axis=1,keepdims=True) # (K x 1 x M x 1)
  tmpR = np.sum(V,axis=1,keepdims=True) # (K x 1 x M x M)
  w = np.linalg.inv(tmpP)@tmpQ
  w = w/np.sqrt(Ht(w)@tmpR@w)
  time_ed = time.time() - time_st
  time_list.append(time_ed)
 Y = X if I == 0 else X-Ht(G)@XBar
 Z = projectionBack(X,Ht(w[:,0])@Y)
 return Z,w,time_list


def CSV_AuxIVE(
        X,  # ndarray (n_frame, n_freq, n_ch); STFT representation of the signal
        dic={},  # dictionary

):  # [Jansky+ 22]
    
    """
    (input)
    X: microphone signal (F(=Frequency) x M(=Microphone) x L(=Time))
    offlineIter: (scalar)
    Nb: frames in each block (scalar)
    (output)
    Y: estimated signal (F x N(=1) x L)
    """
    X = X.transpose(1, 0, 2)
    F, M, L = X.shape
    # L = 314
    Nb = dic.get('Nb', 100)
    offlineIter = dic.get('offlineIter', 100)
    w = np.tile(np.eye(M, 1, dtype='c16'), (F, 1, 1, 1))  # (F x T(=1) x M x N(=1))
    u = np.zeros((offlineIter, F, M, 1), 'c16')  # (offlineIter x F x M x N(=1))
    XBlock = makeXBlock(X, Nb)  # (F x T x M x Nb)
    C = mul(XBlock, Ht(XBlock)) / Nb  # (F x T x M x M)
    
    # geo = sv
    time_list = []
    
    for it in range(offlineIter):
        time_st = time.time()
        Cw = mul(C, w)  # (F x T x M x N(=1))
        sgm2 = np.real(mul(Ht(w), Cw))  # (F x T x 1 x 1)
        a = Cw / sgm2  # (F x T x M x N(=1))
        YBlock = mul(Ht(w), XBlock)  # (F x T x N(=1) x Nb)
        r = norm(YBlock, axis=0)[None]  # (F(=1) x T x 1 x Nb)
        V = mul(XBlock / r, Ht(XBlock)) / Nb  # (F x T x M x M)
        P = np.sum(V / sgm2, axis=1)[:, None]  # (F x 1 x M x M)
        Q = np.sum(mul(Ht(w), V, w) * (a / sgm2), axis=1)[:, None]  # (F x 1 x M x 1)
        R = np.sum(V, axis=1)[:, None]  # (F x 1 x M x M)
        w = mul((np.linalg.pinv(P)), (Q))
        w = w / np.sqrt(mul(Ht(w), R, w))
        u[it] = w[:, 0]  # (offlineIter x F x M x N)
        
        time_ed = time.time() - time_st
        time_list.append(time_ed)
    
    Y = projectionBackXY(X, mul(Ht(w[:, 0]), X))
    return Y, u, time_list


def GCCWPECSVIVE(X, sv, I=4, D=1, maxIter=100, Lb=10, dpi=5, lamb_gc_ur=0.2, lamb_gc_null=0, finewpe=True):
    # (input)
    # X: microphone signal (K(=Frequency) x M(=Microphone) x L(=Time))
    # sv: steering vector
    # maxIter, Lb, lamb_gc_ur, lamb_gc_null, I, D, finewpe: same as in GCCSVAuxIVE and CSVWPEIVE
    # (output)
    # Y: estimated signal (K x N(=1) x L)
    
    time_list = []
    eps = 1e-12
    X = X.transpose(1, 0, 2)
    K, M, L = X.shape
    
    # WPE part
    G = np.zeros((K, M * I, M), 'c16')
    if I > 0:
        XBar = makeXBar(X, I, D)
        XBarBlock = makeXBlock(XBar, Lb)
    
    # GC part
    svur = sv
    svnull = None
    c = np.eye(1)
    w = np.tile(np.eye(M, dtype='c16'), (K, 1, 1, 1))[..., :1]
    
    XBlock = makeXBlock(X, Lb)
    
    for _ in range(maxIter):
        time_st = time.time()
        
        # WPE part
        YBlock = XBlock if I == 0 else XBlock - Ht(G[:, None]) @ XBarBlock
        
        # GC part
        ZBlock = Ht(w) @ YBlock
        C = YBlock @ Ht(YBlock) / Lb
        Cw = C @ w
        sgm2 = np.real(Ht(w) @ Cw)
        a = Cw / sgm2
        r = np.maximum((np.abs(ZBlock) ** 2) / sgm2, eps)
        rSumInFrequency = np.sum(r, axis=0, keepdims=False)
        rWPE = np.sqrt(rSumInFrequency)
        rIVA = np.sqrt(rSumInFrequency)
        if finewpe:
            rWPE = np.sqrt(r)
        
        # WPE part
        if _ % dpi == 0 and I > 0:
            P = np.mean((XBarBlock / (rWPE * sgm2)) @ Ht(XBlock) / Lb, axis=1)
            invR = np.linalg.inv(np.mean((XBarBlock / (rWPE * sgm2)) @ Ht(XBarBlock) / Lb, axis=1))
            G = invR @ P
        
        # GC part
        V = (YBlock / rIVA) @ Ht(YBlock) / Lb
        urlambddH = np.zeros([K, M, M], 'c16')
        lambcd = np.zeros([K, M, 1], 'c16')
        for i in range(sv.shape[2]):
            urlambddH += lamb_gc_ur * svur[:, :, i, None] @ Ht(svur[:, :, i, None])
            lambcd += lamb_gc_ur * c * svur[:, :, i, None]
        tmpP = np.sum(V / sgm2, axis=1, keepdims=True) / (2 * L // Lb) + urlambddH[:, None, :]
        tmpQ = np.sum(Ht(w) @ V @ w * (a / sgm2), axis=1, keepdims=True) / (2 * L // Lb) + lambcd[:, None, :]
        tmpR = np.sum(V, axis=1, keepdims=True)
        P_inv = np.linalg.inv(tmpP) if all(np.linalg.det(tmpP) != 0) else np.linalg.inv(tmpP + eps * np.eye(M))
        w = P_inv @ tmpQ
        w = w / (np.sqrt(Ht(w) @ tmpR @ w))
        
        time_ed = time.time() - time_st
        time_list.append(time_ed)
    
    Y = X if I == 0 else X - Ht(G) @ XBar
    Z = projectionBack(X, Ht(w[:, 0]) @ Y)
    return Z, w, time_list


def MVDR(X, sv, dic={}):
    '''
    (input)
    X: microphone signal (F(=Frequency) x M(=Microphone) x T(=Time))

    sv: steering vector (F x M x nSv)
    calcSv:
        - DOA: calculate sv using DOA.
        - data: calculate sv using eigen vector.

    Cov: covariance matrix (F x M x M), (F x nSv x M x M)
    MVDRType:
     - DS: Assume the noise covariance as whitened.
     - MPDR: Calculate noise covariance using X.
     - MVDR: Calculate noise covariance using training data.

    (output)
    Y: separated signal (F x nSv x T)
    W: separation filter (F x M x nSv)
    '''
    refMic = 0
    X = X.transpose(1, 2, 0)
    F, M, T = X.shape
    Im = np.eye(M, dtype='c16')[None]  # (1 x M x M)
    
    # sv (F x M x nSv)
    
    ## (prepare Cov)
    Cov = dic.get('Cov', None)
    if Cov is None:
        MVDRType = dic.get('MVDRType', 'MPDR')
        if MVDRType == 'MVDR':
            # Cov = getCovUsingSrcIR(dic)  # (F x nSv x M x M)
            Cov = None
        elif MVDRType == 'MPDR':
            Cov = mul(X, Ht(X))  # (F x M x M)
        else:  # MVDRType == 'DS'
            Cov = Im  # (F x M x M)
    
    ## (calc noisy Cov for eacn n)
    _, _, nSv = sv.shape
    if Cov.ndim == 3:  # (F x M x M)
        Cov = Cov[:, None, :, :]  # (F x nSv(=1) x M x M)
    else:
        CovTmp = np.zeros_like(Cov, 'c16')
        for n in range(nSv):
            nTmp = list(range(n)) + list(range(n + 1, nSv))
            CovTmp[:, n] = np.sum(Cov[:, nTmp], axis=1)
        Cov = CovTmp  # (F x nSv x M x M)
    
    ## (MVDR calculation)
    Covinv = inv(Cov)  # (F x nSv x M x M)
    sv = tr(sv)[..., None]  # (F x nSv x M x 1)
    denom = mul(Ht(sv), Covinv, sv)  # (F x nSv x 1 x 1)
    W = sv[:, :, refMic, None, :] \
        * mul(Covinv, sv) / denom  # (F x nSv x M x 1)
    W = tr(W[:, :, :, 0])  # (F x M x nSv)
    
    ## (process)
    Y = mul(Ht(W), X)  # (F x nSv x T)
    return Y, W[None]

# def Ht(X):
#     return np.conj(X).swapaxes(-1, -2)  # Ht: Hermitian transpose

def OCSVAuxIVE(X, Lb = 100, alpha = 0.96, maxiter=10):
    time_list = []
    # (input)
    '''
    X: microphone signal (F(=frequency) * M(=Microphone) * L(Time))
    maiter: (scalar)
    Lb: block length(scalar)
    alpha: forgetting rate(recursive processing)
    '''
    # (output)
    '''
    Y: Source of Interest(SOI)(F(=frequency) * N(= 1) * L(Time))
    N: num of SOI, in CSV model, N=1
    '''
    # X & Y: (time-frequency domain)
    X = X.transpose(1,0,2)
    # (Algorithm)
    F, M, L = X.shape
    T = L // Lb  # num of blocks
    N = 1  # num of SOI

    # initialize
    # 1) to store X into several block ---- XBlock
    XBlock = np.zeros((F, T, M, Lb), dtype=np.complex128)  # (F * T * M * Lb)
    for t in range(T):
        XBlock[:, t, :, :] = X[:, :, t * Lb:(t + 1) * Lb]
    # 2) Y (F(= frequency) * N(= 1 SOI) * L(= times frames))
    Y = np.zeros((F, N, L), dtype=np.complex128)
    # 3) deming vector w
    # Ht(W): (F * N * M)
    w = np.tile(np.eye(M, dtype=np.complex128), (F, 1, 1))[:, :, 0, None]  # (F * M * N)
    # w = np.zeros((F, M, 1), dtype=np.complex128)
    # for i in range(M):
    #     w[i, i, 0] = 1
    # 4) inter-values in the online recursive process P Q R
    P = 1e-12*np.tile(np.eye(M, dtype=np.complex128), (T, F, 1, 1))  # (T * F * M * M)
    # Q = np.tile(np.eye(N, dtype=np.complex128), (T, F, M, 1))  # (T * F * M * N)  Qk = (1, 1, 1, 1).T
    Q = np.tile(np.zeros(N, dtype=np.complex128), (T, F, M, 1))  # (T * F * M * N)  Qk = (0, 0, 0, 0).T
    # Q = np.tile(np.eye(M, dtype=np.complex128)[:,0,None], (T, F, 1, 1))  # (T * F * M * N) Qk = (1, 0, 0, 0).T
    R = 1e-12*np.tile(np.eye(M, dtype=np.complex128), (T, F, 1, 1))  # (T * F * M * M)
    a_block = np.zeros_like(Q) # (T * F * M * N) ak = (0, 0, 0, 0).T

    # for each of block
    for t in range(T):
        lamda = t + 1
        # (Updating)
        # calculate the covariances matrix block by block
        C_Current = XBlock[:,t,:,:]@Ht(XBlock[:,t,:,:])/Lb  # (F * M * M)
        # (iteration in each of block)
        for i in range(maxiter):
            time_st = time.time()
            Cw = C_Current@w  # (F * M * N)
            sgm2 = Ht(w)@Cw  # (F * N * N)
            r_Current = np.sqrt(np.sum(np.abs(Ht(w)@XBlock[:,t,:,:])**2, axis=0))  # (N * Lb)
            a_Current = Cw/sgm2  # (F * M * N)
            V_Current = (XBlock[:, t, :, :]/r_Current)@Ht(XBlock[:, t, :, :])/Lb  # (F * M * M)

            # 原版 recursive way
            P_Current = V_Current / sgm2  # (F * M * M)  P[t] = alpha * P[t-1] + (1 - alpha) * P_Current # (F * M * M)
            P[t] = alpha * P[t - 1] + (1 - alpha) * P_Current  # (F * M * M)  P:(T * F * M * M)

            Q_Current = ((Ht(w) @ V_Current @ w) / sgm2) * a_Current  # (F * M * N)  Q[t] = alpha * Q[t-1] + (1 - alpha) * Q_Current
            Q[t] = alpha * Q[t - 1] + (1 - alpha) * Q_Current  # (F * M * N)
            R[t] = alpha * R[t - 1] + (1 - alpha) * V_Current  # (F * M * M)
            w = np.linalg.inv(P[t]) @ Q[t]  # (F * M * N)
            w = w / (np.sqrt(Ht(w) @ R[t] @ w))  # (F * M * N)
            time_ed = time.time() - time_st
            time_list.append(time_ed)
        # (Enhancement)
        a_block[t] = a_Current
        Ytemp = Ht(w) @ XBlock[:, t, :, :]  # (F * N * Lb)  a block representation of Y
        Y[:, :, t * Lb:(t + 1) * Lb] = a_Current[:, 0, None, :] * Ytemp  # online projection back
        # print(t, 'w', w[0])

    return Y,w,time_list

def OGCCSVAuxIVE(X,sv, lamb_gc_ur=3.0, Lb = 100, alpha = 0.96, maxiter=10):
    time_list = []
    lamb_target = 10
    lamb_interfer = 1000
    lamb_s = 1e-6
    c = np.eye(1)
    # (input)
    '''
    X: microphone signal (F(=frequency) * M(=Microphone) * L(Time))
    maiter: (scalar)
    Lb: block length(scalar)
    alpha: forgetting rate(recursive processing)
    '''
    # (output)
    '''
    Y: Source of Interest(SOI)(F(=frequency) * N(= 1) * L(Time))
    N: num of SOI, in CSV model, N=1
    '''
    # X & Y: (time-frequency domain)
    X = X.transpose(1, 2, 0)
    # (Algorithm)
    F, M, L = X.shape
    T = L // Lb  # num of blocks
    N = 1  # num of SOI
    
    # initialize
    # 1) to store X into several block ---- XBlock
    XBlock = np.zeros((F, T, M, Lb), dtype=np.complex128)  # (F * T * M * Lb)
    for t in range(T):
        XBlock[:, t, :, :] = X[:, :, t * Lb:(t + 1) * Lb]
    # 2) Y (F(= frequency) * N(= 1 SOI) * L(= times frames))
    Y = np.zeros((F, N, L), dtype=np.complex128)
    # 3) deming vector w
    # Ht(W): (F * N * M)
    w = np.tile(np.eye(M, dtype=np.complex128), (F, 1, 1))[:, :, 0, None]  # (F * M * N)
    # w = np.zeros((F, M, 1), dtype=np.complex128)
    # for i in range(M):
    #     w[i, i, 0] = 1
    # 4) inter-values in the online recursive process P Q R
    P = 1e-12 * np.tile(np.eye(M, dtype=np.complex128), (T, F, 1, 1))  # (T * F * M * M)
    # Q = np.tile(np.eye(N, dtype=np.complex128), (T, F, M, 1))  # (T * F * M * N)  Qk = (1, 1, 1, 1).T
    Q = np.tile(np.zeros(N, dtype=np.complex128), (T, F, M, 1))  # (T * F * M * N)  Qk = (0, 0, 0, 0).T
    # Q = np.tile(np.eye(M, dtype=np.complex128)[:,0,None], (T, F, 1, 1))  # (T * F * M * N) Qk = (1, 0, 0, 0).T
    R = 1e-12 * np.tile(np.eye(M, dtype=np.complex128), (T, F, 1, 1))  # (T * F * M * M)
    a_block = np.zeros_like(Q)  # (T * F * M * N) ak = (0, 0, 0, 0).T
    ########################################################
    # Compute geometric constraint terms
    urlambddH = np.zeros([F, M, M], 'c16')  # F x M x M
    lambcd = np.zeros([F, M, 1], 'c16')  # F x M x 1
    for i in range(sv.shape[2]):
        urlambddH += lamb_gc_ur * sv[:, :, i, None] @ Ht(sv[:, :, i, None])  # F x M x M
        lambcd += lamb_gc_ur * c * sv[:, :, i, None]  # F x M x 1

    # Incorporate geometric constraints when updating w
    # for each of block
    for t in range(T):
        lamda = t + 1
        # (Updating)
        # calculate the covariances matrix block by block
        C_Current = XBlock[:, t, :, :] @ Ht(XBlock[:, t, :, :]) / Lb  # (F * M * M)
        # (iteration in each of block)
        for i in range(maxiter):
            time_st = time.time()
            Cw = C_Current @ w  # (F * M * N)
            sgm2 = Ht(w) @ Cw  # (F * N * N)
            r_Current = np.sqrt(np.sum(np.abs(Ht(w) @ XBlock[:, t, :, :]) ** 2, axis=0))  # (N * Lb)
            a_Current = Cw / sgm2  # (F * M * N)
            V_Current = (XBlock[:, t, :, :] / r_Current) @ Ht(XBlock[:, t, :, :]) / Lb  # (F * M * M)
            
            # 原版 recursive way
            P_Current = V_Current / sgm2 + urlambddH[:,None,:] # (F * M * M)  P[t] = alpha * P[t-1] + (1 - alpha) * P_Current # (F
            # * M * M)
            P[t] = alpha * P[t - 1] + (1 - alpha) * P_Current  # (F * M * M)  P:(T * F * M * M)
            
            Q_Current = ((Ht(w) @ V_Current @ w) / sgm2) * a_Current + + lambcd[:,None,:]
            # (F * M * N)  Q[t] =
            # alpha * Q[
            # t-1] + (
            # 1 -
            # alpha) * Q_Current
            Q[t] = alpha * Q[t - 1] + (1 - alpha) * Q_Current  # (F * M * N)
            R[t] = alpha * R[t - 1] + (1 - alpha) * V_Current  # (F * M * M)

            tmpP = P[t]
            tmpQ = Q[t]
            P_inv = np.linalg.inv(tmpP) if np.all(np.linalg.det(tmpP) != 0) else np.linalg.inv(tmpP + eps * np.eye(M))
            
            w = P_inv @ tmpQ  # (F * M * N)
            w = w / (np.sqrt(Ht(w) @ R[t] @ w))  # (F * M * N)
            time_ed = time.time() - time_st
            time_list.append(time_ed)
        # (Enhancement)
        a_block[t] = a_Current
        Ytemp = Ht(w) @ XBlock[:, t, :, :]  # (F * N * Lb)  a block representation of Y
        Y[:, :, t * Lb:(t + 1) * Lb] = a_Current[:, 0, None, :] * Ytemp  # online projection back
        # print(t, 'w', w[0])
    
    return Y, w, time_list

def OFBRCSVAuxIVE(Bfl, X, sv,Gamma_target, Gamma_interfer, alpha = 0.96, maxiter=10):
    time_list = []
    Lb = Bfl
    lamb_target = 10
    lamb_interfer = 1000
    lamb_s = 1e-6
    c = np.eye(1)
    # (input)
    '''
    X: microphone signal (F(=frequency) * M(=Microphone) * L(Time))
    maiter: (scalar)
    Lb: block length(scalar)
    alpha: forgetting rate(recursive processing)
    '''
    # (output)
    '''
    Y: Source of Interest(SOI)(F(=frequency) * N(= 1) * L(Time))
    N: num of SOI, in CSV model, N=1
    '''
    # X & Y: (time-frequency domain)
    X = X.transpose(1, 2, 0)
    # (Algorithm)
    F, M, L = X.shape
    T = L // Lb  # num of blocks
    N = 1  # num of SOI
    
    # initialize
    # 1) to store X into several block ---- XBlock
    XBlock = np.zeros((F, T, M, Lb), dtype=np.complex128)  # (F * T * M * Lb)
    for t in range(T):
        XBlock[:, t, :, :] = X[:, :, t * Lb:(t + 1) * Lb]
    # 2) Y (F(= frequency) * N(= 1 SOI) * L(= times frames))
    Y = np.zeros((F, N, L), dtype=np.complex128)
    # 3) deming vector w
    # Ht(W): (F * N * M)
    w = np.tile(np.eye(M, dtype=np.complex128), (F, 1, 1))[:, :, 0, None]  # (F * M * N)
    # w = np.zeros((F, M, 1), dtype=np.complex128)
    # for i in range(M):
    #     w[i, i, 0] = 1
    # 4) inter-values in the online recursive process P Q R
    P = 1e-12 * np.tile(np.eye(M, dtype=np.complex128), (T, F, 1, 1))  # (T * F * M * M)
    # Q = np.tile(np.eye(N, dtype=np.complex128), (T, F, M, 1))  # (T * F * M * N)  Qk = (1, 1, 1, 1).T
    Q = np.tile(np.zeros(N, dtype=np.complex128), (T, F, M, 1))  # (T * F * M * N)  Qk = (0, 0, 0, 0).T
    # Q = np.tile(np.eye(M, dtype=np.complex128)[:,0,None], (T, F, 1, 1))  # (T * F * M * N) Qk = (1, 0, 0, 0).T
    R = 1e-12 * np.tile(np.eye(M, dtype=np.complex128), (T, F, 1, 1))  # (T * F * M * M)
    a_block = np.zeros_like(Q)  # (T * F * M * N) ak = (0, 0, 0, 0).T
    FBR = np.zeros_like(Gamma_interfer)
    # for each of block
    for t in range(T):
        FBR[:,:,:,t] = lamb_interfer * Gamma_interfer[:,:,:, t] - lamb_target * Gamma_target[:,:, :,t] + lamb_s * np.tile(np.eye(M), (1025, 1, 1))
        # + lamb_s * np.eye(M)
        lamda = t + 1
        # (Updating)
        # calculate the covariances matrix block by block
        C_Current = XBlock[:, t, :, :] @ Ht(XBlock[:, t, :, :]) / Lb  # (F * M * M)
        # (iteration in each of block)
        for i in range(maxiter):
            time_st = time.time()
            Cw = C_Current @ w  # (F * M * N)
            sgm2 = Ht(w) @ Cw  # (F * N * N)
            r_Current = np.sqrt(np.sum(np.abs(Ht(w) @ XBlock[:, t, :, :]) ** 2, axis=0))  # (N * Lb)
            a_Current = Cw / sgm2  # (F * M * N)
            V_Current = (XBlock[:, t, :, :] / r_Current) @ Ht(XBlock[:, t, :, :]) / Lb  # (F * M * M)
            
            # 原版 recursive way
            P_Current = V_Current / sgm2 + FBR[:,:,:,t] # (F * M * M)  P[t] = alpha * P[t-1] + (1 - alpha) * P_Current # (F
            # * M * M)
            P[t] = alpha * P[t - 1]  + (1 - alpha) * P_Current  # (F * M * M)  P:(T * F * M * M)
            
            Q_Current = ((  Ht(w) @ V_Current @ w) / sgm2) * a_Current + 2*c*sv[:,:,t].reshape(1025,6,1)
  # (F * M * N)  Q[t] =
            # alpha * Q[
            # t-1] + (
            # 1 -
            # alpha) * Q_Current
            Q[t] = alpha * Q[t - 1] + (1 - alpha) * Q_Current  # (F * M * N)
            R[t] = alpha * R[t - 1] + (1 - alpha) * V_Current  # (F * M * M)
            # ########################################################
            # # Compute geometric constraint terms
            # urlambddH = np.zeros([F, M, M], 'c16')  # F x M x M
            # lambcd = np.zeros([F, M, 1], 'c16')  # F x M x 1
            # for i in range(sv.shape[2]):
            #     urlambddH += lamb_gc_ur * svur[:, :, i, None] @ Ht(svur[:, :, i, None])  # F x M x M
            #     lambcd += lamb_gc_ur * c * svur[:, :, i, None]  # F x M x 1
            #
            # # Incorporate geometric constraints when updating w
            tmpP = P[t]
            tmpQ = Q[t]
            P_inv = np.linalg.inv(tmpP) if np.all(np.linalg.det(tmpP) != 0) else np.linalg.inv(tmpP + eps * np.eye(M))

            w = P_inv @ tmpQ  # (F * M * N)
            w = w / (np.sqrt(Ht(w) @ R[t] @ w))  # (F * M * N)
            time_ed = time.time() - time_st
            time_list.append(time_ed)
        # (Enhancement)
        a_block[t] = a_Current
        Ytemp = Ht(w) @ XBlock[:, t, :, :]  # (F * N * Lb)  a block representation of Y
        Y[:, :, t * Lb:(t + 1) * Lb] = a_Current[:, 0, None, :] * Ytemp  # online projection back
        # print(t, 'w', w[0])
    
    return Y, w, time_list
