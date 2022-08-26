import numpy as np
import math
from numpy.linalg import inv
import torch

"""
UAV black box model with dynamics from thrid example in:
A. Khazraei, S. Hallyburton, Q. Gao, Y. Wang and M. Pajic, 
"Learning-Based Vulnerability Analysis of Cyber-Physical Systems," 
2022 ACM/IEEE 13th International Conference on Cyber-Physical Systems (ICCPS), 
2022, pp. 259-269, doi: 10.1109/ICCPS54341.2022.00030
"""

def run_uav(times, signals, step=0.01, 
    max_time = 25.0, sims=1, param1 = 1.0, etaMax = 0.3019215166568756, epsilon = 0.01):

    """
    Input
    times:
    signals:
    step: time stpe delta, default to 0.01
    max_tim : finite time horizon for simulations
    sims : number of simulations to be performed to averging/stats
    param1 : dummy parameter
    etaMax : Cuttoff eta value for anomaly detection function

    Output
    Result Numpy Array for Psy-Taliro to process with the following data

    result[:,0] : Time, equally spaced, determined by dt
    result[:,1] : Anomaly rate
    result[:,2] : Error mean
    result[:,3] : Residue mean
    result[:,4] : Num of systems in error

    result[:,5] : Attack signal input 0
    result[:,6] : Attack signal input 1
    result[:,7] : Attack signal input 2
    result[:,8] : Attack signal input 3
    result[:,9] : Attack signal input 4
    result[:,10] : Attack signal input 5
    REMOVED result[:,11] : Attack signal input 6
    REMOVED result[:,12] : Attack signal input 7
    REMOVED result[:,13] : Attack signal input 8
    result[:,11] : Attack magnitude

    result[:, 12] : expected x[0] phi
    result[:, 13] : expected x[1] phi dot
    result[:, 14] : expected x[2] theta
    result[:, 15] : expected x[3] theta dot
    result[:, 16] : expected x[4] psi
    result[:, 17] : expected x[5] psi dot
    result[:, 18] : expected x[6] z
    result[:, 19] : expected x[7] z dot
    result[:, 20] : expected x[8] x
    result[:, 21] : expected x[9] x dot
    result[:, 22] : expected x[10] y
    result[:, 23] : expected x[11] y dot

    result[:, 24] : expected x_hat[0] phi
    result[:, 25] : expected x_hat[1] phi dot
    result[:, 26] : expected x_hat[2] theta
    result[:, 27] : expected x_hat[3] theta dot
    result[:, 28] : expected x_hat[4] psi
    result[:, 29] : expected x_hat[5] psi dot
    result[:, 30] : expected x_hat[6] z
    result[:, 31] : expected x_hat[7] z dot
    result[:, 32] : expected x_hat[8] x
    result[:, 33] : expected x_hat[9] x dot
    result[:, 34] : expected x_hat[10] y
    result[:, 35] : expected x_hat[11] y dot

    result[:, 36] : expected anomaly function

    RESULTS FOR 0TH SIM

    result[:, 37] : 0 x[0] phi
    result[:, 38] : 0 x[1] phi dot
    result[:, 39] : 0 x[2] theta
    result[:, 40] : 0 x[3] theta dot
    result[:, 41] : 0 x[4] psi
    result[:, 42] : 0 x[5] psi dot
    result[:, 43] : 0 x[6] z
    result[:, 44] : 0 x[7] z dot
    result[:, 45] : 0 x[8] x
    result[:, 46] : 0 x[9] x dot
    result[:, 47] : 0 x[10] y
    result[:, 48] : 0 x[11] y dot

    result[:, 49] : 0 x_hat[0] phi
    result[:, 50] : 0 x_hat[1] phi dot
    result[:, 51] : 0 x_hat[2] theta
    result[:, 52] : 0 x_hat[3] theta dot
    result[:, 53] : 0 x_hat[4] psi
    result[:, 54] : 0 x_hat[5] psi dot
    result[:, 55] : 0 x_hat[6] z
    result[:, 56] : 0 x_hat[7] z dot
    result[:, 57] : 0 x_hat[8] x
    result[:, 58] : 0 x_hat[9] x dot
    result[:, 59] : 0 x_hat[10] y
    result[:, 60] : 0 x_hat[11] y dot

    result[:,61] : 0th sim anomaly
    result[:,62] : 0th sim error
    result[:,63] : 0th sim residue


    trace_cols = ['Time','Anom_alarm_rate','Error_Mean', 'Res_Mean',
    'Attack_phi', 'Attack_phi_dot', 'Attack_theta', 'Attack_theta_dot',
    'Attack_psi', 'Attack_psi_dot', 'Attack_z', 'Attack_x', 'Attack_y', 'Attack_mag', 
    'E_phi','E_phi_dot','E_th','E_th_dot','E_psi','E_psi_dot',
    'E_z', 'E_z_dot','E_x', 'E_x_dot','E_y', 'E_y_dot',
    'E_phi_hat','E_phi_dot_hat','E_th_hat','E_th_dot_hat','E_psi_hat','E_psi_dot_hat'
    'E_z_hat', 'E_z_dot_hat','E_x_hat', 'E_x_dot_hat','E_y_hat', 'E_y_dot_hat', 
    '0_phi','0_phi_dot','0_th','0_th_dot','0_psi','0_psi_dot',
    '0_z', '0_z_dot','0_x', '0_x_dot','0_y', '0_y_dot',
    '0_phi_hat','0_phi_dot_hat','0_th_hat','0_th_dot_hat','0_psi_hat','0_psi_dot_hat'
    '0_z_hat', '0_z_dot_hat','0_x_hat', '0_x_dot_hat','0_y_hat', '0_y_dot_hat', 
    'Anomaly_gt','Error','Residue', "Anomaly_sig", "Error_sig"]


    """

    dt=step # Sampling time

    N = sims

    num_signals = len(signals)
    print("Number of input signals is: {} ".format(num_signals))

    T = math.ceil(max_time/step)

    result = np.zeros((T,64))

    cuda=False
    device = torch.device('cuda:0' if cuda else 'cpu')

    gen_r = torch.Generator(device = device)
    gen_r.manual_seed(66789) # This one is good

    l=.17
    b=3.13*.00001
    d=7.5*.0000001
    M=.38
    I_r=6*.00001
    I_x=.0086
    I_y=.0086
    I_z=0.0172
    g=-9.88
    n=12 # State (x) dimension
    m=5
    p=9 #Output/Measurement (y) dimension
    r=.01
    q=.01
    q1=0.001
    Q=torch.zeros(n,n).to(device) # The covariance of system noise
    Q[1,1]=q1
    Q[3,3]=q1
    Q[7,7]=q
    R=r*torch.eye(p).to(device) # The covariance of measurement noise

    a1 = (I_y-I_z)/I_x
    a2 = I_r/I_x
    a3 = (I_z-I_x)/I_y
    a4 = I_r/I_y
    a5 = (I_x-I_y)/I_z
    b1 = 1/I_x
    b2 = 1/I_y
    b3 = 1/I_z

    SS=torch.zeros([T,p]).to(device)

    k1 = -1.6308                           
    k2 = -1.4263
    k3 = -0.7374
    k4 = 4.8459
    k5 = 0.8135
    k6 = 1.6308                                                                             
    k7 =  1.4263
    k8 =  0.7374  
    k9 = 4.8459
    k10 = 0.8135 
    k11 = 4.0968      
    k12 = 2.8000 
    k13 = 2.1534
    k_14 = 5e-4
    k_15 = 0.01
    z_d = 10
    psi_d = 0
    x_d = 0
    y_d = 0

    x=torch.zeros([T,N,n]).to(device)
    s=torch.zeros([T,N,n]).to(device)
    x_hat=torch.zeros([T,N,n]).to(device)
    x_hat_con=torch.zeros([T,N,n]).to(device)
    u=torch.zeros([T,N,m]).to(device)
    y=torch.zeros([T,N,p]).to(device)
    z=torch.zeros([T,N,p]).to(device)
    P=torch.zeros([T,N,n,n]).to(device)
    F=torch.zeros([T,N,n,n]).to(device)
    P_con=torch.zeros([T,N,n,n]).to(device)
    S=torch.zeros([T,N,p,p]).to(device)
    L=torch.zeros([T,N,n,p]).to(device)
    C=torch.zeros([p,n]).to(device)
    C[0,0]=C[1,1]=C[2,2]=C[3,3]=C[4,4]=C[5,5]=C[6,6]=C[7,8]=C[8,10]=1
    ERR=torch.zeros([T,N]).to(device)

    gt=torch.zeros(T,N).to(device)
    gt_pre=torch.zeros([T,N,p]).to(device)
    # Res_n=torch.zeros(T,N).to(device)

    sig1=torch.zeros([T,N]).to(device)
    sig2=torch.zeros([T,N]).to(device)
    sig3=torch.zeros([T,N]).to(device)

    Res=torch.zeros(T,N).to(device)
    # beta=torch.zeros(T,N).to(device)

    mean_err_norm=torch.zeros(T)
    std_err_norm=torch.zeros(T)

    mean_res=torch.zeros(T)
    std_res=torch.zeros(T)

    x[-1,:,:]=torch.zeros(N,n)
    x_hat[-1,:,:]=x[-1,:,:]
    P[-1,:,:,:]=torch.zeros(n,n)

    for i in range(T):
        # System Dynamics; compute next ground state
        x[i,:,0]=x[i-1,:,0]+dt*(x[i-1,:,1])
        x[i,:,1]=x[i-1,:,1]+dt*(a1*x[i-1,:,3]*x[i-1,:,5]+a2*x[i-1,:,3]*u[i-1,:,4]+b1*u[i-1,:,1])+q1*torch.randn(N, generator=gen_r).to(device)
        x[i,:,2]=x[i-1,:,2]+dt*(x[i-1,:,3])
        x[i,:,3]=x[i-1,:,3]+dt*(a3*x[i-1,:,1]*x[i-1,:,5]+a4*x[i-1,:,1]*u[i-1,:,4]+b2*u[i-1,:,2])+q1*torch.randn(N, generator=gen_r).to(device)
        x[i,:,4]=x[i-1,:,4]+dt*(x[i-1,:,5])
        x[i,:,5]=x[i-1,:,5]+dt*(a5*x[i-1,:,1]*x[i-1,:,3]+b3*u[i-1,:,3])
        x[i,:,6]=x[i-1,:,6]+dt*(x[i-1,:,7])
        x[i,:,7]=x[i-1,:,7]+dt*(-g+(u[i-1,:,0]/M)*torch.cos(x[i-1,:,0])*torch.cos(x[i-1,:,2]))+q*torch.randn(N, generator=gen_r).to(device)
        x[i,:,8]=x[i-1,:,8]+dt*(x[i-1,:,9])
        x[i,:,9]=x[i-1,:,9]+dt*(u[i-1,:,0]/M)*(torch.cos(x[i-1,:,0])*torch.sin(x[i-1,:,2])*torch.cos(x[i-1,:,4])+torch.sin(x[i-1,:,0])*torch.sin(x[i-1,:,4]))
        x[i,:,10]=x[i-1,:,10]+dt*(x[i-1,:,11])
        x[i,:,11]=x[i-1,:,11]+dt*(u[i-1,:,0]/M)*(torch.cos(x[i-1,:,0])*torch.sin(x[i-1,:,2])*torch.sin(x[i-1,:,4])-torch.sin(x[i-1,:,0])*torch.cos(x[i-1,:,4]))
     
        # Comute measurement
        y[i,:,:]=torch.matmul(C,x[i,:,:].T).T+r*torch.randn(N,p, generator=gen_r).to(device)
     
        # First step of state estimate based on previous state control
        x_hat_con[i,:,0]=x_hat[i-1,:,0]+dt*(x_hat[i-1,:,1])
        x_hat_con[i,:,1]=x_hat[i-1,:,1]+dt*(a1*x_hat[i-1,:,3]*x_hat[i-1,:,5]+a2*x_hat[i-1,:,3]*u[i-1,:,4]+b1*u[i-1,:,1])
        x_hat_con[i,:,2]=x_hat[i-1,:,2]+dt*(x_hat[i-1,:,3])
        x_hat_con[i,:,3]=x_hat[i-1,:,3]+dt*(a3*x_hat[i-1,:,1]*x_hat[i-1,:,5]+a4*x_hat[i-1,:,1]*u[i-1,:,4]+b2*u[i-1,:,2])
        
        x_hat_con[i,:,4]=x_hat[i-1,:,4]+dt*(x_hat[i-1,:,5])
        x_hat_con[i,:,5]=x_hat[i-1,:,5]+dt*(a5*x_hat[i-1,:,1]*x_hat[i-1,:,3]+b3*u[i-1,:,3])
        x_hat_con[i,:,6]=x_hat[i-1,:,6]+dt*(x_hat[i-1,:,7])
        
        x_hat_con[i,:,7]=x_hat[i-1,:,7]+dt*(-g+(1/M)*torch.cos(x_hat[i-1,:,0])*torch.cos(x_hat[i-1,:,2])*u[i-1,:,0])
        x_hat_con[i,:,8]=x_hat[i-1,:,8]+dt*(x[i-1,:,9])
        x_hat_con[i,:,9]=x_hat[i-1,:,9]+dt*(1/M)*(torch.cos(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,2])*torch.cos(x_hat[i-1,:,4])+torch.sin(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,4]))*u[i-1,:,0]
        x_hat_con[i,:,10]=x_hat[i-1,:,10]+dt*(x_hat[i-1,:,11])
        x_hat_con[i,:,11]=x_hat[i-1,:,11]+dt*(1/M)*(torch.cos(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,2])*torch.sin(x_hat[i-1,:,4])-torch.sin(x_hat[i-1,:,0])*torch.cos(x_hat[i-1,:,4]))*u[i-1,:,0]

        # Controller dynamics
        z_13=dt*(a1*x_hat[i-1,:,5]+a2*u[i-1,:,4])
        z_15=dt*(a1*x_hat[i-1,:,3])
        z_31=dt*(a3*x_hat[i-1,:,5]+a4*u[i-1,:,4])
        z_35=dt*(a3*x_hat[i-1,:,1])
        z_51=dt*(a5*x_hat[i-1,:,3])
        z_53=dt*(a5*x_hat[i-1,:,1])
        z_70=-dt*(1/M)*torch.sin(x_hat[i-1,:,0])*torch.cos(x_hat[i-1,:,2])*u[i-1,:,0]
        z_72=-dt*(1/M)*torch.cos(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,2])*u[i-1,:,0]
        z_90=dt*(torch.cos(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,4])-torch.sin(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,2])*torch.cos(x_hat[i-1,:,4]))*(1/M)*u[i-1,:,0]
        z_92=dt*(torch.cos(x_hat[i-1,:,0])*torch.cos(x_hat[i-1,:,2])*torch.cos(x_hat[i-1,:,4]))*(1/M)*u[i-1,:,0]
        z_94=dt*(torch.sin(x_hat[i-1,:,0])*torch.cos(x_hat[i-1,:,4])-torch.cos(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,2])*torch.sin(x_hat[i-1,:,4]))*(1/M)*u[i-1,:,0]
        
        
        z_110=dt*(-torch.sin(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,2])*torch.sin(x_hat[i-1,:,4])-torch.cos(x_hat[i-1,:,0])*torch.cos(x_hat[i-1,:,4]))*(1/M)*u[i-1,:,0]
        z_112=dt*(torch.cos(x_hat[i-1,:,0])*torch.cos(x_hat[i-1,:,2])*torch.sin(x_hat[i-1,:,4]))*(1/M)*u[i-1,:,0]
        z_114=dt*(torch.cos(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,2])*torch.cos(x_hat[i-1,:,4])+torch.sin(x_hat[i-1,:,0])*torch.sin(x_hat[i-1,:,4]))*(1/M)*u[i-1,:,0]
        
        # Controller dynamics
        for j in range(N):
            
            F[i,j,:,:]=torch.tensor([[1,dt,0,0,0,0,0,0,0,0,0,0],
                                   [0,1,0,z_13[j],0,z_15[j],0,0,0,0,0,0],
                                   [0,0,1,dt,0,0,0,0,0,0,0,0],
                                   [0,z_31[j],0,1,0,z_35[j],0,0,0,0,0,0],
                                   [0,0,0,0,1,dt,0,0,0,0,0,0],
                                   [0,z_51[j],0,z_53[j],0,1,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,1,dt,0,0,0,0],
                                   [z_70[j],0,z_72[j],0,0,0,0,1,0,0,0,0],
                                   [0,0,0,0,0,0,0,0,1,dt,0,0],
                                   [z_90[j],0,z_92[j],0,z_94[j],0,0,0,0,1,0,0],
                                   [0,0,0,0,0,0,0,0,0,0,1,dt],
                                   [z_110[j],0,z_112[j],0,z_114[j],0,0,0,0,0,0,1]])
        
        # Kalman Filter Matrices
        P_con[i,:,:,:]=torch.matmul(torch.matmul(F[i,:,:,:],P[i-1,:,:,:]),torch.transpose(F[i,:,:,:], 1, 2))+Q

        S[i,:,:,:]=torch.matmul(torch.matmul(C,P_con[i,:,:,:]),C.T)+R
        L[i,:,:,:]=torch.matmul(torch.matmul(P_con[i,:,:,:],C.T),torch.inverse(S[i,:,:,:]))
        P[i,:,:,:]=torch.matmul(torch.eye(n).to(device)-torch.matmul(L[i,:,:,:],C),P_con[i,:,:,:])

        # System specific dynamic
        y[i,:,6]=y[i,:,6]-z_d
        x_hat[i-1,:,6]=x_hat[i-1,:,6]-z_d
        SS=torch.cat((y[i,:,:][ None,:,:],x_hat[i-1,:,:][ None,:,:]),2)
        y[i,:,6]=y[i,:,6]+z_d
        x_hat[i-1,:,6]=x_hat[i-1,:,6]+z_d

        # z[i,:,:]=y[i,:,:]+model(SS).detach()[0,:,:]-torch.matmul(C,x_hat_con[i,:,:].T).T

        # Attack is introduced here

        # y[i,:,:] = y[i,:,:]+signals[:,i]

        # for k in range(num_signals):
        #    y[i,:,k] = y[i,:,k]+signals[k,i]

        y[i,:,0] = y[i,:,0]+signals[0,i]
        # y[i,:,1] = y[i,:,1]+signals[1,i]
        y[i,:,2] = y[i,:,2]+signals[1,i]
        # y[i,:,3] = y[i,:,3]+signals[3,i]
        y[i,:,4] = y[i,:,4]+signals[2,i]
        # y[i,:,5] = y[i,:,5]+signals[5,i]
        y[i,:,6] = y[i,:,6]+signals[3,i]
        y[i,:,7] = y[i,:,7]+signals[4,i]
        y[i,:,8] = y[i,:,8]+signals[5,i]
        
        #Residue: measurement - estimated measurement
        z[i,:,:]=y[i,:,:]-torch.matmul(C,x_hat_con[i,:,:].T).T
        # print("z size is: ")
        # print(z[j,:,:].size())
        # print(z[j,:,:])
        # print(S[j,:,:,:])
        
        # zz=z[i,:,:].view( N, p, 1)
        zz=z.view(T, N, p, 1).to(device)
        # print('zz size is')
        # print(zz.size())
        # print(zz)
        # print(zz.T)
        # x_hat[j,:,:]=x_hat_con[j,:,:]+torch.matmul(L[j,:,:,:],zz[j,:,:,:]).view( N, p)
        
        
        # print("x_hat_con i is size is:")
        # print(x_hat_con[i,:,:].size())
        
        # print("L i is size is:")
        # print(L[i,:,:,:].size())
        
        # print("zz i is size is:")
        # print(zz[i,:,:,:].size())
        
        # print("L*zz i is size is:")
        # print(torch.matmul(L[i,:,:,:],zz[i,:,:,:]).size())
        
        # print("L*zz i reshaped is size is:")
        # print(torch.matmul(L[i,:,:,:],zz[i,:,:,:]).view( N, n).size())
        
        # Next steps of Kalman filter state estimate
        x_hat[i,:,:]=x_hat_con[i,:,:]+torch.matmul(L[i,:,:,:],zz[i,:,:,:]).view( N, n)    

        # Controller 

        sig1[i,:]=sig1[i-1,:]+dt*(x_hat[i,:,8]-x_d)
        sig2[i,:]=sig2[i-1,:]+dt*(x_hat[i,:,10]-y_d)
        sig3[i,:]=sig3[i-1,:]+dt*(x_hat[i,:,6]-z_d)
        u[i,:,2]=-(1/b2)*a3*x_hat[i-1,:,1]*x_hat[i-1,:,5]-k1*(x_hat[i,:,8]-x_d)-k2*x_hat[i,:,9]-k3*sig1[i,:]-k4*x_hat[i,:,2]-k5*x_hat[i,:,3]
        
        u[i,:,1]=-k6*(x_hat[i,:,10]-y_d)-k7*x_hat[i,:,11]-k8*sig2[i,:]-k9*x_hat[i,:,0]-k10*x_hat[i,:,1]
        
        u[i,:,0]=((1)/(torch.cos(x_hat[i,:,0])*torch.cos(x_hat[i,:,2])))*(M*g-k11*(x_hat[i,:,6]-z_d)-k12*x_hat[i,:,7]-k13*sig3[i,:])
        

        # print("x[i,:,:] is:")
        # print(x[i,:,:])

        # print("x_hat[i,:,:] is:")
        # print(x_hat[i,:,:])

        # System Error
        ERR[i,:]=torch.norm(x[i,:,:]-x_hat[i,:,:], dim=1)

        # print("ERR[i,:] is:")
        # print(ERR[i,:])

        # System residue
        Res[i,:]=torch.norm(z[i,:,:], dim=1)
        
        mean_err_norm[i]=torch.mean(ERR[i,:])
        std_err_norm[i]=torch.std(ERR[i,:])
        
        mean_res[i]=torch.mean(Res[i,:])
        std_res[i]=torch.std(Res[i,:])
        
        # Testing:

        # print("S i is size is:")
        # print(S[i,:,:,:].size())
        
        # print("zz i is size is:")
        # print(zz[i,:,:,:].size())
        
        # print("Sinv*zz i is size is:")
        # print(torch.matmul(torch.inverse(S[i,:,:,:]),zz[i,:,:,:]).size())
        
        # print("Sinv*zz i reshaped is size is:")
        # print(torch.matmul(torch.inverse(S[i,:,:,:]),zz[i,:,:,:]).view(N, p))
        
        # End testing
        
        # Anomaly Function Calculated (Not in original Duke code)
        # gt = z^T S^-1 z
        gt_pre[i,:,:] = torch.matmul(torch.inverse(S[i,:,:,:]),zz[i,:,:,:]).view(N, p) 
        gt_pre2=gt_pre[i,:,:].view(N, p, 1).to(device)
        zzT=z.view(T, N, 1, p).to(device)

        gt[i,:]=torch.matmul(zzT[i,:,:,:],gt_pre2).view(N)

        result[i,0] = i*dt
        # result[ii,1] = param1*dt*ii + signals[:,ii][0]
        # print(np.array(x[:, :, 0]).flatten().shape)
        # print(result[:,1].shape) 

        # END OF SIMULATION TIME LOOP     
    
    
    # Compute boolean if anomaly exceeded eta
    gt_over = gt>etaMax
    # print("gt_over size is {}".format(gt_over.shape))
    # print("gt_over is:")
    # print(gt_over)


    # Compute the rate at which anomaly function exceed eta
    # over the N simulations
    error_rate = torch.count_nonzero(gt_over, dim=1)/float(N)
    # print("error_rate shape is {}".format(error_rate.shape))


    systems_werror = torch.count_nonzero(gt_over, dim=0)/float(T)
    systems_werror = systems_werror > epsilon
    systems_werror_num = torch.count_nonzero(systems_werror)
    # print("systems_werror size is {}".format(systems_werror.shape))
    # print("systems_werror is:")
    # print(systems_werror)

    print("systems_werror_num is:")
    print(systems_werror_num.item())


    # systems_werror_over = systems_werror>0.01

    # Save Mean statistics over the 'sims' number of simulations

    # Save the anomaly/error rate
    print("Anomaly rate (error_rate):")
    result[:,1] = np.array(error_rate[:])
    print(result[:,1])

    # Save the Error Mean (expected value)
    ERR_mean = torch.mean(ERR, dim=1)
    # print("ERR_mean array shape is:")
    # print(ERR_mean.shape)
    # print("Printing ERR_mean")
    # print(ERR_mean)
    result[:,2] = np.array(ERR_mean[:])

    # Save the mean of the Residue (expected value)
    Res_mean = torch.mean(Res, dim=1)
    # print("Res_mean array shape is:")
    # print(Res_mean.shape)
    # print("Printing Res_mean")
    # print(Res_mean)
    result[:,3] = np.array(Res_mean[:])


    # Save the number of systems in error
    cum_index = 4
    result[:,cum_index] = systems_werror_num
    print("Number of systems of error in result:")
    print(result[:,4])
    cum_index += 1


    # Save the attack and Attack Magnitude
    poppop = np.zeros((T,1)) # Magnitude

    for k in range(num_signals):
        result[:,cum_index] = signals[k]
        poppop[:,0] += np.square(signals[k])
        cum_index += 1

    #cum_index = num_signals+cum_index

    poppop[:,0] = np.sqrt(poppop[:,0]) # Magnitude

    result[:,cum_index] = poppop[:,0] # Magnitude

    cum_index += 1


    # Save expected state trajectories

    # result[:,cum_index] = np.array(torch.mean(x[:, :, 6], dim=1))
    # print("Expected z coordinate is:")
    # print(result[:,cum_index])
    # print(result[:,cum_index].shape)
    # cum_index += 1

    for kk in range(n):
        result[:,cum_index] = np.array(torch.mean(x[:, :, kk], dim=1))
        cum_index += 1

    # Save expected state estimation trajectories

    for kk in range(n):
        result[:,cum_index] = np.array(torch.mean(x_hat[:, :, kk], dim=1))
        cum_index += 1

    # Save anomaly function expectation value
    result[:,cum_index] = np.array(torch.mean(gt[:,:], dim=1))
    cum_index += 1



    # Save state trajectories for 0th simulation
    for kk in range(n):
        result[:,cum_index] = (np.array(x[:, 0, kk])).flatten()
        cum_index += 1

    # Save state estimation trajectories for 0th simulation

    # print(cum_index)

    for kk in range(n):
        result[:,cum_index] = (np.array(x_hat[:, 0, kk])).flatten()
        cum_index += 1

    # Save anomaly function value for 0th sim
    result[:,cum_index] = gt[:,0]
    # print("gt shape is:")
    # print(gt.shape)
    # print("Printing gt")
    # print(gt)
    print("0th Anomaly value:")
    print(result[:,cum_index])
    cum_index += 1

    # Save error for 0th sim
    result[:,cum_index] = np.array(ERR[:,0])
    print("0th System Error value:")
    print(result[:,cum_index])
    # print("ERR array shape is:")
    # print(ERR.shape)
    # print("Printing ERR")
    # print(ERR)
    cum_index += 1
   
    # Save resiude for 0th sim
    result[:,cum_index] = np.array(Res[:,0])
    print("0th Residue:")
    print(result[:,cum_index])
    cum_index += 1





    # Save Traces of particular simulation
    
    # # Save State for 0th sim
    # result[:,14] = (np.array(x[:, 0, 6])).flatten() # z coordinate
    # result[:,15] = (np.array(x[:, 0, 8])).flatten() # x coord
    # result[:,16] = (np.array(x[:, 0, 10])).flatten() # y coord

    # # Save State Estimate for 0th sim
    # result[:,17] = (np.array(x_hat[:, 0, 6])).flatten() # z coord
    # result[:,18] = (np.array(x_hat[:, 0, 8])).flatten() # x coord
    # result[:,19] = (np.array(x_hat[:, 0, 10])).flatten() # y coord

    

    print('result shape:')
    print(result.shape)
    print('final cum_index is {}'.format(cum_index))

    return result