import logging
import math
from random import sample
import numpy as np
import pandas as pd
# import plotly.graph_objects as go
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import datetime
import time
import os

# Uncomment if creating gifs
# import matplotlib.animation as animation
# from matplotlib.animation import FuncAnimation, PillowWriter 

from staliro.core.model import Failure
from staliro.models import ModelData, SignalTimes, SignalValues, StaticInput, blackbox
from staliro.optimizers import DualAnnealing, UniformRandom
from staliro.options import Options, SignalOptions
from staliro.specifications import RTAMTDense, RTAMTDiscrete, TLTK
from staliro.staliro import simulate_model, staliro
from staliro.core.interval import Interval
from staliro.signals import delayed, pchip, harmonic

import uav_v3_model_seeded

from helpers.sampling import uniform_sampling
from helpers.utils import Fn
from helpers.bayesianOptimization import InternalBO
from helpers.utils.computeRobustness import compute_robustness
from helpers.gprInterface import InternalGPR

test_flag = False

start = time.time()
ct = datetime.datetime.now()

directory_path = os.getcwd()
data_dir = "data"
path = os.path.join(directory_path, data_dir)
data_dir_run = "uav3_"+ct.strftime('%Y-%m-%d_%HH%MM%SS') + "/"

if test_flag:
    data_dir_run = "test_"+ data_dir_run

data_path = os.path.join(path, data_dir_run)
print(data_path)
os.makedirs(data_path, exist_ok = True)

simpleDataT = ModelData[NDArray[np.float_], None]

params = {
    "step" : 0.01, # Time step
    "sim_end_time" : 45.0,
    "eta" : 0.3019215166568756,
    "epsilon" : 0.01, # Epsilon from alpha-epsion attack, Anomaly rate corresponding to eta
    "alpha" : 0.05, # Alpha from alpha-epislon definition: desired threshold for the system error to cross
    # "No_control_pts" : 7,  $ No longer needed in psy-taliro v1.0.0b5
    "delay" : 10.0,
    # "error_duration" : 0.2,
    "num_signals" : 6,
    "control_points0": [(-0.02, 0.02),
    (-0.04, 0.04),(0.0, 200.0),(0.0, math.pi)],
    "control_points1": [(-0.02, 0.02),
    (-0.04, 0.04),(0.0, 200.0),(0.0, math.pi)],
    "control_points2": [(-0.02, 0.02),
    (-0.04, 0.04),(0.0, 200.0),(0.0, math.pi)],
    "control_points3": [(-0.02, 0.02),
    (-0.04, 0.04),(0.0, 20.0),(0.0, math.pi)],
    "control_points4": [(-0.02, 0.02),
    (-0.04, 0.04),(0.0, 20.0),(0.0, math.pi),
    (-0.04, 0.04),(0.0, 200.0),(0.0, math.pi)],
    "control_points5": [(-0.02, 0.02),
    (-0.04, 0.04),(0.0, 20.0),(0.0, math.pi)],
    # "control_points6": [(-0.01, 0.01),
    # (-0.02, 0.02),(0.0, 200.0),(0.0, math.pi)],
    # "control_points7": [(-0.01, 0.01),
    # (-0.02, 0.02),(0.0, 200.0),(0.0, math.pi)],
    # "control_points8": [(-0.01, 0.01),
    # (-0.02, 0.02),(0.0, 200.0),(0.0, math.pi)],
    # "cp_locations" : np.array([5,6,7,8,10,12,15]), # Used if Pchip signal
    "initial_conditions" : [],
    "runs" : 1,
    #"iterations" : 5000,
    "num_uniform_samples" : 100,
    "num_bo_samples" : 100,
}

@blackbox(sampling_interval = params["step"])
def uav_model(static: StaticInput, times: SignalTimes, 
   signals: SignalValues) -> simpleDataT:

    """


    """

    sims = 100

    if static.size > 0:
        param1 = static[0]
    else:
        param1 = 1.0
    
    result = uav_v3_model_seeded.run_uav(times, signals, max_time = max(times), 
        sims = sims, param1 = param1, etaMax = params["eta"], epsilon = params["epsilon"])

    # Define anomaly drugs

    logistic_anomrate = np.exp(-(result[:,1]-params["epsilon"]))
    logistic_anomrate = 1.0/(1.0 + logistic_anomrate)

    logistic_anomrate = np.reshape(logistic_anomrate,(logistic_anomrate.shape[0],1))

    print("logistic_anomrate shape is {}".format(logistic_anomrate.shape))
    #print("logistic_anomrate shape is {}".format(logistic_anomrate.shape[0]))

    # Define error on X coordinate

    error_x = np.absolute(result[:,20] - result[:,32])

    logistic_error = np.exp(-(error_x-params["alpha"]))
    logistic_error = 1.0/(1.0 + logistic_error)

    logistic_error = np.reshape(logistic_error,(logistic_error.shape[0],1))



    result = np.concatenate((result,logistic_anomrate), axis = 1)
    result = np.concatenate((result,logistic_error), axis = 1)

    trajectories: NDArray[np.float_] = result.T
    timestamps: NDArray[np.float_] = result[:,0]

    return ModelData(trajectories, timestamps)


# Specification with always and eventually

phi = "not ( (always[0, {}] (anomaly_sig <= 0.5)) and ( eventually[0, {}] (error_sig >= 0.5) ) )".format(
    params["sim_end_time"],params["sim_end_time"])
print("Specification: "+phi)
# specification = RTAMTDense(phi, {"anomaly_rate": 2, "error": 3})
specification = RTAMTDense(phi, {"anomaly_sig": 64, "error_sig": 65})

params["specification"] = phi

# Signal Input definition
signals = [SignalOptions(control_points=params["control_points0"], 
    factory=delayed(signal_factory = harmonic, delay=params["delay"])),
    SignalOptions(control_points=params["control_points1"], 
    factory=delayed(signal_factory = harmonic, delay=params["delay"])),
    SignalOptions(control_points=params["control_points2"], 
    factory=delayed(signal_factory = harmonic, delay=params["delay"])),
    SignalOptions(control_points=params["control_points3"], 
    factory=delayed(signal_factory = harmonic, delay=params["delay"])),
    SignalOptions(control_points=params["control_points4"], 
    factory=delayed(signal_factory = harmonic, delay=params["delay"])),
    SignalOptions(control_points=params["control_points5"], 
    factory=delayed(signal_factory = harmonic, delay=params["delay"])),
    # SignalOptions(control_points=params["control_points6"], 
    # factory=delayed(signal_factory = harmonic, delay=params["delay"])),
    # SignalOptions(control_points=params["control_points7"], 
    # factory=delayed(signal_factory = harmonic, delay=params["delay"])),
    # SignalOptions(control_points=params["control_points8"], 
    # factory=delayed(signal_factory = harmonic, delay=params["delay"])),
]

# Options
options = Options(interval=(0, params["sim_end_time"]), signals=signals, 
    static_parameters=params["initial_conditions"])


def generateRobustness(sample):
    options = Options(interval=(0, params["sim_end_time"]), signals=signals,
        static_parameters=params["initial_conditions"])

    result = simulate_model(uav_model, options, sample)

    rob = specification.evaluate(result.states, result.times)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Robustness value calculated by generateRobustness is {}".format(rob))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@")

    return rob


def generateTrace(sample):
    
    options = Options(interval=(0, params["sim_end_time"]), signals=signals,
        static_parameters=params["initial_conditions"])

    result = simulate_model(uav_model, options, sample)

    return result

def plotTrace_uav(sample):
    """
    Plots and saves the trace
    """
    
    result = generateTrace(sample)

    #Extract the best sample robustness
    rob = specification.evaluate(result.states, result.times)

    # Compute the integral of the attack for the best run
    # att_integral = result.states[5,:].sum()

    assert not isinstance(result, Failure)

    plt.style.use('seaborn')

    # Plot relevant variables for the best run
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10,14), tight_layout=True)
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)

    header = 'Simulation Aggregates'
    spec_str = 'Specification: '+phi
    rob_att_str = '\nRobustness={:.3e}, SR={}'.format(rob, int(100-result.states[4,0]))
    misc_params_str = 'eta={:.3f}'.format(params["eta"])
    title_str = header+'\n'+spec_str+'\n'+rob_att_str + '\n' + misc_params_str

    fig.suptitle(title_str)

    ax1.plot(result.times, result.states[11,:], '-')
    ax1.set_ylabel('Attack Magnitude')

    ax2.plot(result.times, result.states[1,:], '-')
    ax2.axhline(y=params["epsilon"], color='red', linestyle='--', label="epsilon")
    ax2.legend()
    ax2.set_ylabel('Anomaly Rate')

    ax3.plot(result.times, result.states[2,:], '-')
    # ax3.axhline(y=params["alpha"], color='red', linestyle='--', label="alpha")
    ax3.legend()
    ax3.set_ylabel('State Estimation Error Mean')

    # error_x = np.absolute(result[:,22] - result[:,34])
    # ax3.plot(result.times, error_x, '-')
    # ax3.axhline(y=params["alpha"], color='red', linestyle='--', label="alpha")
    # ax3.legend()
    # ax3.set_ylabel('X Error')

    

    ax4.plot(result.times, result.states[3,:], '-')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Residue Mean')

    #plt.show()
    filename = os.path.join(data_path, 'uav_means_{}.png'.format(ct.strftime('%Y-%m-%d_%HH%MM%SS')))
    plt.savefig(filename, format = 'png')
    plt.close()



    # Plot trace of 0th simulation of best sample
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, figsize=(10,14), tight_layout=True)
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)

    header = 'Expectation Values'
    spec_str = 'Specification: '+phi
    rob_att_str = '\nRobustness={:.3e}, SR={}'.format(rob, int(100-result.states[4,0]))
    misc_params_str = 'eta={:.3f}'.format(params["eta"])
    title_str = header+'\n'+spec_str+'\n'+rob_att_str + '\n' + misc_params_str

    fig.suptitle(title_str)

    ax1.plot(result.times, result.states[18,:], '-', label='z')
    ax1.plot(result.times, result.states[30,:], 'r', label='z_hat')
    ax1.legend()
    ax1.set_ylabel('Z')

    ax2.plot(result.times, result.states[20,:], '-', label='x')
    ax2.plot(result.times, result.states[32,:], 'r', label='x_hat')
    ax2.legend()
    ax2.set_ylabel('X')

    ax3.plot(result.times, result.states[22,:], '-', label='y')
    ax3.plot(result.times, result.states[34,:], 'r', label='y_hat')
    ax3.legend()
    ax3.set_ylabel('Y')

    #ax4.plot(result.times, result.states[20,:] - result.states[32,:], '-', label='z')
    ax4.plot(result.times, result.states[20,:] - result.states[32,:], 'r', label='x')
    ax3.axhline(y=params["alpha"], color='red', linestyle='--', label="alpha")
    #ax4.plot(result.times, result.states[24,:] - result.states[36,:], 'g', label='y')
    ax4.legend()
    ax4.set_ylabel('X Estimate Deviation')

    ax5.plot(result.times, result.states[11,:], '-')
    ax5.set_ylabel('Attack Magnitude')

    ax6.plot(result.times, result.states[36,:], '-')
    ax6.axhline(y=params["eta"], color='red', linestyle='--', label="eta")
    ax6.legend()
    ax6.set_ylabel('Anomaly Function (g_t)')

    ax7.plot(result.times, result.states[2,:], '-')
    #ax7.axhline(y=params["alpha"], color='red', linestyle='--', label="alpha")
    #ax7.legend()
    ax7.set_ylabel('State Estimation Error')

    ax8.plot(result.times, result.states[3,:], '-')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Residue')

    #plt.show()
    filename = os.path.join(data_path, 'uav_expec_{}.png'.format(ct.strftime('%Y-%m-%d_%HH%MM%SS')))
    plt.savefig(filename, format = 'png')
    plt.close()


    # Plot traces for single run

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, figsize=(10,14), tight_layout=True)
    header = 'Single Run'
    spec_str = 'Specification: '+phi
    rob_att_str = '\nRobustness={:.3e}, SR={}'.format(rob, int(100-result.states[4,0]))
    misc_params_str = 'eta={:.3f}'.format(params["eta"])
    title_str = header+'\n'+spec_str+'\n'+rob_att_str + '\n' + misc_params_str

    fig.suptitle(title_str)

    ax1.plot(result.times, result.states[43,:], '-', label='z')
    ax1.plot(result.times, result.states[55,:], 'r', label='z_hat')
    ax1.legend()
    ax1.set_ylabel('Z')

    ax2.plot(result.times, result.states[45,:], '-', label='x')
    ax2.plot(result.times, result.states[57,:], 'r', label='x_hat')
    ax2.legend()
    ax2.set_ylabel('X')

    ax3.plot(result.times, result.states[47,:], '-', label='y')
    ax3.plot(result.times, result.states[59,:], 'r', label='y_hat')
    ax3.legend()
    ax3.set_ylabel('Y')

    ax4.plot(result.times, result.states[43,:] - result.states[55,:], '-', label='z')
    ax4.plot(result.times, result.states[45,:] - result.states[57,:], 'r', label='x')
    ax4.plot(result.times, result.states[57,:] - result.states[59,:], 'g', label='y')
    ax4.legend()
    ax4.set_ylabel('Estimate Deviation')

    ax5.plot(result.times, result.states[11,:], '-')
    ax5.set_ylabel('Attack Magnitude')

    ax6.plot(result.times, result.states[61,:], '-')
    ax6.axhline(y=params["eta"], color='red', linestyle='--', label="eta")
    ax6.legend()
    ax6.set_ylabel('Anomaly Function (g_t)')

    ax7.plot(result.times, result.states[62,:], '-')
    # ax7.axhline(y=params["alpha"], color='red', linestyle='--', label="alpha")
    # ax7.legend()
    ax7.set_ylabel('State Estimation Error')

    ax8.plot(result.times, result.states[63,:], '-')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Residue')

    #plt.show()
    filename = os.path.join(data_path, 'uav_0th_{}.png'.format(ct.strftime('%Y-%m-%d_%HH%MM%SS')))
    plt.savefig(filename, format = 'png')
    plt.close()

    best_Trace = np.transpose(np.array(result.states))


    trace_cols = ['Time','Anom_alarm_rate','Error_Mean', 'Res_Mean', 'Num_error_systems', 
    'Attack_phi', 
    # 'Attack_phi_dot', 
    'Attack_theta', 
    # 'Attack_theta_dot',
    'Attack_psi', 
    # 'Attack_psi_dot', 
    'Attack_z', 'Attack_x', 'Attack_y', 'Attack_mag', 
    'E_phi','E_phi_dot','E_th','E_th_dot','E_psi','E_psi_dot',
    'E_z', 'E_z_dot','E_x', 'E_x_dot','E_y', 'E_y_dot',
    'E_phi_hat','E_phi_dot_hat','E_th_hat','E_th_dot_hat','E_psi_hat','E_psi_dot_hat',
    'E_z_hat', 'E_z_dot_hat','E_x_hat', 'E_x_dot_hat','E_y_hat', 'E_y_dot_hat', 
    'E_gt',
    '0_phi','0_phi_dot','0_th','0_th_dot','0_psi','0_psi_dot',
    '0_z', '0_z_dot','0_x', '0_x_dot','0_y', '0_y_dot',
    '0_phi_hat','0_phi_dot_hat','0_th_hat','0_th_dot_hat','0_psi_hat','0_psi_dot_hat',
    '0_z_hat', '0_z_dot_hat','0_x_hat', '0_x_dot_hat','0_y_hat', '0_y_dot_hat', 
    'Anomaly_gt','Error','Residue', 'Anomaly_sig', 'Error_sig']

    # print("Length of trace_cols is {}".format(len(trace_cols)))

    # print("trace_cols is")

    # print(trace_cols)

    best_result_DF = pd.DataFrame(best_Trace, columns = trace_cols)
    filename = os.path.join(data_path, 'uav_best_trace_{}.csv'.format(ct.strftime('%Y-%m-%d_%HH%MM%SS')))
    best_result_DF.to_csv(filename)

    # return best_result

# Write file with run parameters
sim_param_df = pd.DataFrame.from_dict(params, orient='index')
filename = os.path.join(data_path, 'uav_params_{}.csv'.format(ct.strftime('%Y-%m-%d_%HH%MM%SS')))
sim_param_df.to_csv(filename)


# Compute the Region Support and problem dimensionality
region_support = []

for k in range(params["num_signals"]):
    for rs in params["control_points{}".format(k)]:
        lb = rs[0]
        ub = rs[1]
        region_support.append([lb,ub])

region_support = np.array(region_support)

dimensionality = 0

for k in range(params["num_signals"]):
    dimensionality += len(params["control_points{}".format(k)])


#Build Test Function
rng = np.random.default_rng(12345)
test_function = Fn(generateRobustness)


# Compute the robustness of the uniform random samples
x_train = uniform_sampling(params["num_uniform_samples"], region_support, 
    dimensionality, rng)
y_train = compute_robustness(x_train, test_function)


# Bayesian Optimization: one sample at a time
gpr_model = InternalGPR()
bo = InternalBO()
x_train_1, y_train_1, x_new, y_new, bo_times, sim_times = bo.sample(test_function, params["num_bo_samples"], x_train, y_train, region_support, gpr_model, rng)

print("******************************************************")
print(x_train.shape)
print(y_train.shape)
print(x_new.shape)
print(y_new.shape)
print(x_train_1.shape)
print(y_train_1.shape)

print(y_new)
print(test_function.point_history)
print(test_function.count)



# Build data arrays for output
n_cp = dimensionality
data = np.zeros((params["num_uniform_samples"]+params["num_bo_samples"],n_cp+2))


# Find the sample with the best robustness and sabe the sample
best_rob = 9999999999
best_rob_idx = -2
best_sample = []
for ii in range(test_function.count):
    current_rob = test_function.point_history[ii][2]

    evalNo = test_function.point_history[ii][0]
    sampleArr = np.array(test_function.point_history[ii][1])
    sampleArr = np.concatenate([[evalNo],sampleArr,[current_rob]])

    data[ii,:] = sampleArr


    
    print("rob of {} is {}".format(ii, current_rob))
    if current_rob < best_rob:
        best_rob = current_rob
        best_rob_idx = test_function.point_history[ii][0]
        best_sample = test_function.point_history[ii][1]

cols = ["Iteration"]
for ii in range(n_cp):
    cols = cols +["cp_{}".format(ii+1)]
cols = cols + ['Robustness']
# print(cols)
data_df = pd.DataFrame(data, columns = cols)
filename = os.path.join(data_path, 'uav_data_{}.csv'.format(ct.strftime('%Y-%m-%d_%HH%MM%SS')))
# data_df.to_csv('data/uav_data_'+ct.strftime('%Y-%m-%d_%HH%MM%SS')+'.csv')
data_df.to_csv(filename)


# Plot Robustness histogram and robustness vs epoch
fig, (ax1, ax2) = plt.subplots(2, tight_layout=True)

ax1.hist(data[:,-1], bins = 20)

ax1.set_ylabel('Frequency')
ax1.set_xlabel('Robustness Value')

ax2.plot(data[:,-1], '-')
# ax2.axvline(x=params["num_uniform_samples"], ymin=np.min(data[:,-1]), ymax=np.max(data[:,-1]), color='red', linestyle='--', label="End of unfiorm sampling")
ax2.axvline(x=params["num_uniform_samples"], color='red', linestyle='--', label="End of unfiorm sampling")
ax2.legend()
ax2.set_ylabel('Robustness Value')
ax2.set_xlabel('Epoch')

filename = os.path.join(data_path, 'uav_rob_{}.png'.format(ct.strftime('%Y-%m-%d_%HH%MM%SS')))
plt.savefig(filename, format = 'png')
plt.close()


# Plot Sim and BO computation times
fig, (ax1, ax2) = plt.subplots(2, tight_layout=True)

title = "UAV sim time: {}".format(params["sim_end_time"])
title += ", steps: {}".format(params["sim_end_time"]/params["step"])
title += "\n Dimensionality: {}".format(dimensionality)
fig.suptitle(title)

ax1.plot(sim_times)

ax1.set_title(f"Average Sim Time: {np.mean(np.array(sim_times)):0.3f} secs")
ax1.set_ylabel('Simulation time (sec)')
ax1.set_xlabel('Epoch')

ax2.set_title(f"Average BO Time: {np.mean(np.array(bo_times)):0.3f} secs")
ax2.plot(bo_times, '-')
ax2.set_ylabel('BO calculation time (sec)')
ax2.set_xlabel('Epoch')

filename = os.path.join(data_path, 'uav_times_{}.png'.format(ct.strftime('%Y-%m-%d_%HH%MM%SS')))
plt.savefig(filename, format = 'png')
plt.close()


# Plot Trace of the best sampe

plotTrace_uav(best_sample)

# End
end = datetime.datetime.now()
elapsed_time = end - ct
print("Total elapsed time: {}".format(elapsed_time))
print("Best robustness is {}".format(best_rob))
print("Best robustness index is {}".format(best_rob_idx))
print("Best robustness sample is {}".format(best_sample))

print(data_df)

print("BO times are:")
print(bo_times)
print("Sim Times are:")
print(sim_times)

# Save results data frame
results = {
    "elapsed_time" : elapsed_time,
    "best_sample" : best_sample,
    "best_robustness" : best_rob,
    #"best_attack_integral" : att_integral,
    }
results_df = pd.DataFrame.from_dict(results, orient='index')
filename = os.path.join(data_path, 'uav_results_{}.csv'.format(ct.strftime('%Y-%m-%d_%HH%MM%SS')))
results_df.to_csv(filename)


