import math
import pathlib

import numpy as np
from bo import BO
from bo.bayesianOptimization import InternalBO
from bo.gprInterface import InternalGPR
from kubernetes import client, config
from numpy.typing import NDArray
from staliro.models import (ModelData, SignalTimes, SignalValues, StaticInput,
                            blackbox)
from staliro.options import Options, SignalOptions
from staliro.specifications import RTAMTDense
from staliro.staliro import staliro
from staliro.signals import delayed, pchip, harmonic

from job_helper import JobHelper
from kubernetes_utils import create_job_object, create_job, get_job_status

benchmark_name = 'uav-v3-security-exp01'
results_directory = pathlib.Path().cwd().joinpath('results', benchmark_name)
results_directory.mkdir(parents=True, exist_ok=True)

params = {
    "step" : 0.01, # Time step
    "sim_end_time" : 45.0,
    "eta" : 0.3019215166568756,
    "epsilon" : 0.01, # Epsilon from alpha-epsion attack, Anomaly rate corresponding to eta
    "alpha" : 0.05, # Alpha from alpha-epislon definition: desired threshold for the system error to cross
    "delay" : 10.0,
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
    "initial_conditions" : [],
    "runs" : 1,
    "num_uniform_samples" : 100,
    "num_bo_samples" : 100,
}

# Setup the experiments
_job_helper = JobHelper(
    mongo_host='10.218.101.19',
    rabbitmq_host='amqp://guest:guest@10.107.254.193:5672',
    queue_name='uav_v3_jobs'
)

config.load_kube_config()
kubernetes_client = client.BatchV1Api()

kubernetes_job_object = create_job_object(
    job_name=benchmark_name,
    completions=params['runs'] * (params['num_uniform_samples'] + params['num_bo_samples']),
    parallelism=10,
    queue_name='uav_v3_jobs',
    container_image='aniruddhchandratre/uav_v3:latest'
)

create_job(kubernetes_client, kubernetes_job_object, benchmark_name)


@blackbox(sampling_interval = params["step"])
def uav_model(static: StaticInput, times: SignalTimes, signals: SignalValues) -> ModelData[NDArray[np.float_], None]:
    print("Creating simulation")
    job_id = _job_helper.create_job(static, times, signals, params)
    trajectories, timestamps = _job_helper.get_job_result(job_id)
    
    trajectories = np.array(trajectories)
    timestamps = np.array(timestamps)
    
    return ModelData(trajectories, timestamps)


phi = "not ( (always[0, {}] (anomaly_sig <= 0.5)) and ( eventually[0, {}] (error_sig >= 0.5) ) )".format(
    params["sim_end_time"],params["sim_end_time"])

specification = RTAMTDense(phi, {"anomaly_sig": 64, "error_sig": 65})

optimizer = BO(
            benchmark_name=f"{benchmark_name}_budget_{params['num_bo_samples'] + params['num_uniform_samples']}_{params['runs']}_reps",
            init_budget = params['num_uniform_samples'],
            gpr_model = InternalGPR(),
            bo_model = InternalBO(),
            folder_name = results_directory,
            init_sampling_type = "lhs_sampling"
        )

options = Options(
        runs=params['runs'],
        iterations=params['num_bo_samples'] + params['num_uniform_samples'], 
        interval=(0, params["sim_end_time"]), 
        seed = 12346, 
        signals=[
            SignalOptions(control_points=params["control_points0"], factory=delayed(signal_factory = harmonic, delay=params["delay"])),
            SignalOptions(control_points=params["control_points1"], factory=delayed(signal_factory = harmonic, delay=params["delay"])),
            SignalOptions(control_points=params["control_points2"], factory=delayed(signal_factory = harmonic, delay=params["delay"])),
            SignalOptions(control_points=params["control_points3"], factory=delayed(signal_factory = harmonic, delay=params["delay"])),
            SignalOptions(control_points=params["control_points4"], factory=delayed(signal_factory = harmonic, delay=params["delay"])),
            SignalOptions(control_points=params["control_points5"], factory=delayed(signal_factory = harmonic, delay=params["delay"])),
        ],      
        static_parameters=params["initial_conditions"]
    )

result = staliro(uav_model, specification, optimizer, options)
