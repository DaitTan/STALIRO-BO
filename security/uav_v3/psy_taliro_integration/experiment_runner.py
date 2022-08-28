import threading
import pathlib
from typing import Dict, List

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

class ExperimentRunner:
    def __init__(self, all_params: List[Dict], system_config: Dict, experiment_name: str):
        self.all_params = all_params
        self.system_config = system_config
        self.experiment_name = experiment_name
        
        config.load_kube_config()
        self.kubernetes_client = client.BatchV1Api()
        self.__setup_kubernetes_job()
        pass
    
    def __get_total_experiment_count(self):
        experiment_count = 0
        for experiment_params in self.all_params:
            experiment_count += experiment_params['runs'] * (experiment_params['num_uniform_samples'] + experiment_params['num_bo_samples'])
        return experiment_count
        
    def __setup_kubernetes_job(self):
        kubernetes_job_object = create_job_object(
            job_name=self.experiment_name,
            completions=self.__get_total_experiment_count(),
            parallelism=len(self.all_params),
            queue_name=self.experiment_name,
            container_image='aniruddhchandratre/uav_v3:latest'
        )
        create_job(self.kubernetes_client, kubernetes_job_object, self.experiment_name)
        
    
    def run_single(self, params):
        _job_helper = JobHelper(
            mongo_host=self.system_config['mongo_host'],
            rabbitmq_host=self.system_config['rabbitmq_host'],
            queue_name=self.experiment_name
        )
        
        results_directory = pathlib.Path().cwd().joinpath('results', params['benchmark_name'])
        results_directory.mkdir(parents=True, exist_ok=True)
        
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
            benchmark_name=f"{params['benchmark_name']}_budget_{params['num_bo_samples'] + params['num_uniform_samples']}_{params['runs']}_reps",
            init_budget = params['num_uniform_samples'],
            gpr_model = InternalGPR(),
            bo_model = InternalBO(),
            folder_name = results_directory,
            init_sampling_type = "lhs_sampling"
        )
        
        signals = []
        for control_point in params['control_points']:
            print(control_point)
            signals.append(
                SignalOptions(control_points=control_point, factory=delayed(signal_factory = harmonic, delay=params["delay"]))
            )
        
        options = Options(
            runs=params['runs'],
            iterations=params['num_bo_samples'] + params['num_uniform_samples'], 
            interval=(0, params["sim_end_time"]), 
            seed = 12346, 
            signals=signals,      
            static_parameters=params["initial_conditions"]
        )
        
        result = staliro(uav_model, specification, optimizer, options)
        return result
    
    def run_all(self):
        for params in self.all_params:
            thread = threading.Thread(target=self.run_single, args=(params,))
            thread.start()