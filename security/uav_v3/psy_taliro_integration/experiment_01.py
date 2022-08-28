import math

from experiment_runner import ExperimentRunner

all_experiment_params = [
    {   
        "benchmark_name": "uav-experiment-01-run-01-alpha-0.05-epsilon-0.01-bo-sigmoid-sim-end-45.0",
        "step" : 0.01, # Time step
        "sim_end_time" : 45.0,
        "eta" : 0.3019215166568756,
        "epsilon" : 0.01, # Epsilon from alpha-epsion attack, Anomaly rate corresponding to eta
        "alpha" : 0.05, # Alpha from alpha-epislon definition: desired threshold for the system error to cross
        "delay" : 10.0,
        "num_signals" : 6,
        "control_points": [
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04),(0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi), (-0.04, 0.04),(0.0, 200.0),(0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)]
        ],
        "initial_conditions" : [],
        "runs" : 1,
        "num_uniform_samples" : 100,
        "num_bo_samples" : 100,
        "total_budget": 200
    },
    {
        "benchmark_name": "uav-experiment-01-run-02-alpha-0.025-epsilon-0.01-bo-sigmoid-sim-end-45.0",
        "step" : 0.01, # Time step
        "sim_end_time" : 45.0,
        "eta" : 0.3019215166568756,
        "epsilon" : 0.01, # Epsilon from alpha-epsion attack, Anomaly rate corresponding to eta
        "alpha" : 0.025, # Alpha from alpha-epislon definition: desired threshold for the system error to cross
        "delay" : 10.0,
        "num_signals" : 6,
        "control_points": [
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04),(0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi), (-0.04, 0.04),(0.0, 200.0),(0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)]
        ],
        "initial_conditions" : [],
        "runs" : 1,
        "num_uniform_samples" : 100,
        "num_bo_samples" : 100,
        "total_budget": 200
    },
    {
        "benchmark_name": "uav-experiment-01-run-03-alpha-0.015-epsilon-0.01-bo-sigmoid-sim-end-45.0",
        "step" : 0.01, # Time step
        "sim_end_time" : 45.0,
        "eta" : 0.3019215166568756,
        "epsilon" : 0.01, # Epsilon from alpha-epsion attack, Anomaly rate corresponding to eta
        "alpha" : 0.015, # Alpha from alpha-epislon definition: desired threshold for the system error to cross
        "delay" : 10.0,
        "num_signals" : 6,
        "control_points": [
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04),(0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi), (-0.04, 0.04),(0.0, 200.0),(0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)]
        ],
        "initial_conditions" : [],
        "runs" : 1,
        "num_uniform_samples" : 100,
        "num_bo_samples" : 100,
        "total_budget": 200
    },
    {
        "benchmark_name": "uav-experiment-01-run-04-alpha-0.015-epsilon-0.01-bo-sigmoid-sim-end-25.0",
        "step" : 0.01, # Time step
        "sim_end_time" : 25.0,
        "eta" : 0.3019215166568756,
        "epsilon" : 0.01, # Epsilon from alpha-epsion attack, Anomaly rate corresponding to eta
        "alpha" : 0.015, # Alpha from alpha-epislon definition: desired threshold for the system error to cross
        "delay" : 10.0,
        "num_signals" : 6,
        "control_points": [
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04),(0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi), (-0.04, 0.04),(0.0, 200.0),(0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)]
        ],
        "initial_conditions" : [],
        "runs" : 1,
        "num_uniform_samples" : 100,
        "num_bo_samples" : 100,
        "total_budget": 200
    },
    {
        "benchmark_name": "uav-experiment-01-run-05-alpha-0.025-epsilon-0.01-bo-sigmoid-sim-end-25.0",
        "step" : 0.01, # Time step
        "sim_end_time" : 25.0,
        "eta" : 0.3019215166568756,
        "epsilon" : 0.01, # Epsilon from alpha-epsion attack, Anomaly rate corresponding to eta
        "alpha" : 0.025, # Alpha from alpha-epislon definition: desired threshold for the system error to cross
        "delay" : 10.0,
        "num_signals" : 6,
        "control_points": [
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04),(0.0, 200.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi), (-0.04, 0.04),(0.0, 200.0),(0.0, math.pi)],
            [(-0.02, 0.02), (-0.04, 0.04), (0.0, 20.0), (0.0, math.pi)]
        ],
        "initial_conditions" : [],
        "runs" : 1,
        "num_uniform_samples" : 100,
        "num_bo_samples" : 100,
        "total_budget": 200
    }
]

runner = ExperimentRunner(
    all_params=all_experiment_params,
    system_config={
        'mongo_host': '10.218.101.19',
        'rabbitmq_host': 'amqp://guest:guest@10.107.254.193:5672'
    },
    experiment_name='uav-v3-bo-sigmoid-experiment-01'
)

all_results = runner.run_all()