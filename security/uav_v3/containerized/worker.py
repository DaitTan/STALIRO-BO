#!/usr/bin/env python3

import sys
import json
import numpy as np
import os

import uav_v3_model_seeded

from pymongo import MongoClient

mongo_client = MongoClient('en4202145l.cidse.dhcp.asu.edu', 27017)
database = mongo_client.simulations
collection = database.results

if __name__ == "__main__":
    simulation_id = sys.stdin.readlines()[0]
    print("Processing {}".format(simulation_id))
    simulation_documents = collection.find_one({
        'id': simulation_id
    })
    print("Got {}".format(simulation_documents))
    parameters = simulation_documents['parameters']
    
    
    print("Processing {}".format(parameters['id']))
    static = parameters['static']
    times = np.array(parameters['times'])
    params = parameters['params']
    signals = np.array(parameters['signals'])
    
    pod_hostname = os.environ['HOSTNAME']
    
    sims = 100

    if len(static) > 0:
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

    trajectories = result.T.tolist()
    timestamps = result[:,0].tolist()
    
    collection.update_one({'id': parameters['id']}, {'$set': {'trajectories': trajectories, 'timestamps': timestamps, 'status': 'done', 'pod_hostname': pod_hostname}})
    
    print("Done with {}".format(parameters['id']))
