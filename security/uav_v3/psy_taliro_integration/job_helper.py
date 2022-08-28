import json
import time
import uuid
import logging

from pymongo import MongoClient
import pika

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class JobHelper:
    def __init__(self, mongo_host, rabbitmq_host, queue_name):
        self.mongo_host = mongo_host
        self.rabitmq_host = rabbitmq_host
        self.queue_name = queue_name
        self.__connect_to_database()
        self.__connect_to_queue()
        
    def __connect_to_database(self):
        print('Connecting to MongoDB at {}'.format(self.mongo_host))
        self.mongo_client = MongoClient(self.mongo_host, 27017)
        self.database = self.mongo_client.simulations
        self.collection = self.database.results
        print('Connected to MongoDB')
        
    def __connect_to_queue(self):
        print('Connecting to RabbitMQ at {}'.format(self.rabitmq_host))
        params = pika.URLParameters(self.rabitmq_host)
        params.socket_timeout = 120
        params.heartbeat = 0
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        self.channel.queue_delete(queue=self.queue_name)
        self.channel.queue_declare(queue=self.queue_name)
        print('Connected to RabbitMQ')
        
    def create_job(self, static, times, signals, params):
        unique_id = str(uuid.uuid4())
        print('Creating job with id {}'.format(unique_id))
        parameters = {
            'id': unique_id,
            'static': static,
            'times': times.tolist(),
            'signals': signals.tolist(),
            'params': params
        }
        
        self.collection.insert_one({
            'id': unique_id,
            'parameters': parameters,
            'status': 'created',
        })
        
        self.channel.basic_publish(exchange='', routing_key=self.queue_name, body=unique_id)
        print('Created job with id {}'.format(unique_id))
        return unique_id
    
    def get_job_result(self, unique_id):
        print('Getting result for job with id {}'.format(unique_id))
        
        result = self.collection.find_one({'id': unique_id})
        
        while result is None:
            result = self.collection.find_one({'id': unique_id}, {'id': 1, 'status': 1})
            time.sleep(1)
        
        status = result['status']
        while status != 'done':
            status = self.collection.find_one({'id': unique_id}, {'id': 1, 'status': 1})['status']
            time.sleep(1)
            
        result = self.collection.find_one({'id': unique_id}, {'trajectories': 1, 'timestamps': 1, 'id': 1, 'status': 1, 'pod_hostname': 1})
        print('Got result for job with id {}, completed on {}'.format(unique_id, result['pod_hostname']))
        return result['trajectories'], result['timestamps']
        