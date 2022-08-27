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
        params.socket_timeout = 10
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
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
        
        parameters_json_string = json.dumps(parameters)
        
        self.channel.basic_publish(exchange='', routing_key=self.queue_name, body=parameters_json_string)
        print('Created job with id {}'.format(unique_id))
        return unique_id
    
    def get_job_result(self, unique_id):
        print('Getting result for job with id {}'.format(unique_id))
        
        result = self.collection.find_one({'id': unique_id})
        
        while result is None:
            result = self.collection.find_one({'id': unique_id})
            time.sleep(1)
        
        print('Got result for job with id {}'.format(unique_id))
        return result['trajectories'], result['timestamps']
        