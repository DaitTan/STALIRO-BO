import time
import threading

from kubernetes import client

def create_job_object(job_name, completions, parallelism, queue_name, container_image):
    # Configureate Pod template container
    container = client.V1Container(
        name="uav-v3",
        image=container_image,
        env=[
            client.V1EnvVar(
                name='BROKER_URL',
                value='amqp://guest:guest@rabbitmq-service:5672'
            ),
            client.V1EnvVar(
                name='QUEUE',
                value=queue_name
            )
        ])
    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(name=job_name),
        spec=client.V1PodSpec(restart_policy="Never", containers=[container]))
    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        completions=completions,
        parallelism=parallelism,
        backoff_limit=1)
    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=spec)

    return job

def create_job(api_instance, job, job_name):
    api_response = api_instance.create_namespaced_job(
        body=job,
        namespace="default")
    print("Job created. status='%s'" % str(api_response.status))
    # x = threading.Thread(target=get_job_status, args=(api_instance, job_name,))
    # x.start()


def get_job_status(api_instance, job_name):
    job_completed = False
    while not job_completed:
        api_response = api_instance.read_namespaced_job_status(
            name=job_name,
            namespace="default")
        if api_response.status.succeeded is not None or \
                api_response.status.failed is not None:
            job_completed = True
        time.sleep(1)
        print("Job status='%s'" % str(api_response.status))
