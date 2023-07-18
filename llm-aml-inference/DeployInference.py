from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings
)
from azure.identity import ClientSecretCredential

#Details of AzureML workspace
subscription_id = '<>'
resource_group = '<>'
workspace_name = '<>'

tenant_id='<>'
client_id='<>'
client_secret='<>'

creds = ClientSecretCredential(tenant_id, client_id, client_secret)
ml_client = MLClient(credential=creds, subscription_id=subscription_id, resource_group_name=resource_group, workspace_name=workspace_name)

endpoint_name = "llm-inference-endpoint"

endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="This is endpoint for LLM Inferencing.",
    auth_mode="key"
)

model = ml_client.models.get(name="llmossmodel",version="1")
llm_env = ml_client.environments.get(name='llminferencingcustomenvironment', version="1")

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=llm_env,
    scoring_script="score.py",
    code_path="C:/pythonprojects/aml/llm-aml-inference",
    instance_type="Standard_NC12s_v3",
    instance_count=1,
    request_settings=OnlineRequestSettings(max_concurrent_requests_per_instance=3,
                                           request_timeout_ms= 90000,
                                           max_queue_wait_ms=60000)
)

endpoint.traffic = {"blue": 100}
endpoint_poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
if endpoint_poller.result():
    print("Endpoint Creation Complete!")
    print(endpoint_poller.result())
    deployment_poller = ml_client.online_deployments.begin_create_or_update(deployment=blue_deployment)
    if deployment_poller.result():
        print("Deployment of Endpoint Complete!")
        print(deployment_poller.result())
