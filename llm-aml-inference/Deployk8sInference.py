from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    OnlineRequestSettings,
    ResourceRequirementsSettings,
    ResourceSettings,
    ProbeSettings
)
from azure.identity import ClientSecretCredential

#Details of AzureML workspace
subscription_id = "<>"
resource_group = "<>"
workspace_name = "<>"

tenant_id="<>"
client_id="<>"
client_secret="<>"

creds = ClientSecretCredential(tenant_id, client_id, client_secret)
ml_client = MLClient(credential=creds, subscription_id=subscription_id, resource_group_name=resource_group, workspace_name=workspace_name)

endpoint_name = "llm-aks-gpu-inference-endpoint"

endpoint = KubernetesOnlineEndpoint(
    name = endpoint_name,
    compute = "aks-gpu-aml",
    description="This is endpoint for LLM Inferencing.",
    auth_mode="key"
)

model = ml_client.models.get(name="llmossmodel",version="1")
llm_env = ml_client.environments.get(name='llminferencingcustomenvironment', version="1")

blue_deployment = KubernetesOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=llm_env,
    scoring_script="score.py",
    code_path="C:/pythonprojects/aml/llm-aml-inference",
    instance_count=1,
    request_settings=OnlineRequestSettings(max_concurrent_requests_per_instance=3,
                                           request_timeout_ms= 90000,
                                           max_queue_wait_ms=60000),
    liveness_probe=ProbeSettings(period=1200,initial_delay=20,timeout=2500,success_threshold=1,failure_threshold=3),
    readiness_probe=ProbeSettings(period=1200,initial_delay=20,timeout=2500,success_threshold=1,failure_threshold=3)
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

"""
az aks get-credentials --resource-group <> --name ***aksgpuws****
kubectl create namespace gpu-resources
kubectl apply -f nvidia-device-plugin-ds.yaml
az k8s-extension create --cluster-type managedClusters --cluster-name ***aksgpuws**** --resource-group <> --name dapr --extension-type Microsoft.Dapr
az k8s-extension create --name amlextension --extension-type Microsoft.AzureML.Kubernetes --config enableInference=True enableTraining=True inferenceRouterServiceType=LoadBalancer allowInsecureConnections=True inferenceLoadBalancerHS=False --cluster-type managedClusters --cluster-name ***aksgpuws**** --resource-group <> --scope cluster
azure.core.exceptions.HttpResponseError: (DeploymentFailed) InferencingClient HttpRequest error, error detail: {"errors":{"InferenceContainerConfiguration.ServingContainer.LivenessProbe.PeriodSeconds":["Invalid value provided for Period Seconds for Probe: <1200>. The value should be less than 300. To learn more, please see https://learn.microsoft.com/azure/machine-learning/reference-yaml-deployment-managed-online#probesettings."],"InferenceContainerConfiguration.ServingContainer.LivenessProbe.TimeoutSeconds":["Invalid value provided for Timeout Seconds for Probe: <2500>. The value should be less than 300. To learn more, please see https://learn.microsoft.com/azure/machine-learning/reference-yaml-deployment-managed-online#probesettings."],"InferenceContainerConfiguration.ServingContainer.ReadinessProbe.PeriodSeconds":["Invalid value provided for Period Seconds for Probe: <1200>. The value should be less than 300. To learn more, please see https://learn.microsoft.com/azure/machine-learning/reference-yaml-deployment-managed-online#probesettings."],"InferenceContainerConfiguration.ServingContainer.ReadinessProbe.TimeoutSeconds":["Invalid value provided for Timeout Seconds for Probe: <2500>. The value should be less than 300. To learn more, please see https://learn.microsoft.com/azure/machine-learning/reference-yaml-deployment-managed-online#probesettings."]},"type":"https://tools.ietf.org/html/rfc7231#section-6.5.1","title":"One or more validation errors occurred.","status":400,"traceId":""}
Code: DeploymentFailed
Message: InferencingClient HttpRequest error, error detail: {"errors":{"InferenceContainerConfiguration.ServingContainer.LivenessProbe.PeriodSeconds":["Invalid value provided for Period Seconds for Probe: <1200>. The value should be less than 300. To learn more, please see https://learn.microsoft.com/azure/machine-learning/reference-yaml-deployment-managed-online#probesettings."],"InferenceContainerConfiguration.ServingContainer.LivenessProbe.TimeoutSeconds":["Invalid value provided for Timeout Seconds for Probe: <2500>. The value should be less than 300. To learn more, please see https://learn.microsoft.com/azure/machine-learning/reference-yaml-deployment-managed-online#probesettings."],"InferenceContainerConfiguration.ServingContainer.ReadinessProbe.PeriodSeconds":["Invalid value provided for Period Seconds for Probe: <1200>. The value should be less than 300. To learn more, please see https://learn.microsoft.com/azure/machine-learning/reference-yaml-deployment-managed-online#probesettings."],"InferenceContainerConfiguration.ServingContainer.ReadinessProbe.TimeoutSeconds":["Invalid value provided for Timeout Seconds for Probe: <2500>. The value should be less than 300. To learn more, please see https://learn.microsoft.com/azure/machine-learning/reference-yaml-deployment-managed-online#probesettings."]},"type":"https://tools.ietf.org/html/rfc7231#section-6.5.1","title":"One or more validation errors occurred.","status":400,"traceId":""}"""
#https://github.com/Azure/azureml-examples/blob/main/sdk/python/endpoints/online/kubernetes/kubernetes-online-endpoints-safe-rollout.ipynb
#https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-kubernetes-online?view=azureml-api-2#containerresourcerequests
#https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-kubernetes-online?view=azureml-api-2#probesettings
"""