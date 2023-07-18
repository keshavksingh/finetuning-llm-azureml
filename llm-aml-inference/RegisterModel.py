from azure.ai.ml import MLClient
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


from azure.ai.ml.entities import Model
from azure.ai.ml.constants import ModelType

RunId="122ceb53-c7d6-4069-9488-166595d49895"

run_model = Model(
    path="runs:/"+RunId+"/outputs/",
    name="llmossmodel",
    version="1",
    description="Model Registered & Created from run.",
    type="custom_model" #[custom_model, mlflow_model, triton_model]
)

ml_client.models.create_or_update(run_model)