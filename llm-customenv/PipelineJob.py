from azure.identity import ClientSecretCredential
from azureml.core import Workspace
from azureml.core.experiment import Experiment
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.compute import ComputeInstance
from azureml.core import Experiment,Environment
#from azureml.core.authentication import ClientSecretCredential
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep, SynapseSparkStep

#Enter details of your AzureML workspace
subscription_id = '<>'
resource_group = '<>'
workspace_name = '<>'

tenant_id='<>'
client_id='<>'
client_secret='<>'

# Create the credentials object
#creds = ClientSecretCredential(tenant_id, client_id, client_secret)
creds = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=client_id,
    service_principal_password=client_secret
)

# Load the workspace from the subscription, resource group, and workspace name
ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name, auth=creds)

# Get the compute instance
compute_name = "gpuvmcomputev100"
try:
    compute_instance = ComputeInstance(workspace=ws, name=compute_name)
    print(f"Found existing compute instance: {compute_name}")
except ComputeTargetException:
    print(f"Not Found compute instance: {compute_name}")

from azureml.core import RunConfiguration
from azureml.core import ScriptRunConfig 
from azureml.core import Experiment

"""
llm_env = Environment.get(ws, 'llmfinetuneoss', version=25)
#llm_env = Environment(ws,"llm_env")
llm_env.docker.base_image = None
llm_env.python.user_managed_dependencies = True

exp = Experiment(workspace=ws, name='AML-FINETUNE-LLM-DOCKER-V1')
src = ScriptRunConfig(source_directory='C:pythonprojects/aml/llm-customenv',
                      script='Train.py',
                      compute_target=compute_name,
                      environment=llm_env)
run = Experiment(ws,'AML-FINETUNE-LLM-DOCKER-V1').submit(src)
"""

llm_env = Environment.get(ws, 'llmtrainingcustomenvironment', version=1)
llm_env.docker.base_image = None
llm_env.python.user_managed_dependencies = True
runconfig = RunConfiguration()
runconfig.environment = llm_env

exp = Experiment(workspace=ws, name="CUSTOMENV-FINETUNE-ELEUTHERAI_GPT_NEOX_20B-LLM-V1")

step_1 = PythonScriptStep( source_directory='C:pythonprojects/aml/llm-customenv',
                          script_name='Train.py',
                          compute_target=compute_name,
                          runconfig=runconfig,
                          allow_reuse=False)

pipeline = Pipeline(workspace=ws, steps=[step_1])
pipeline_run = pipeline.submit("CUSTOMENV-FINETUNE-ELEUTHERAI_GPT_NEOX_20B-LLM-V1", regenerate_outputs=True)