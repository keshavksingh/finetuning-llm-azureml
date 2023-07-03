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
workspace_name = "<>"

tenant_id="<>"
client_id="<>"
client_secret="<>"

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

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

conda = CondaDependencies()
conda.add_channel('default')
conda.add_channel('conda-forge')
conda.add_channel('pytorch')
conda.add_channel('nvidia')

conda.add_conda_package('python=3.8.2')
conda.add_conda_package('pip=20.2.4')
conda.add_conda_package('pytorch=2.0.1')
conda.add_conda_package('torchvision=0.15.2')
conda.add_conda_package('torchaudio=2.0.2')
conda.add_conda_package('pytorch-cuda=11.7')

conda.add_pip_package('bitsandbytes==0.39.1')
conda.add_pip_package('git+https://github.com/huggingface/transformers.git')
conda.add_pip_package('git+https://github.com/huggingface/peft.git')
conda.add_pip_package('git+https://github.com/huggingface/accelerate.git')
conda.add_pip_package('datasets')
conda.add_pip_package('scipy')
conda.add_pip_package('pyarrow')
conda.add_pip_package('loralib')
conda.add_pip_package('sentencepiece')

myenv = Environment(name="llmdollyenv")
run_config = RunConfiguration(conda_dependencies=conda)
run_config.environment.docker.enabled = True

exp = Experiment(workspace=ws, name="AML-FINETUNE-DOLLY-LLM-V1")
step_1 = PythonScriptStep(source_directory='C:pythonprojects/aml/llm-dolly',
                          script_name="Train.py",
                          compute_target=compute_name,
                          runconfig=run_config,
                          allow_reuse=False)

pipeline = Pipeline(workspace=ws, steps=[step_1])
pipeline_run = pipeline.submit("AML-FINETUNE-DOLLY-LLM-V1", regenerate_outputs=True)