import json
import shutil
from pathlib import Path

REGISTRY_PATH = "model_registry/production_model.json"
ARTIFACT_PATH = "artifacts/metrics.json"
DEPLOYED_PATH = "deployment/deployed_model"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def deploy_model(new_metrics):
    Path(DEPLOYED_PATH).mkdir(parents=True, exist_ok=True)
    shutil.copytree("faiss_db", DEPLOYED_PATH, dirs_exist_ok=True)
    save_json(REGISTRY_PATH, new_metrics)

def conditional_deploy():
    prod_metrics = load_json(REGISTRY_PATH)
    new_metrics = load_json(ARTIFACT_PATH)

    if new_metrics["answer_relevance_score"] >= prod_metrics["answer_relevance_score"]:
        deploy_model(new_metrics)
        return True, "New model deployed successfully."
    else:
        return False, "New model did not outperform production. Deployment aborted."
