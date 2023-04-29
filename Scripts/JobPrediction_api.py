
MLFLOW_TRACKING_URI='../models/mlruns'
MLFLOW_EXPERIMENT_NAME="skills_jobs_stackoverflow"
MLFLOW_RUN_ID="cb06262089ad4c5c82d2da3046219870"
CLUSTERS_YAML_PATH="../Talent_tracking/Notebook/features_skills_clusters_description.yaml"


from JobPrediction import JobPrediction
from flask import Flask, request, jsonify
import sys


app=Flask(__name__)

job_model=JobPrediction(mlflow_uri=MLFLOW_TRACKING_URI,
                        run_id=MLFLOW_RUN_ID,
                        clusters_yaml_path=CLUSTERS_YAML_PATH)


@app.route('/predict_jobs_probs',methods=['POST'])

def predict_job_probs():

    available_skills=request.get_json()

    predictions=job_model.predict_jobs_probabilities(available_skills).to_dict()

    return jsonify(predictions)



@app.route('/recommend_new_skills',methods=['POST'])

def recommand_skill():

    request_details=request.get_json()

    available_skills=request_details['available_skills']

    target_job=request_details['target_job']

    return jsonify(job_model.recommend_new_skills(available_skills,target_job).to_dict())

if __name__=='__main__':

    app.run(port=5000)

