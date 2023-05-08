LOG_DATA_PKL="data.pkl"
LOG_MODEL_PKL="model.pkl"
LOG_METRICS_PKL="metrics.pkl"
#-------------------------------------
import os
import sklearn

import pickle
import yaml

import pandas as pd

import mlflow

from mlflow.tracking import MlflowClient

#----------------------------------------

class JobPrediction:

    # Constructor
    def __init__(self, mlflow_uri, run_id, clusters_yaml_path):

        # Constants
        # print(mlflow_uri)
        self.mlflow_uri  = mlflow_uri
        self.run_id      = run_id

        # Retrieve model and features
        mlflow_objs = self.load_mlflow_objs()
        self.model           = mlflow_objs[0]
        self.features_names  = mlflow_objs[1]
        self.targets_names   = mlflow_objs[2]
        
        # Load clusters config 
        self.path_clusters_config = clusters_yaml_path
        self.skills_clusters_df   = self.load_clusters_config(clusters_yaml_path)


    
    def load_mlflow_objs(self):

        mlflow.set_tracking_uri(self.mlflow_uri)

        client=MlflowClient()

        run=mlflow.get_run(self.run_id)

        artificats_path=run.info.artifact_uri

        data_path=os.path.join(artificats_path,LOG_DATA_PKL)
        with open(data_path,'rb') as handle:

            data_pkl=pickle.load(handle)

        
        model_path=os.path.join(artificats_path,LOG_MODEL_PKL)

        with open(model_path,'rb') as f:

            model_pkl=pickle.load(f)

        
        return model_pkl['model_object'],\
                data_pkl['features_names'],\
                data_pkl['targets_names']
    

    def load_clusters_config(self,path_clusters_config):


        with open(path_clusters_config,'r') as stream:

            clusters_config=yaml.safe_load(stream)

        
        clusters_df=[(cluster_name,cluster_skill)
                     for cluster_name, cluster_skills in clusters_config.items()
                     for cluster_skill in cluster_skills]
        
        clusters_df=pd.DataFrame(clusters_df,columns=['cluster_name','skill'])


        return clusters_df
    


    def get_skills(self):

        return self.features_names
    
    def get_jobs(self):

        return self.targets_names
    


    def create_features_array(self,available_skills):

        def create_cluster_features(self,available_skills):

            sample_clusters=self.skills_clusters_df.copy()

            sample_clusters['available_skills']=sample_clusters['skill'].isin(available_skills)

            cluster_features=sample_clusters.groupby("cluster_name")["available_skills"].sum()

            return cluster_features
        

        def create_skills_features(self,available_skills):

            sample_clusters=self.skills_clusters_df.copy()
            sample_clusters['available_skills']=sample_clusters['skill'].isin(available_skills)

            skill_feature=sample_clusters.groupby('skill')['available_skills'].sum()

            

            return skill_feature
        

        clusters_features=create_cluster_features(self,available_skills)

        skills_features=create_skills_features(self,available_skills)


        features=pd.concat([skills_features,clusters_features],axis=0)

        features=features[self.features_names]

        return features.values
    
    

    def predict_jobs_probabilities(self,available_skills):

        features_array=self.create_features_array(available_skills)


        predictions=self.model.predict_proba([features_array])
        predictions=[prob[0][1] for prob in predictions]

        predictions=pd.Series(predictions,index=self.targets_names)

        return predictions
    

    def recommend_new_skills(self,available_skills,target_job,threshold=0):


        base_predictions=self.predict_jobs_probabilities(available_skills)

        all_skills=pd.Series(self.get_skills())

        new_skills=all_skills[~all_skills.isin(available_skills)].copy()


        simulated_results=[]

        for skill in new_skills:

            add_skill_prob=self.predict_jobs_probabilities([skill]+available_skills)

            add_skill_uplift=(add_skill_prob-base_predictions)/base_predictions

            add_skill_uplift.name=skill
            simulated_results.append(add_skill_uplift)

        simulated_results=pd.DataFrame(simulated_results)


        target_resutls=simulated_results[target_job].sort_values(ascending=False)

        return target_resutls[(target_resutls>threshold)]
            




