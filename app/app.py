import streamlit as st
import pandas as pd
import sys
sys.path.append('../Scripts')
from JobPrediction import JobPrediction
import plotly
import plotly.express as px



MLFLOW_TRACKING_URI='../models/mlruns'
MLFLOW_EXPERIMENT_NAME="skills_jobs_stackoverflow"
MLFLOW_RUN_ID="cb06262089ad4c5c82d2da3046219870"
CLUSTERS_YAML_PATH="../Notebook/features_skills_clusters_description.yaml"

data=pd.read_pickle('../Notebook/skills_freq.pkl')



st.title('Enter the world of tech')


group_skills=data.groupby(data.columns[0]).apply(lambda x:x[x.columns[1]].tolist()).to_dict()


group_skills={
    "Database you have worked with":group_skills["DatabaseHaveWorkedWith"],
    "Language you have worked with":group_skills["LanguageHaveWorkedWith"],
    "Language you want to work with:":group_skills["LanguageWantToWorkWith"],
    "other Technology that you have worked with":group_skills["MiscTechHaveWorkedWith"],
    "Collaboration tools that you have worked with":group_skills["NEWCollabToolsHaveWorkedWith"],
    "Plateform that you have worked with":group_skills["PlatformHaveWorkedWith"],
    "Tools that you have worked with":group_skills["ToolsTechHaveWorkedWith"],
    "Web framework that you have worked with":group_skills["WebframeHaveWorkedWith"]
}
selected_skills={}

container_style = """
    .stContainer {
        background-color: blue;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 1px 1px 10px grey;
    }
"""

for group,skills in group_skills.items():

    # st.write(f'**{group}')

    with st.markdown(f'<div style="{container_style}">', unsafe_allow_html=True):



        st.header(group)
        
        selected_skills[group]=st.multiselect(f'select skill for group {group}',skills)



container=st.container()

button_col1,button_col2=container.columns(2)

with button_col1:
    st.write('')

with button_col2:

    button=st.button("What is my match?",key='find my match button',help="click to find your match")

current_skills = [skill for group, skills in selected_skills.items() if len(skills) != 0 for skill in skills]


st.markdown(
    """<style>
        .stButton button {
            background-color: #1E90FF;
            color: white;
            padding: 0.7rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 1.2rem;
            display: block;
            margin: 0 auto;
        }
    </style>""",
    unsafe_allow_html=True,
)

if button :    

    if len(current_skills)>0:
        model = JobPrediction(MLFLOW_TRACKING_URI, MLFLOW_RUN_ID, CLUSTERS_YAML_PATH)

        base_predictions=model.predict_jobs_probabilities(current_skills)
        base_predictions = base_predictions.sort_values(ascending=True)

        fig = px.bar(x=base_predictions.values, y=base_predictions.index, orientation='h')
        fig.update_layout(title='Top Job Recommendations Based on Your Skills', xaxis_title='Probability of Job Match', yaxis_title='Job Titles',height=500, width=800)

        st.plotly_chart(fig)

        

    else:

        st.write('You have to select a skills !')












# if __name__=='__main__':
