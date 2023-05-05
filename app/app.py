import streamlit as st
import pandas as pd


data=pd.read_pickle('../Notebook/skills_freq.pkl')



st.title('Enter the world of tech')


group_skills=data.groupby(data.columns[0]).apply(lambda x:x[x.columns[1]].tolist()).to_dict()

selected_skills={}


for group,skills in group_skills.items():

    st.write(f'**{group}')

    selected_skills[group]=st.multiselect(f'select skill for group {group}',skills)




# if __name__=='__main__':
