import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
from src.model import load_models
from src.features import model_features
from src.recommender import generate_recommendations

st.set_page_config(page_title='Personalised Healthcare Recommendation System', page_icon='🩺', layout='wide')

st.title('🩺 Personalised Healthcare Recommendation System')
st.caption('Synthetic-dataset research prototype • ML risk stratification + explainable wellness recommendations')
st.warning('Research/educational prototype only. This system does not diagnose disease, prescribe treatment, or replace a qualified healthcare professional.')

DATA=Path(__file__).parent/'data'/'personalised_dataset.csv'
models_dir=Path(__file__).parent/'models'

@st.cache_resource
def get_models():
    return load_models()

@st.cache_data
def get_data():
    return pd.read_csv(DATA)

if not all((models_dir/f'{x}.joblib').exists() for x in ['Heart_Disease_Risk','Diabetes_Risk','Health_Risk']):
    st.error('Models are not trained yet. Run: python train.py')
    st.stop()

models=get_models(); df=get_data()

st.sidebar.header('Patient profile')
mode=st.sidebar.radio('Input mode',['Select synthetic patient','Enter patient manually'])

if mode=='Select synthetic patient':
    pid=st.sidebar.number_input('Patient ID',1,len(df),1)
    patient=df[df.Patient_ID==pid].iloc[0].to_dict()
else:
    patient={}
    c1,c2=st.sidebar.columns(2)
    patient['Age']=c1.number_input('Age',18,100,40)
    patient['Gender']=c2.selectbox('Gender',['Male','Female'])
    patient['BMI']=st.sidebar.number_input('BMI',10.0,60.0,25.0,0.1)
    patient['Smoking_Status']=st.sidebar.selectbox('Smoking status',['Non-smoker','Former smoker','Current smoker'])
    patient['Alcohol_Consumption']=st.sidebar.selectbox('Alcohol consumption',['Low','Moderate','High'])
    patient['Physical_Activity_Level']=st.sidebar.selectbox('Activity',['Sedentary','Lightly Active','Moderately Active','Highly Active'])
    patient['Diet_Type']=st.sidebar.selectbox('Diet',['Balanced','Mediterranean','Vegetarian','Vegan','Keto','High Protein'])
    patient['Blood_Pressure']=st.sidebar.text_input('Blood pressure (e.g. 120/80)','120/80')
    patient['Cholesterol']=st.sidebar.number_input('Total cholesterol',100.0,400.0,200.0)
    patient['Glucose_Level']=st.sidebar.number_input('Glucose level',50.0,300.0,95.0)
    patient['HbA1c']=st.sidebar.number_input('HbA1c',3.0,15.0,5.4,0.1)
    patient['PRS_Cardiometabolic']=st.sidebar.number_input('PRS cardiometabolic',-3.0,3.0,0.0)
    patient['PRS_Type2Diabetes']=st.sidebar.number_input('PRS type 2 diabetes',-3.0,3.0,0.0)
    for col in ['APOE_e4_Carrier','BRCA_Pathogenic_Variant','Family_History_CVD','Family_History_T2D']:
        patient[col]=int(st.sidebar.checkbox(col.replace('_',' ')))
    patient['Stress_Level']=st.sidebar.slider('Stress level',0.0,10.0,5.0)
    patient['Depression_Score']=st.sidebar.slider('Depression score',0,10,2)
    patient['Anxiety_Score']=st.sidebar.slider('Anxiety score',0,10,2)
    patient['Social_Isolation_Index']=st.sidebar.slider('Social isolation',0,10,2)
    patient['Sleep_Hours']=st.sidebar.number_input('Sleep hours',0.0,14.0,7.0,0.1)
    patient['Sleep_Quality']=st.sidebar.selectbox('Sleep quality',['Poor','Fair','Good','Excellent'])
    patient['Resting_Heart_Rate']=st.sidebar.number_input('Resting heart rate',35,150,70)
    patient['HRV']=st.sidebar.number_input('HRV',1,150,50)
    patient['Systolic_BP']=st.sidebar.number_input('Systolic BP',70,250,120)
    patient['Diastolic_BP']=st.sidebar.number_input('Diastolic BP',40,150,80)
    patient['LDL']=st.sidebar.number_input('LDL',20.0,300.0,110.0)
    patient['HDL']=st.sidebar.number_input('HDL',10.0,120.0,50.0)
    patient['Triglycerides']=st.sidebar.number_input('Triglycerides',20.0,600.0,120.0)
    patient['CRP']=st.sidebar.number_input('CRP',0.0,30.0,1.0)
    patient['eGFR']=st.sidebar.number_input('eGFR',5.0,180.0,90.0)
    patient['Waist_Circumference']=st.sidebar.number_input('Waist circumference',40.0,180.0,85.0)

row=pd.DataFrame([patient])
X=model_features(row)
results={}
for target,model in models.items():
    label=model.predict(X)[0]
    probs=model.predict_proba(X)[0]
    classes=model.classes_
    results[target]={'label':str(label),'probabilities':{str(c):float(p) for c,p in zip(classes,probs)}}

cols=st.columns(3)
for col,(target,res) in zip(cols,results.items()):
    col.metric(target.replace('_',' '),res['label'])
    with col.expander('Model probabilities'):
        for k,v in res['probabilities'].items():
            st.write(f'**{k}:** {v:.1%}')

st.subheader('Personalised recommendations')
recs=generate_recommendations(patient,results)
if not recs:
    st.info('No specific rule was triggered. Maintain healthy habits and use routine preventive care.')
else:
    for rec in recs:
        icon={'High':'🔴','Medium':'🟠','Low':'🟢'}.get(rec.priority,'⚪')
        with st.container(border=True):
            st.markdown(f'### {icon} {rec.title}')
            st.write(f'**Category:** {rec.category}  •  **Priority:** {rec.priority}')
            st.write(rec.action)
            st.caption('Why: '+rec.reason)

st.subheader('Patient data used')
show_cols=['Age','Gender','BMI','Smoking_Status','Alcohol_Consumption','Physical_Activity_Level','Diet_Type','Glucose_Level','HbA1c','Stress_Level','Sleep_Hours','Sleep_Quality','Systolic_BP','Diastolic_BP','LDL','HDL','Triglycerides','eGFR']
st.dataframe(pd.DataFrame({c:[patient.get(c,'')] for c in show_cols}).T.rename(columns={0:'Value'}),use_container_width=True)
