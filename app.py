# -*- coding: utf-8 -*-
"""
Created on Fri May 1 00:27:59 2025

@author: Aftab
"""

import pickle
import streamlit as st
import pandas as pd

# Load all components
with open("naukari_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
le1 = bundle["le1"]
le2 = bundle["le2"]
label_encoders = bundle["label_encoders"]


st.set_page_config(page_title="naukari Skills Prediction", layout="wide")    
page_bg_img = '''
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://img.freepik.com/free-photo/flat-lay-workstation-with-copy-space-laptop_23-2148430879.jpg?semt=ais_hybrid&w=740");
        background-size: cover;
        background-attachment: fixed;
    }
    
    [data-testid="stAppViewContainer"], h1, h2, h3, h4, h5, h6, p {
        color: white !important;
    }
    
    #skill_output,skill_output b{
        color:orange !important;
        font-size:28px;
        font-weight:400;
    }
    
    div[data-baseweb="input"] label {
        color: white !important;
    }

    .stButton > button {
            background-color: blue !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 8px;
            padding: 10px 50px;
            border: none;
            margin-top:20px;
    }

    .stButton > button:hover {
            background-color: gray !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
       background-color: white !important;
   }

   .stSelectbox div[data-baseweb="single-value"] > div {
       color: black !important;
   }

   /* Dropdown items background & text color */
   .stSelectbox div[data-baseweb="option"] >div {
       background-color: white !important;
       color: black !important;
   }
</style>
'''

# Injecting CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


# Defining Prediction Function

def predict(number_of_reviews, ratings, min_exp, location, company_name, is_salary_disclosed, max_salary, max_exp, job_title, min_salary):
    
    data = [[number_of_reviews, ratings, min_exp, location, company_name, is_salary_disclosed, max_salary, max_exp, job_title, min_salary]]
    
    new_df = pd.DataFrame(data, columns=['Reviews', 'Ratings', 'MinExp', 'Location', 'Company',
       'Salary_Disclosed', 'MaxSalary', 'MaxExp', 'Title', 'MinSalary'])
    
    for col in label_encoders.keys():
        if col in new_df.columns:
            new_df[col] = label_encoders[col].transform(new_df[col])
    
    prediction = model.predict(new_df)
    
    y_pred_label = le2.inverse_transform(prediction)
    y_real_label = le1.inverse_transform(y_pred_label)
    
    return y_real_label


if 'data' not in st.session_state:
    st.session_state.data = []


def main():
    
    st.title('Naukari Skills Predictor')

    st.markdown("### Enter Required Parameters:")
        
    col1,col2,col3,col4 = st.columns(4)
    
    with col1:
        
        company_name_options = ['Accenture', 'CoinDCX', 'Oracle', 'Siemens', 'Rave Technologies',
           'HealthSpring', 'Citibank, N.A', 'Snaphunt', 'Duff & Phelps',
           'BNY Mellon', 'Credit Suisse', 'Prodair Air Products',
           'Air Products', 'Ubisoft', 'CompuCom', 'Kraftmaid Services India',
           'Method Studios', 'Company3 Method India Private Limited ', 'RRD',
           'Thinksynq Solutions', 'Shell', 'Icon Clinical Research',
           'Aspire Systems', 'Icon Pharmaceutical s',
           'Associated Auto Solutions International Pvt. Ltd.',
           'Sona Comstar', 'NatWest Group', 'Eversendai']
        
        company_name = st.selectbox(label='Company',options=company_name_options)
        
        location_options = ['Mumbai', 'Hyderabad/Secunderabad', 'Pune', 'Chennai',
           'Delhi / NCR', 'Bangalore/Bengaluru', 'Gurgaon/Gurugram',
           'Aurangabad', 'Vadodara']
        
        location = st.selectbox(label='Job location',options=location_options)
        
        is_salary_disclosed_label = st.selectbox("Is salary disclosed?", ["No", "Yes"])
        is_salary_disclosed = 1 if is_salary_disclosed_label == "Yes" else 0
        
        disable_fields = is_salary_disclosed == 0
        
        title_options = ['Senior Analyst', 'Business Analyst', 'Analyst', 'Analyst/Manager','Analyst/Consultant']
        job_title = st.selectbox(label='Job title',options=title_options)
    
    with col2:
        
        number_of_reviews = st.number_input('Number of reviews for the company',step=1,min_value=0)
        
        min_exp = st.number_input('Minimum experience',step=1,min_value=0,max_value=10)
        
        min_salary = st.number_input("Minimum salary",step=1,min_value=-1,value=0 if is_salary_disclosed else -1,disabled=disable_fields)
        
        
    with col3:
        
        ratings = st.number_input('Ratings of the company',step=0.1,format="%.1f",min_value=0.0,max_value=5.0)
        
        max_exp = st.number_input('Maximum experience',step=1,min_value=0,max_value=15)
        
        max_salary = st.number_input("Maximum salary",step=1,min_value=-1,value=0 if is_salary_disclosed else -1,disabled=disable_fields)
        
        button = st.button('Predict')
        
    
    
    if button:
        result = predict(number_of_reviews, ratings, min_exp, location, company_name, is_salary_disclosed, max_salary, max_exp, job_title, min_salary)
        
        st.markdown(f"<span id='skill_output'>{result[0]} skill(s) are required for this job.</span>",unsafe_allow_html=True)
        
        st.session_state.data.append({
            'Name of the Company': company_name,
            'Number of Reviews': number_of_reviews,
            'Comapny Ratings': ratings,
            'Job Location': location,
            'Minimum Experience': min_exp,
            'Maximum Experience': max_exp,
            'Job Title': job_title,
            'Is Salary Disclosed': is_salary_disclosed,
            'Minimum Salary': min_salary,
            'Maximum Salary': max_salary,
            'Skill(s) Required': result
            })
        
    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data)
        st.write("### Previous Predictions")
        st.dataframe(df)
        

        
if __name__ == '__main__':
    main()