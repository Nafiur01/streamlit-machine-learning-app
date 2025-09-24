import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('Penguin Species Prediction with Machine Learning')

st.info('We are going to predict penguin species based on the dataset using machine learning')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop(columns='species',axis=1)
  X_raw


  st.write('**Y**')
  Y_raw = df.species
  Y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')
  
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
  bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male','female'))

# create dataframe from the input
  data = {
    'island' : island,
    'bill_length_mm' : bill_length_mm,
    'bill_depth_mm' : bill_depth_mm,
    'flipper_length_mm' : flipper_length_mm,
    'body_mass_g' : body_mass_g,
    'sex' : gender
  }
  input_df = pd.DataFrame(data,index=[0])
  input_penguins = pd.concat([input_df,X_raw], axis=0)

#encode cat column
encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins,prefix=encode)
X = df_penguins[1:]
input_row = df_penguins[:1]

# encode Y
target_mapper = {
  'Adelie' : 0,
  'Chinstrap' : 1,
  'Gentoo' : 2,
}

def target_encode(val):
  return target_mapper[val]

y = Y_raw.apply(target_encode)

with st.expander('Input Features'):
  st.write('**Input**')
  input_df
  st.write('**Combined Input Features**')
  input_penguins

with st.expander('Data Preparation'):
  st.write('**Encoded X**')
  input_row
  st.write('**Y**')
  y


# Machine Learning Training
clf = RandomForestClassifier()
clf.fit(X,y)
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)
prediction_proba


