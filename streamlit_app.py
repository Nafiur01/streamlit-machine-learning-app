import streamlit as st
import pandas as pd
import numpy as np

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
