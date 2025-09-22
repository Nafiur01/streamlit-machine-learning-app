import streamlit as st
import pandas as pd
import numpy as np

st.title('Penguin Species Prediction with Machine Learning')

st.info('We are going to predict penguin species based on the dataset using machine learning')

with st.expander('Data'):
  st.write('**Raw Data**')
