from cProfile import label
from tkinter import E
from matplotlib.style import use
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas_datareader as data
import pylab
import numpy as np


html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Stock Price Prediction</h2>
</div>
<br>
<br>
"""
st.markdown(html_temp,unsafe_allow_html=True)

user_input = st.text_input('Enter Stock Ticker','AAPL')
st.text('Find Stock Ticker from yahoo finance Eg.TSLA,GOOG')

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #1888d9;
    color:balck;
}
div.stButton > button:hover {
    background-color: #2ad44c;
    color:white;
    }
</style>""", unsafe_allow_html=True)

if st.button("Predict"):
    #st.title('Stock trend')

    start  = '2010-01-01'
    end =  '2019-12-31'

    

    df = data.DataReader(user_input,'yahoo',start,end)

    st.subheader('Data from 2010 - 2019')
    st.write(df.head(3))
    st.write(df.tail(3))

    st.subheader('Describing data')
    st.write(df.describe())

    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close,label='Closing price')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100 days Moving Avg')
    ma100 = df.Close.rolling(100).mean() 
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'r',label='MA100days')
    plt.plot(df.Close,'b',label='Real Closing Price')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    st.text('When the red line is above the green line then it is Uptrend')
    st.text('When the red line is below the green line then it is Downtrend')
    ma100 = df.Close.rolling(100).mean() 
    ma200 = df.Close.rolling(200).mean() 
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'r',label='MA100')
    plt.plot(ma200,'g',label='MA200')
    plt.plot(df.Close,'b',label='Real Closing Price')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

    #################### split data train test 70:30
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)


    ##### load already trained model
    model = load_model('keras_model.h5')

    ### testing part
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing,ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test,y_test = np.array(x_test),np.array(y_test)

    ### predictions
    y_predicted = model.predict(x_test)
    #scaleup all values
    scaler = scaler.scale_    

    scalar_factor = 1/scaler[0]
    y_predicted = y_predicted * scalar_factor
    y_test = y_test * scalar_factor

    ########### final visualization
    st.subheader('Prediction vs Original')
    lstmfig = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(y_predicted,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(lstmfig)

    html_temp2 = """
    <div style="background-color:tomato;padding:10px">
    <h3 style="color:white;text-align:center;">Thank You !</h3>
    </div>
    <br>
    """
    st.markdown(html_temp2,unsafe_allow_html=True)

    
##### check for run successfully
print('hello') 