import math
import numpy as np
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

model = load_model('C:/Users/HP/Desktop/Vinay/MajorProject/newModel.keras')

st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2012-01-01'
end = '2023-12-14'

df = yf.download(stock, start ,end)

#Create a new dataframe with only Close column
data=df.filter(['Close'])

#Convert the dataframe to a numpy array
dataset=data.values


st.subheader('Stock Data')

st.write(df.head())
st.write(df.tail())

training_data_len=math.ceil(len(df) * .8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

#Function to calculate RMSE
def calculate_rmse(moving_average, actual_close_prices, window):
    # Calculate squared errors
    squared_error = (moving_average.dropna() - actual_close_prices[window - 1:]) ** 2    
    # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse = np.mean(squared_error)
    rmse = np.sqrt(mse)    
    return rmse


st.subheader('Price vs MA200')
ma_200_days = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.plot(ma_200_days)
plt.legend(['Price', 'MA200'], loc='lower right')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
st.pyplot(fig3)

rmse_ma200 = calculate_rmse(ma_200_days, df['Close'], 200)
st.markdown(f"##### RMSE :  {rmse_ma200}")



st.subheader('Price vs MA200 vs MA100')
ma_100_days = df.Close.rolling(100).mean()
#Visualize the Moving Averages 100
fig2=plt.figure(figsize=(16,8))
plt.plot(df['Close'])
plt.plot(ma_200_days)
plt.plot(ma_100_days)
plt.legend(['Price', 'MA200','MA100'], loc='lower right')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
st.pyplot(fig2)

rmse_ma100 = calculate_rmse(ma_100_days, df['Close'], 100)
st.markdown(f"##### RMSE :  {rmse_ma100}")


st.subheader('Price vs MA100 vs MA50')
ma_50_days = df.Close.rolling(50).mean()
#Visualize the Moving Averages 100
fig1=plt.figure(figsize=(16,8))
plt.plot(df['Close'])
plt.plot(ma_100_days)
plt.plot(ma_50_days)
plt.legend(['Price', 'MA100','MA50'], loc='lower right')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
st.pyplot(fig1)

rmse_ma50 = calculate_rmse(ma_50_days, df['Close'], 50)
st.markdown(f"##### RMSE :  {rmse_ma50}")



#Create the testing data set
test_data=dataset[training_data_len - 60: , :]

#Scale the test data
scaled_test_data = scaler.fit_transform(test_data)  

#Create the data sets x_test and y_test
x_test = []
y_test=dataset[training_data_len:, :]
for i in range(60, len(scaled_test_data)): 
    x_test.append(scaled_test_data[i-60:i, 0])

#Convert the data to a numpy array x_test = np.array(x_test)
x_test = np.array(x_test)

#Reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


#Get the models predicted price values
predictions = model.predict(x_test)

#Inverse transform them 
predictions= scaler.inverse_transform(predictions)


st.subheader('Original Price vs Predicted Price (Using LSTM)')
#Plot the data
train = df[: training_data_len] 
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
fig4=plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Test', 'predict'], loc='lower right')
st.pyplot(fig4)

#Actual vs Predictions
st.write(valid.tail())

#Get the RMSE
n = len(y_test)
rmse = np.sqrt(np.sum((y_test - predictions) ** 2) / n)

st.markdown(f"##### RMSE :  {rmse}")


#Get new Quote
new_quote = yf.download(stock, start, end)

#For getting tomorrows predictions

#Create a new dataframe 
new_df = new_quote.filter(['Close'])
#Get the last 60 day closing price values and convert the dataframe to an array
last_60_days=new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled=scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test=np.array(X_test)
#Reshape the data
X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price=model.predict(X_test)
#undo the scaling
pred_price=scaler.inverse_transform(pred_price)
st.markdown(f"#### Next day prediction = {pred_price[0][0]}")



start = '2023-12-14'
end = '2023-12-15'
new_quote2 = yf.download(stock, start, end)


# Display the actual closing price

appl = new_quote2.iloc[0]
date = appl.name.date()  # Extract the date
closing_price = appl['Close']  # Extract the closing price
st.markdown(f"#### Actual on Date: {date} = {closing_price}")
