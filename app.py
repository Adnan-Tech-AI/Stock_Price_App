import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error



st.title('Stock Price Analytics & Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App')
st.sidebar.info("Mini Project by  : \n Amatullah Fatima \n Mehvish Fatima")

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()



@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df




option = st.sidebar.selectbox("Select Company",["Google Inc.","Microsoft","Tesla","Air BNB","Meta"])
option = option.upper()
print(option)
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
# if st.sidebar.button('Send'):
    # if start_date < end_date:
if option=="GOOGLE INC.":
    st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
    stock="GooG"
elif option=="MICROSOFT":
    st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
    # df=download_data("MSFT", start_date, end_date)
    stock="MSFT"
elif option=="TESLA":
    st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
    # df=download_data("TSLA", start_date, end_date)
    stock="TSLA"
elif option=="META":
    st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
    # df=download_data("META", start_date, end_date)
    stock="META"
elif option=="AIR BNB":
    st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
    # df=download_data("ABNB", start_date, end_date)
    stock = "ABNB"

    # else:
    #     st.sidebar.error('Error: End date must fall after start date')


data = download_data(stock,start_date,end_date)


scaler = StandardScaler()

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb=data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)


def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))



def predict():
    num = st.number_input('How many days of forecast?', value=1)
    num = int(num)
    if st.button('Predict'):
        engine = LinearRegression()
        model_engine(engine, num)
      

def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = df.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'Predicted with the accuracy of : {r2_score(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Predicted Closing Price For Day {day} is : {i}')
        day += 1


if __name__ == '__main__':
        main()
