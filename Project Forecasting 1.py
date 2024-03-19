#!/usr/bin/env python
# coding: utf-8

# # AsianPaints Stock Price Forecasting 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


# # importing dataset from database

# In[2]:


df = pd.read_csv("G:\\My Drive\\ExcelR\\ASIANPAINT.csv")
df


# # Convert Date column to datetime

# In[3]:


df['Date'] = pd.to_datetime(df['Date'])
df


# # converting Date Column into Index

# In[4]:


df.set_index('Date', inplace=True)
df 


# # performing 1. EDA

# In[5]:


df.head()


# # the above data displaying the first five Rows of the dataset

# In[6]:


df.tail()


# # the above data displaying the last five Rows of the dataset

# In[7]:


df.shape


# In[8]:


df.info()


# # a. Describing the dataset

# In[9]:


df.describe()


# #  b. Cleaning the data

# # Assuming 'symbol' and 'series' are not needed, Because two columns contains the same info through out the dataset

# In[10]:


df.drop(['Symbol', 'Series'], axis=1, inplace=True)
df


# # c. Checking invalid records 

# In[11]:


df.isnull()


# # d. Missing value detection and imputation

# In[12]:


missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)


# # Replacing the null values

# In[13]:


df['Trades'].fillna(df['Trades'].mean(), inplace=True)
df['Deliverable Volume'].fillna(df['Deliverable Volume'].mean(), inplace=True)
df['%Deliverble'].fillna(df['%Deliverble'].mean(), inplace=True)
df.to_csv('modified_file.csv', index=False)
df


# In[14]:


missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)


# # e. Duplicated records

# In[15]:


duplicate_records = df[df.duplicated()]
print("Duplicate records:", duplicate_records)


# # f. Outliers

# In[16]:


# We used boxplot method to detect outliers


# In[17]:


import matplotlib.pyplot as plt
plt.figure(figsize=(22, 10)) 
# Create a boxplot
df.boxplot(column=['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP','%Deliverble'])
plt.show()


# # Resloving the outliers

# In[18]:


# Define the threshold for capping outliers
threshold = 1 # You can adjust this threshold value as per your requirement
# Define the columns you want to cap outliers for
columns_to_cap = ['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']
# Loop through each column and cap outliers
for column in columns_to_cap:
    # Calculate the mean and standard deviation of the column
    mean = df[column].mean()
    std = df[column].std()
    
    # Calculate the upper and lower bounds for capping outliers
    upper_bound = mean + threshold * std
    lower_bound = mean - threshold * std
    
    # Cap outliers
    df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

# Save the capped dataset
df.to_csv("ASIANPAINT_capped.csv", index=False)


# In[19]:


import matplotlib.pyplot as plt
plt.figure(figsize=(22, 10)) 
# Create a boxplot
df.boxplot(column=['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP','%Deliverble'])
plt.show()


# # 2. Data Visualization

# # we have used some Graphical methods like Histogram, Boxplot, Correlation matrix heatmap and Time series plot
# 

# # Histogram

# In[20]:


plt.figure(figsize=(10, 6))
sns.histplot(df['Close'], bins=20, kde=True)
plt.title('Histogram of Close Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.show()


# #conclusion 
# 
# Above histogram shows that the Histogram is skewed rightly 
# postiviely skewed
# 

# # Density plot

# In[21]:


# Create density plot
plt.figure(figsize=(8, 6))
sns.kdeplot(df['Close'], shade=True)
plt.title('Density Plot for {}'.format('Close'))
plt.xlabel('Close')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# #conclusion
# the Density plot shows the density of the close Price. 
# In the plot the close price density between 0 to 1000 is more.  

# # boxplot

# In[22]:


plt.figure(figsize=(10, 6))
df.boxplot(column='Close', figsize=(12, 6))
plt.title('Boxplot of Closing Prices')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# #conclusion 
# From the above box plot we can see that there are some outliers.
# There are outliers from the range of 3400 Close Price.
# 

# # Correlation matrix heatmap

# In[23]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# # Time series plot

# In[24]:


# Plotting Open Price Over Time

plt.figure(figsize=(10, 6), facecolor='white')
plt.plot(df['Open'], label='Open Price', color='Green')
plt.title('Open Price Over Time')
plt.xlabel('Date')
plt.ylabel('Open')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #conclusion
# From the Above Time Series Plot for open price over Time.
# from 2000 to 2008 there is a constant growth in open price.
# But from 2008 to 2009 the price was slowly droped.
# Right from the 2009 there is a rapid increase in open price till 2013.
# But Right from 2013 there is danger drop of price.
# And then the open price continous kept increasing.

# # Plotting Closing Price Over Time

# In[25]:


plt.figure(figsize=(10, 6), facecolor='white')
plt.plot(df['Close'], label='Close Price', color='blue')
plt.title('Closing Price Over Time')
plt.xlabel("Date")
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #conclusion
# From the Above Time Series Plot for close price over Time.
# from 2000 to 2008 there is a constant growth in close price.
# But from 2008 to 2009 the price was slowly droped.
# Right from the 2009 there is a rapid increase in close price till 2013.
# But Right from 2013 there is danger drop of price.
# And then the close price continous kept increasing.

# In[26]:


plt.figure(figsize=(10, 6), facecolor='white')
plt.plot(df['Prev Close'], label='Prev Close Price', color='Red')
plt.title('Prev Close Over Time')
plt.xlabel('Date')
plt.ylabel('Prev Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# #conclusion
# From the Above Time Series Plot for prev close over Time.
# from 2000 to 2008 there is a constant growth in prev close price.
# But from 2008 to 2009 the price was slowly droped.
# Right from the 2009 there is a rapid increase in open price till 2013.
# But Right from 2013 there is danger drop of price.
# And then the prev close price continous kept increasing.

# In[27]:


# Filter columns
selected_columns = ['Prev Close', 'Open', 'Close', ]
df_selected = df[selected_columns]

# Plot
plt.figure(figsize=(10, 6))
for col in df_selected.columns:
    plt.plot(df_selected.index, df_selected[col], label=col)

plt.title('Time Series Plot (Selected Columns)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# # pair plot

# In[28]:


# Drop non-numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_cols]

# Create pair plot
sns.pairplot(df_numeric)
plt.show()


# # Feature Engineering

# In[29]:


# Calendar Features
df['Day_of_Week'] = df.index.dayofweek
df['Month'] = df.index.month
df['Quarter'] = df.index.quarter


# In[30]:


# Price Change Ratios
df['Daily_Return'] = df['Close'].pct_change()
df['Weekly_Return'] = df['Close'].pct_change(periods=5)


# In[31]:


# Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Close'], model='additive', period=30)  # Assuming a seasonal period of 30 days
df['Trend'] = result.trend
df['Seasonal'] = result.seasonal
df['Residual'] = result.resid


# In[32]:


# Display the engineered features
print(df.head())


# # Model Building

# # ARIMA Model Prediction

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset
df = pd.read_csv("ASIANPAINT.csv", index_col="Date", parse_dates=True)

# Ensure that the column names are correctly spelled and match your dataset
print(df.columns)

# Extracting only the 'Close' column
close_data = df['Close']

# Splitting the data into train and test sets
train_size = int(len(close_data) * 0.7)
train_data = close_data.iloc[:train_size]
test_data = close_data.iloc[train_size:]

# Fit ARIMA model
model = ARIMA(train_data, order=(5,1,0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test_data))

# Plotting predictions
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Train Data')
plt.plot(test_data.index, test_data, label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('ARIMA Model Prediction')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error

# Calculate MSE
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)



# # Moving Average Forecast

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt

# Define the window size for the moving average
window_size = 10

# Calculate the moving average
df['MA'] = df['Close'].rolling(window=window_size).mean()

# Plotting
plt.plot(df.index, df['Close'], label='Actual')
plt.plot(df.index, df['MA'], label=f'Moving Average (Window Size {window_size})')
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Moving Average Forecast')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error

# Remove rows with NaN values in 'MA' column
df_ma = df.dropna(subset=['MA'])

# Calculate MSE
mse = mean_squared_error(df_ma['Close'], df_ma['MA'])
print('Mean Squared Error (MSE) for Moving Average Forecast:', mse)


# # LSTM Forecast

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

# Define function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Set sequence length and split data into train and test sets
sequence_length = 10
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for training
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Build LSTM model
model = Sequential([
    LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate model
loss = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)

# Inverse scale
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size+sequence_length:], y_test_inv, label='Actual')
plt.plot(df.index[train_size+sequence_length:], predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('LSTM Forecast')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error

# Assuming 'predictions' and 'y_test_inv' contain predicted and actual values respectively

# Calculate MSE
mse = mean_squared_error(y_test_inv, predictions)
print('Mean Squared Error (MSE):', mse)
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_inv, predictions)
print('Mean Absolute Error (MAE):', mae)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
print('Root Mean Squared Error (RMSE):', rmse)


# # Simple Exponential Smoothing Forecast(SES)

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the smoothing parameter
alpha = 0.2

# Split data into training and testing sets
train_size = int(0.7 * len(df))  # 80% for training, 20% for testing
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Simple Exponential Smoothing implementation
def simple_exponential_smoothing(series, alpha):
    """
    Perform Simple Exponential Smoothing forecast.
    
    Args:
    - series: Time series data as a pandas Series.
    - alpha: Smoothing parameter (0 < alpha < 1).
    
    Returns:
    - Forecasted values as a numpy array.
    """
    forecasts = [series.iloc[0]]  # Initial forecast is the first observation
    for i in range(1, len(series)):
        forecast = alpha * series.iloc[i] + (1 - alpha) * forecasts[-1]
        forecasts.append(forecast)
    return np.array(forecasts)

# Calculate forecast using Simple Exponential Smoothing on the training data
forecast_train = simple_exponential_smoothing(train_data['Close'], alpha)

# Plotting
plt.figure(figsize=(10, 6))

# Plot training data and forecasted values
plt.plot(train_data.index, train_data['Close'], label='Training Data')
plt.plot(train_data.index, forecast_train, label='Forecast (Training)', linestyle='--')

# Plot testing data
plt.plot(test_data.index, test_data['Close'], label='Testing Data', color='orange')

# Plot legend and labels
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Simple Exponential Smoothing Forecast')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error

# Calculate MSE
mse = mean_squared_error(train_data['Close'][1:], forecast_train[:-1])
print('Mean Squared Error (MSE):', mse)


# We conclude that comparing with all models LSTM is giving low MSE value.So we choose the LSTM for model deployment.
