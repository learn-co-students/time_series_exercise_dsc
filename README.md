This is a Divvy Bike (bike share) data set from Chicago. It is downloaded from [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Divvy-Trips/fg6s-gzvg/data). 

The csv in the data folder has been filtered to look at only 1 pickup station: Clark and Lake. That is a busy intersection in the Loop.  


```python
# Import the dataset

df = None
```


```python
#__SOLUTION__
# Raw Dataset
import pandas as pd
df = pd.read_csv('data/Divvy_Trips.csv')
df.head()
```

We will be interested in the number of Divvy rentals over any given **week** at the Clark and Lake pickup dock.  

In order to begin inspecting the data across different time intervals, we need to give our dataset a datetime index.


```python
# Create datetime index based on the start time

```


```python
#__SOLUTION__
# Create datetime index based on the start time

df['START TIME'] = pd.to_datetime(df['START TIME'])
df.set_index(df['START TIME'], inplace=True)
```

Now that we have a datatime index, we can resample to a time frame of our choosing.

We are going from a smaller unit of time to a larger unit, so we will downsample.  With any resampling technique, we need to specify an aggregate function. In this case, we want to count the number of rides per hour.


```python
# Downsample to the week and count the number of rides

```


```python
#__SOLUTION__
# Downsample to the hour and count the number of rides
df_week = df.resample('W').count()['TRIP ID']
df_week
```

Before we start looking at patterns in the data, perform a train test split.  We can't do it like we usually do. The sklearn version we use randomly picks points to assign to each set.  With time series data, we have to preserve the order, since the models depend on prior days.

We will split of the last 52 weeks, 1 year, for our test set.


```python
# Split the data so that 52 weeks are in the test set
```


```python
#__SOLUTION__
# Train test split
train = df_week[:-52]
test = df_week[-52:]

```


```python
# Plot the training data
train.plot()
```

Describe the seasonality you see in the plot above.

==SOLUTION==
There is a repeating pattern that has a period of 1 year.  There are the lowest bike rental counts in the winter, most in the summer.

It is harder to see if there is an overall upward or downard trend. To investigate trend, let's look at the rolling mean across a year.


```python
# Plot the rolling mean with a window of a year
```


```python
#__SOLUTION__
train.rolling(52).mean().plot()
```

That clearly visualizes an upward trend in our data, especially towards the first half of our dataset.

One way to remove trend this trend from the data is differencing.  A first order difference with a period of 1 (the default), will leave us with a timeseries composed of the change of rentals from 1 week to the next.


```python
# Apply a 1st order difference to the time series and plot the rolling mean
```


```python
#__SOLUTION__
train.diff().rolling(52).mean().plot()

```

The trend is much less pronounced, but still looks to be sloping downwards. Let's see if a second order difference stabilizes it further.


```python
# Apply a 2nd order difference: i.e. difference the difference, and plot the rolling mean.
```


```python
#__SOLUTION__
train_week.diff().diff().rolling(52).mean().plot()

```

The above eda gives us some clues about how to go about choosing the order in our SARIMAX models.  
The first choice of order will be the non-seasonal order. `p,d,q`. You can remember what those letters stand for by thinking AR=p, d=difference, MA=q.   The letters of the acronym aligns with the letters in the order argument.


```python
# Fit a SARIMAX model on the trian set with a first order difference.
```


```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

sm = SARIMAX(<your_code_here>).fit()

```


```python
#__SOLUTION__
sm = SARIMAX(train, order=[0,1,0]).fit()

```


```python
# Create a set of predictions for the train data (just use predict)
```


```python
#__SOLUTION__
sm.predict()
```


```python
# Calculate the training root mean squared error (same syntax as a linear regression prediction)
```


```python
#__SOLUTION__
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(train, sm.predict(), squared=False)
rmse
```

Next, calculate the test rmse.  To do this, you have to pass in the date of the first test element and the last test element as arguments.


```python
#__SOLUTION__
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(test, sm.predict(test.index[0], test.index[-1]), squared=False)
rmse
```

The function below will print out train and test scores with a given set of orders.  


```python
def print_ts_metrics(endog=train, test=test, order=[0,0,0], seasonal_order=[0,0,0,0]):
    
    '''
    Print out RMSE for a given set of orders (seasonal and non-seasonal)
    
    Return the model fit on the training set.
    '''
    
    sm = SARIMAX(endog, order=order, seasonal_order=seasonal_order).fit()
    
    print(mean_squared_error(endog, 
                             sm.predict(endog.index[0], endog.index[-1], 
                                        typ='levels'), 
                                        squared=False))
    
    print(mean_squared_error(test, 
                             sm.predict(test.index[0], test.index[-1], 
                                        typ='levels'), 
                                        squared=False))
    return sm


```

The function below will plot the test predictions along with the true test values.


```python
def plot_predictions(test=test, sm=sm):
    
    '''
    Pass a test set, as well as a model fit to the training set 
    to this function, and plot the test predictions against
    the true test values
    '''
    
    sm.predict(test.index[0], test.index[-1], typ='levels').plot()
    test.plot()
    
```

Try out a few non-seasonal order combinations and consider the effect on the test rmse.


```python
# set print_ts_metrics equal to model, then feed that model to plot_predictions
model = None
plot_predictions(test, model)
```


```python
#__SOLUTION__
model = print_ts_metrics(train, test, [1,1,1])
plot_predictions(test, model)
```


```python
#__SOLUTION__
model = print_ts_metrics(train, test, [2,1,2])
plot_predictions(test, model)

```

The models make predictions, and different non-seasonal orders affect the rmse on the test set.  But the predictions certainly leave something to be desired.

From our eda above, we know that the data has seasonality.  We can account for this using the seasonal_order parameter.  The first step in generating better predictions is to choose the correct period.  The seasonal_order list should have for elements: P,D,Q,period (in that order).

Take the best parameters from above, then add to it a 1st order seasonal difference that makes sense with the eda.


```python
#__SOLUTION__
model = print_ts_metrics( train, test, [2,1,1],[0,1,0,52])
plot_predictions(test, model)

```
