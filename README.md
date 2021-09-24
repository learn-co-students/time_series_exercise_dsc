This is a Divvy Bike (bike share) data set from Chicago. It is downloaded from [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Divvy-Trips/fg6s-gzvg/data). 

The csv in the data folder has been filtered to look at only 1 pickup station: Clark and Lake. That is a busy intersection in the Loop.  


```python
# Import the dataset

df = None
```


```python
# Raw Dataset
import pandas as pd
df = pd.read_csv('data/Divvy_Trips.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TRIP ID</th>
      <th>START TIME</th>
      <th>STOP TIME</th>
      <th>BIKE ID</th>
      <th>TRIP DURATION</th>
      <th>FROM STATION ID</th>
      <th>FROM STATION NAME</th>
      <th>TO STATION ID</th>
      <th>TO STATION NAME</th>
      <th>USER TYPE</th>
      <th>GENDER</th>
      <th>BIRTH YEAR</th>
      <th>FROM LATITUDE</th>
      <th>FROM LONGITUDE</th>
      <th>FROM LOCATION</th>
      <th>TO LATITUDE</th>
      <th>TO LONGITUDE</th>
      <th>TO LOCATION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4905583</td>
      <td>04/18/2015 01:00:00 PM</td>
      <td>04/18/2015 01:04:00 PM</td>
      <td>3225</td>
      <td>277</td>
      <td>38</td>
      <td>Clark St &amp; Lake St</td>
      <td>286</td>
      <td>Franklin St &amp; Quincy St</td>
      <td>Subscriber</td>
      <td>Male</td>
      <td>1989.0</td>
      <td>41.886021</td>
      <td>-87.630876</td>
      <td>POINT (-87.630876 41.886021)</td>
      <td>41.878724</td>
      <td>-87.634793</td>
      <td>POINT (-87.634793 41.878724)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4905756</td>
      <td>04/18/2015 01:08:00 PM</td>
      <td>04/18/2015 01:16:00 PM</td>
      <td>3224</td>
      <td>454</td>
      <td>38</td>
      <td>Clark St &amp; Lake St</td>
      <td>26</td>
      <td>McClurg Ct &amp; Illinois St</td>
      <td>Subscriber</td>
      <td>Male</td>
      <td>1986.0</td>
      <td>41.886021</td>
      <td>-87.630876</td>
      <td>POINT (-87.630876 41.886021)</td>
      <td>41.891020</td>
      <td>-87.617300</td>
      <td>POINT (-87.6173 41.89102)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4906415</td>
      <td>04/18/2015 01:48:00 PM</td>
      <td>04/18/2015 02:45:00 PM</td>
      <td>3226</td>
      <td>3415</td>
      <td>38</td>
      <td>Clark St &amp; Lake St</td>
      <td>255</td>
      <td>Indiana Ave &amp; Roosevelt Rd</td>
      <td>Customer</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.886021</td>
      <td>-87.630876</td>
      <td>POINT (-87.630876 41.886021)</td>
      <td>41.867888</td>
      <td>-87.623041</td>
      <td>POINT (-87.623041 41.867888)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4906418</td>
      <td>04/18/2015 01:48:00 PM</td>
      <td>04/18/2015 02:45:00 PM</td>
      <td>3223</td>
      <td>3398</td>
      <td>38</td>
      <td>Clark St &amp; Lake St</td>
      <td>255</td>
      <td>Indiana Ave &amp; Roosevelt Rd</td>
      <td>Customer</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.886021</td>
      <td>-87.630876</td>
      <td>POINT (-87.630876 41.886021)</td>
      <td>41.867888</td>
      <td>-87.623041</td>
      <td>POINT (-87.623041 41.867888)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4908035</td>
      <td>04/18/2015 03:25:00 PM</td>
      <td>04/18/2015 03:38:00 PM</td>
      <td>2893</td>
      <td>793</td>
      <td>38</td>
      <td>Clark St &amp; Lake St</td>
      <td>86</td>
      <td>Eckhart Park</td>
      <td>Subscriber</td>
      <td>Female</td>
      <td>1981.0</td>
      <td>41.886021</td>
      <td>-87.630876</td>
      <td>POINT (-87.630876 41.886021)</td>
      <td>41.896373</td>
      <td>-87.660984</td>
      <td>POINT (-87.660984 41.896373)</td>
    </tr>
  </tbody>
</table>
</div>



We will be interested in the number of Divvy rentals over any given **week** at the Clark and Lake pickup dock.  

In order to begin inspecting the data across different time intervals, we need to give our dataset a datetime index.


```python
# Create datetime index based on the start time

```


```python
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
# Downsample to the hour and count the number of rides
df_week = df.resample('W').count()['TRIP ID']
df_week
```




    START TIME
    2015-04-19     24
    2015-04-26    106
    2015-05-03    182
    2015-05-10    160
    2015-05-17    182
                 ... 
    2019-12-08    265
    2019-12-15    200
    2019-12-22    178
    2019-12-29    119
    2020-01-05     32
    Freq: W-SUN, Name: TRIP ID, Length: 247, dtype: int64



Before we start looking at patterns in the data, perform a train test split.  We can't do it like we usually do. The sklearn version we use randomly picks points to assign to each set.  With time series data, we have to preserve the order, since the models depend on prior days.

We will slit of the last 52 weeks, 1 year, for our test set.


```python
# Split the data so that 52 weeks are in the test set
```


```python
# Train test split
train = df_week[:-52]
test = df_week[-52:]

```


```python
# Plot the training data
train.plot()
```




    <AxesSubplot:xlabel='START TIME'>




    
![png](index_files/index_13_1.png)
    


Describe the seasonality you see in the plot above.

There is a repeating pattern that has a period of 1 year.  There are the lowest bike rental counts in the winter, most in the summer.

It is harder to see if there is an overall upward or downard trend. To investigate trend, let's look at the rolling mean across a year.


```python
# Plot the rolling mean with a window of a year
```


```python
train.rolling(52).mean().plot()
```




    <AxesSubplot:xlabel='START TIME'>




    
![png](index_files/index_18_1.png)
    


That clearly visualizes an upward trend in our data, especially towards the first half of our dataset.

One way to remove trend this trend from the data is differencing.  A first order difference with a period of 1 (the default), will leave us with a timeseries composed of the change of rentals from 1 week to the next.


```python
# Apply a 1st order difference to the time series and plot the rolling mean
```


```python
train.diff().rolling(52).mean().plot()

```




    <AxesSubplot:xlabel='START TIME'>




    
![png](index_files/index_22_1.png)
    


The trend is much less pronounced, but still looks to be sloping downwards. Let's see if a second order difference stabilizes it further.


```python
# Apply a 2nd order difference: i.e. difference the difference, and plot the rolling mean.
```


```python
train_week.diff().diff().rolling(52).mean().plot()

```




    <AxesSubplot:xlabel='START TIME'>




    
![png](index_files/index_25_1.png)
    


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
sm = SARIMAX(train, order=[0,1,0]).fit()

```


```python
# Create a set of predictions for the train data (just use predict)
```


```python
sm.predict()
```




    START TIME
    2015-04-19      0.0
    2015-04-26     24.0
    2015-05-03    106.0
    2015-05-10    182.0
    2015-05-17    160.0
                  ...  
    2018-12-09    168.0
    2018-12-16    229.0
    2018-12-23    217.0
    2018-12-30    242.0
    2019-01-06     76.0
    Freq: W-SUN, Name: predicted_mean, Length: 195, dtype: float64




```python
# Calculate the training root mean squared error (same syntax as a linear regression prediction)
```


```python
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(train, sm.predict(), squared=False)
rmse
```




    59.9824333258689



Next, calculate the test rmse.  To do this, you have to pass in the date of the first test element and the last test element as arguments.


```python
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(test, sm.predict(test.index[0], test.index[-1]), squared=False)
rmse
```




    218.70915003193696



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
model = print_ts_metrics(train, test, [1,1,1])
plot_predictions(test, model)
```

    57.415975890558435
    221.23405768411936



    
![png](index_files/index_42_1.png)
    



```python
model = print_ts_metrics(train, test, [2,1,2])
plot_predictions(test, model)

```

    55.17796571359788
    257.77450434757844



    
![png](index_files/index_43_1.png)
    


The models make predictions, and different non-seasonal orders affect the rmse on the test set.  But the predictions certainly leave something to be desired.

From our eda above, we know that the data has seasonality.  We can account for this using the seasonal_order parameter.  The first step in generating better predictions is to choose the correct period.  The seasonal_order list should have for elements: P,D,Q,period (in that order).

Take the best parameters from above, then add to it a 1st order seasonal difference that makes sense with the eda.


```python
model = print_ts_metrics( train, test, [2,1,1],[0,1,0,52])
plot_predictions(test, model)

```

    64.14226324609506
    59.16560214343101



    
![png](index_files/index_46_1.png)
    

