# Smart-Meter-Data-Analytics
A Predictive Analytics Problem Statement to forecast future Electricity Consumption (Active Power) using Household Power Consumption 
Dataset
# Household Power Consumption Dataset
The Household Power Consumption dataset is a multivariate time series dataset of power-related variables that describes the electricity 
consumption for a single household over four years. The data was collected between December 2006 and November 2010 and observations of 
power consumption within the household were collected every minute. It is a multivariate series comprised of seven variables (besides 
the date and time); they are:
1) global active power: The total active power consumed by the household (kilowatts).
2) global reactive power: The total reactive power consumed by the household (kilowatts).
3) voltage: Average voltage (volts).
4) global intensity: Average current intensity (amps).
5) sub metering 1: Active energy for kitchen (watt-hours of active energy).
6) sub metering 2: Active energy for laundry (watt-hours of active energy).
7) sub metering 3: Active energy for climate control systems (watt-hours of active energy).

Active and reactive energy refer to the technical details of alternative current. In general terms, the active energy is the real power 
consumed by the household, whereas the reactive energy is the unused power in the lines. We can see that the dataset provides the active 
power as well as some division of the active power by main circuit in the house, speciﬁcally the kitchen, laundry, and climate control. 
These are not all the circuits in the household. The remaining watt-hours can be calculated from the active energy by ﬁrst converting the 
active energy to watt-hours then subtracting the other sub-metered active energy in watt-hours, as follows:

#### remainder = (global act pwr×1000 60)−(sub met 1 + sub met 2 + sub met 3) 

The dataset is described and has been made freely available on the UCI Machine Learning repository1. The dataset can be downloaded as a 
single 20 megabyte zip ﬁle. A direct download link is provided blow:
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

The data columns are separated by semicolons (‘;’). The data is reported to have one row for each day in the time period. The data does 
have missing values marked as '?' in the dataset

The Data File can be loaded as a Pandas DataFrame and summarize the loaded data. We can use the read csv() function to load 
the data with a few customizations:
1) Specify the separate between columns as a semicolon (sep=‘;’)
2) Specify that line 0 has the names for the columns (header=0)
3) Specify that we have lots of RAM to avoid a warning that we are loading the data as an array of objects instead of an array of 
   numbers, because of the ‘?’ values for missing data (low memory=False).
4) Specify that it is okay for Pandas to try to infer the date-time format when parsing dates, which is way faster 
   (infer datetime format=True)
5) Specify that we would like to parse the date and time columns together as a new column called ’datetime’ 
   (parse dates=‘datetime’:[0,1])
6) Specify that we would like our new datetime column to be the index for the DataFrame (index col=[’datetime’]).

Putting all of this together, the data can be loaded and summarize the loaded shape and first few rows.

![Load_Data_SmartMeter](https://user-images.githubusercontent.com/25223180/56530953-db35f600-6570-11e9-8427-c8732df05575.PNG)

all missing values indicated with a ‘?’ character are marked with a NaN value, which is a ﬂoat. This will allow us to work with the data as one array of ﬂoating point values rather than mixed types, which is less eﬃcient.

![missing_values](https://user-images.githubusercontent.com/25223180/56531453-b2fac700-6571-11e9-930a-b63903da25d3.PNG)

a new column is then created that contains the remainder of the sub-metering, using the calculation from the previous section.

![add_column](https://user-images.githubusercontent.com/25223180/56531586-edfcfa80-6571-11e9-92b7-e00db22cb73a.PNG)

We can now save the cleaned-up version of the dataset to a new ﬁle; in this case we will just change the ﬁle extension to .csv and save the dataset as household power consumption.csv.

![saveandload_updteddataset](https://user-images.githubusercontent.com/25223180/56531464-b8581180-6571-11e9-81ab-69dece914cda.PNG)
# Modelling the Dataset
The data is only for a single household, but perhaps eﬀective modeling approaches could be generalized across to similar households. 
Perhaps the most useful framing of the dataset is to forecast an interval of future active power consumption.
Four examples include: 
1) Forecast hourly consumption for the next day.
2) Forecast daily consumption for the next week.
3) Forecast daily consumption for the next month.
4) Forecast monthly consumption for the next year. 

These types of forecasting problems are referred to as multi-step forecasting. Models that make use of all of the variables might be 
referred to as a multivariate multi-step forecasting models. Each of these models is not limited to forecasting the minutely data, but 
instead could model the problem at or below the chosen forecast resolution. Forecasting consumption in turn, at scale, could aid in a 
utility company forecasting demand, which is a widely studied and important problem.
# Data Preparation
There is a lot of ﬂexibility in preparing this data for modeling. The speciﬁc data preparation methods and their beneﬁt really depend on 
the chosen framing of the problem and the modeling methods. Below is a list of general data preparation methods that may be useful: 
1) Daily diﬀerencing may be useful to adjust for the daily cycle in the data.
2) Annual diﬀerencing may be useful to adjust for any yearly cycle in the data.
3) Normalization may aid in reducing the variables with diﬀering units to the same scale. 
There are many simple human factors that may be helpful in engineering features from the data, that in turn may make speciﬁc days easier 
to forecast. Some examples include: 
1) Indicating the time of day, to account for the likelihood of people being home or not.
2) Indicating whether a day is a weekday or weekend.
3) Indicating whether a day is a North American public holiday or not. These factors may be signiﬁcantly less important for forecasting 
monthly data, and perhaps to a degree for weekly data. 
More general features may include: 
4) Indicating the season, which may lead to the type or amount environmental control systems being used.
# Modeling Methods
1. Naive Methods:
   Naive methods would include methods that make very simple, but often very eﬀective assumptions. Some examples include:
   1) Tomorrow will be the same as today.
   2) Tomorrow will be the same as this day last year.
   3) Tomorrow will be an average of the last few days.

2. Classical Linear Methods:
   Classical linear methods include techniques are very eﬀective for univariate time series forecasting. 
   Two important examples include:
   1) SARIMA.
   2) ETS (triple exponential smoothing).
   They would require that the additional variables be discarded and the parameters of the model be conﬁgured or tuned to the speciﬁc
   framing of the dataset. Concerns related to adjusting the data for daily and seasonal structures can also be supported directly
      
3. Machine Learning Methods:
   Machine learning methods require that the problem be framed as a supervised learning problem. This would require that lag 
   observations for a series be framed as input features, discarding the temporal relationship in the data. A suite of nonlinear and 
   ensemble methods could be explored, including:
   1) k-Nearest Neighbors.
   2) Support Vector Machines.
   3) Decision Trees.
   4) Random Forest.
   5) Gradient Boosting Machines.
   Careful attention is required to ensure that the ﬁtting and evaluation of these models preserved the temporal structure in the data. 
   This is important so that the method is not able to cheat by harnessing observations from the future. These methods are often 
   agnostic to large numbers of variables and may aid in teasing out whether the additional variables can be harnessed and add value to 
   predictive models.

4. Deep Learning Methods:
   Generally, neural networks have not proven very eﬀective at autoregression type problems. Nevertheless, techniques such as 
   convolutional neural networks are able to automatically learn complex features from raw data, including one-dimensional signal data. 
   And recurrent neural networks, such as the long short-term memory network, are capable of directly learning across multiple parallel 
   sequences of input data. Further, combinations of these methods, such as CNN-LSTM and ConvLSTM, have proven eﬀective on time series 
   classiﬁcation tasks. It is possible that these methods may be able to harness the large volume of minute-based data and multiple 
   input variables.
















