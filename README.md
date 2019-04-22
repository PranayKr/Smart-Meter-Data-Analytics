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

a new column is then created that contains the remainder of the sub-metering, using the calculation from the previous section.

We can now save the cleaned-up version of the dataset to a new ﬁle; in this case we will just change the ﬁle extension to .csv and save the dataset as household power consumption.csv.











