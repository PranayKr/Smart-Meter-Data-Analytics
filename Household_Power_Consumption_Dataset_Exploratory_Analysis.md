# Household_Power_Consumption_Dataset (Exploratory_Analysis)
# Patterns in Observations Over Time
The data is a multivariate time series and the best way to understand a time series is to create line plots. We can start oﬀ by creating a 
separate line plot for each of the eight variables

A single image is created with eight subplots, one for each variable. This gives us a really high level of the four years of one minute 
observations. In case of Sub metering 3 (environmental control) that may not directly map to hot or cold years. Perhaps new systems were 
installed.  
The contribution of sub metering 4 seems to decrease with time, or show a downward trend, perhaps matching up with the solid increase in 
seen towards the end of the series for Sub metering 3. These observations do reinforce the need to honor the temporal ordering of 
subsequences of this data when ﬁtting and evaluating any model. We might be able to see the wave of a seasonal eﬀect in the Global active 
power and some other variates. There is some spiky usage that may match up with a speciﬁc period, such as weekends.



