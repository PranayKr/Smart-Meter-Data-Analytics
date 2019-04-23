# Household_Power_Consumption_Dataset (Exploratory_Analysis)
# Patterns in Observations Over Time
The data is a multivariate time series and the best way to understand a time series is to create line plots. We can start oﬀ by creating a 
separate line plot for each of the eight variables

![lineplot_eachvar](https://user-images.githubusercontent.com/25223180/56534139-afb60a00-6576-11e9-9a8c-caa14ba86322.PNG)

A single image is created with eight subplots, one for each variable. 
# Inferences Drawn
1) This gives us a really high level of the four years of one minute observations. 
2) In case of Sub metering 3 (environmental control) energy usage may not directly map to hot or cold years. Perhaps new systems were 
   installed.
3) The contribution of sub metering 4 seems to decrease with time, or show a downward trend, perhaps matching up with the solid increase 
   is seen towards the end of the series for Sub metering 3. 
   These observations do reinforce the need to honor the temporal ordering of subsequences of this data when ﬁtting and evaluating any 
   model. 
4) The wave of a seasonal eﬀect can be seen in the Global active power and some other variates. There is some spiky usage that 
   may match up with a speciﬁc period, such as weekends.


# Global Active Power Usage across the years in the Dataset
(The ﬁrst year, 2006, has less than one month of data, so will remove it from the plot. )

![activepower_years](https://user-images.githubusercontent.com/25223180/56534780-f6f0ca80-6577-11e9-864d-fa212cd0eb60.PNG)

A single image is created with four line plots, one for each full year (or mostly full years) of data in the dataset. 
# Inferences Drawn
1) Some common gross patterns can be seen across the years, such as around Feb-Mar and around Aug-Sept where we see a marked decrease in 
   consumption.
2) A downward trend is seen over the summer months (middle of the year in the northern hemisphere) and perhaps more 
   consumption in the winter months towards the edges of the plots. 
3) These may show an annual seasonal pattern in consumption. 
4) A few patches of missing data can be seen in at least the ﬁrst, third, and fourth plots.
# Global Active Power Usage for each month of 2009
To know the daily and weekly patterns of pwer usage per month graph of energy consumption is plotted
![activepower-2009months_1](https://user-images.githubusercontent.com/25223180/56569650-05220380-65d7-11e9-8b75-235b555d9d01.PNG)
![activepower-2009months_2](https://user-images.githubusercontent.com/25223180/56569658-094e2100-65d7-11e9-949a-63faeab0a89c.PNG)
A single image is created with twelve line plots, one for each month of 2009
# Inferences Drawn
1) The sign-wave of power consumption of the days within each month can be observed. This is understood as some kind of daily 
   pattern in power consumption can be expected daily.
2) There are stretches of days with very minimal consumption, such as in August and in April. These may represent 
   vacation periods where the home was unoccupied and where power consumption was minimal.



