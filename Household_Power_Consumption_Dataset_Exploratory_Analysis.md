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
# Global Active Power Usage for each day of the months of November and May of the year 2009

![activepower-Nov2009](https://user-images.githubusercontent.com/25223180/56571170-00ab1a00-65da-11e9-8889-fe305ee94aba.PNG)

A single image is created with thirty line plots, one for each month of  November 2009

![activepower-May2009](https://user-images.githubusercontent.com/25223180/56571181-04d73780-65da-11e9-9532-6acf7ced5676.PNG)

A single image is created with thirty-one line plots, one for each month of  May 2009
# Inferences Drawn
1) There is common trend of energy consumption across the days.
2) For many days consumption starts early morning, around 6-7AM. 
3) Some days show a drop in consumption in the middle of the day, which might make sense if most occupants are out of the house. 
4) Some strong overnight consumption is observed on some days, that in a northern hemisphere November may match up with a heating system 
   being used. 
5) Time of year, specically the season and the weather that it brings, will be an important factor in modeling this data
# Time Series Data Distributions
Distribution of data across all the variables can be investigated using Histogram plots 
# Histogram generated for each variable in time-series

![hsitogram_plot_eachvariable](https://user-images.githubusercontent.com/25223180/56573346-7e712480-65de-11e9-8c99-7860389de088.PNG)

A single plot is generated with separate histogram for each of the 8 variables 
# Inferences Drawn
1) Active and Reactive power, Intensity, as well as the Sub-metered Power are all skewed distributions down towards small watt-hour or 
   kilowatt values.
2) Distribution of voltage data is strongly Gaussian.
3) The distribution of active power appears to be bi-modal, meaning it looks like it has two mean groups of observations.
# The distribution of active power consumption for the four full years of data using Histograms

![hsitogram_plot_eachyear](https://user-images.githubusercontent.com/25223180/56573348-80d37e80-65de-11e9-96e2-4ab0ccf18c8b.PNG)

A single plot with 4 figures is created , one for each year between 2007 and 2010.
# Inferences Drawn
1) The distribution of active power consumption across those years looks very similar
2) The distribution is bimodal with one peak around 0.3 KW and perhaps another around 1.3 KW.
3) There is a long tail on the distribution to higher kilowatt values.
4) It might open the door to notions of discretizing the data and separating it into peak1, peak 2 or long tail.
5) These groups or clusters for usage on a day or hour may be helpful in developing a predictive model.
6) It is possible that the identied groups may vary over the seasons of the year.
# The distribution for active power for each month in year 2009 using Histograms

![hsitogram_plot_2009months](https://user-images.githubusercontent.com/25223180/56573352-83ce6f00-65de-11e9-8d57-4d26966d11a0.PNG)

12 plots are created , one for each month of 2009.
# Inferences Drawn
1) Generally the same data distribution is observed for each month.
2) The axes for the plots appear to align (given the similar scales), and the peaks are shifted down in the warmer 
   northern hemisphere months and shifted up for the colder months.
3) A thicker or more prominent tail toward larger kilowatt values for the cooler months of December through to March is also observed.
   
