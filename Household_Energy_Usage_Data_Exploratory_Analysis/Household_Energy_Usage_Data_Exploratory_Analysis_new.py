#!/usr/bin/env python
# coding: utf-8

# In[8]:


# load and clean-up power usage data
from numpy import nan
from pandas import read_csv
from matplotlib import pyplot
# load all data
dataset = read_csv('smart meter dataset\household_power_consumption\household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
# summarize
print(dataset.shape)
print(dataset.head())
# mark all missing values
dataset.replace('?', nan, inplace=True)
# add a column for for the remainder of sub metering
values = dataset.values.astype('float32')
dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
# save updated dataset
dataset.to_csv('smart meter dataset\household_power_consumption\household_power_consumption.csv')
# load the new dataset and summarize
dataset = read_csv('smart meter dataset\household_power_consumption\household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(dataset.head())


# In[58]:


# line plot for each variable
pyplot.figure()
fig, axies = pyplot.subplots(nrows=len(dataset.columns), ncols=1,figsize=(7,7), squeeze=False)
fig.tight_layout()
fig.suptitle('line plot for each variable', fontsize=20, y=1.1)
for i in range(len(dataset.columns)):
    name = dataset.columns[i]
    axies[i][0].plot(dataset[name])
    axies[i][0].set_title(name, y=0)
    axies[i][0].set_yticks([])
    axies[i][0].set_xticks([])
    
pyplot.show()


# In[56]:


# plot active power for each year
years = ['2007', '2008', '2009', '2010']
pyplot.figure()
fig, axies = pyplot.subplots(nrows=4, ncols=1,figsize=(7,7), squeeze=False)
fig.tight_layout()
fig.suptitle('active power for each year in the dataset', fontsize=20, y=1.1)
for i in range(len(years)):
    year = years[i]
    # get all observations for the year
    result = dataset[str(year)]
    
    axies[i][0].plot(result['Global_active_power'])
    axies[i][0].set_title(str(year), y=0, loc='left')
    axies[i][0].set_yticks([])
    axies[i][0].set_xticks([])
    
    
pyplot.show()


# In[50]:


# plot active power for each month of 2009
months = [x for x in range(1, 13)]
pyplot.figure()
fig, axies = pyplot.subplots(nrows=12, ncols=1,figsize=(9,10), squeeze=False)
fig.tight_layout()
fig.suptitle('active power for each month of 2009', fontsize=20, y=1.1)

for i in range(len(months)):
    month = '2009-' + str(months[i]) 
    result = dataset[month]  
    axies[i][0].plot(result['Global_active_power'])
    axies[i][0].set_title(str(month), y=0, loc='left')
    axies[i][0].set_yticks([])
    axies[i][0].set_xticks([])
    
pyplot.show()


# In[44]:


# plot active power for each day of the month of November of 2009
days = [x for x in range(1, 31)]
pyplot.figure()

fig, axies = pyplot.subplots(nrows=30, ncols=1,figsize=(5,7), squeeze=False)

fig.tight_layout()
fig.suptitle('active power for each day of November of 2009', fontsize=20, y=1.1)

for i in range(len(days)):
    day = '2009-11-' + str(days[i])
    result = dataset[day]
    axies[i][0].plot(result['Global_active_power'])
    axies[i][0].set_title(day, y=0, loc='left', size=6)
    axies[i][0].set_yticks([])
    axies[i][0].set_xticks([])    
    axies[i][0].invert_yaxis()
    
pyplot.show()


# In[43]:


# plot active power for each day of the month of May of 2009
days = [x for x in range(1, 32)]
pyplot.figure()
fig, axies = pyplot.subplots(nrows=31, ncols=1,figsize=(5,7), squeeze=False)
fig.tight_layout()
fig.suptitle('active power for each day of May of 2009', fontsize=20, y=1.1)

for i in range(len(days)):
    day = '2009-05-' + str(days[i])
    result = dataset[day]
    axies[i][0].plot(result['Global_active_power'])
    axies[i][0].set_title(day, y=0, loc='left', size=6)
    axies[i][0].set_yticks([])
    axies[i][0].set_xticks([])  
    axies[i][0].invert_yaxis()
    
pyplot.show()


# In[78]:


# histogram plot for each variable
pyplot.figure().suptitle('histogram plot for each variable', fontsize=15, y=1.1)
#fig, axies = pyplot.subplots(nrows=len(dataset.columns), ncols=1,figsize=(7,7), squeeze=False)
#fig.tight_layout()
#fig.suptitle('histogram plot for each variable', fontsize=20, y=1.1)

for i in range(len(dataset.columns)):
    pyplot.subplot(len(dataset.columns), 1, i+1)
    #name = dataset.columns[i]
    # create histogram
    dataset[dataset.columns[i]].hist(bins=100)
    pyplot.title(name, y=0, loc='right')
    pyplot.yticks([])
    pyplot.xticks([])
   
    #axies[i][0].set_title(dataset.columns[i], y=0, loc='right')
    #axies[i][0].set_yticks([])
    #axies[i][0].set_xticks([])
    
pyplot.show()


# In[82]:


#active power for each year
years = ['2007', '2008', '2009', '2010']
pyplot.figure().suptitle('yearly histogram plots for power usage', fontsize=15, y=1.1)
for i in range(len(years)):
    ax = pyplot.subplot(len(years), 1, i+1)
    year = years[i]
    result = dataset[str(year)]
    result['Global_active_power'].hist(bins=100)
    ax.set_xlim(0, 5)
    pyplot.title(str(year), y=0, loc='right')
    pyplot.yticks([])
    pyplot.xticks([])
pyplot.show()


# In[83]:


# plot active power for each month of year 2009
months = [x for x in range(1, 13)]
pyplot.figure().suptitle('monthly histogram plots for power usage for each month of year 2009', fontsize=15, y=1.1)
for i in range(len(months)):
    ax = pyplot.subplot(len(months), 1, i+1)
    month = '2009-' + str(months[i])
    result = dataset[month]
    result['Global_active_power'].hist(bins=100)
    ax.set_xlim(0, 5)
    pyplot.title(month, y=0, loc='right')
    pyplot.yticks([])
    pyplot.xticks([])
pyplot.show()


# In[ ]:




