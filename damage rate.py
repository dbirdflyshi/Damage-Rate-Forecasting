###############################################################################
###############################################################################
##                                                                           ##
##                       ~-.Damage Rate Predictor.-~                         ##
##                             -Dane Anderson-                               ##
##                                                                           ##
###############################################################################
###############################################################################
# Purpose: Creates a forecast for damage rate

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import sklearn.metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from datetime import timedelta

###############################################################################
##                          Importing The Data                               ##
###############################################################################
# Damage Rate Actuals from the previous X Years - Damage Rate App in Blueview
#drhistoricAll = pd.read_csv('DRActualsAll.csv')
drhistoricAll = pd.read_csv('DamageRateActuals.csv')
drhistoricNetwork = pd.read_csv('DamageRateActualsNetwork.csv')
#drhistoricNetwork = pd.read_csv('DRActualsNetwork.csv')
# Last 3 years of inflow actuals on a weekly level
inflows = pd.read_csv('InflowsActuals.csv')
# Next Months of Inflow Forecast Monthly
inflowsfcst = pd.read_csv('AllInflowsFcst.csv')
###############################################################################
##                             Preprocessing                                 ##
###############################################################################
calendarWeekly = pd.read_csv('CalendarFull.csv',encoding = 'ISO-8859-1')
calendarWeekly = calendarWeekly[['Calendar WS Date','Month Short Name','Week','Year','Month']]
calendarWeekly.columns = ['Date','CM','CW','CY','CM2']
calendarWeekly = calendarWeekly.assign(Date = pd.to_datetime(calendarWeekly.Date, format='%m/%d/%Y')  )
calendarWeekly = calendarWeekly.set_index(calendarWeekly['Date']).sort_index(ascending = False)
calendarDaily = calendarWeekly.resample('D').ffill()
calendarDaily = calendarDaily.assign(Date = calendarDaily.index)
calendarMonthly = calendarWeekly.sort_index(ascending = True)
calendarMonthly = calendarMonthly.groupby(['CM','CY']).agg({'Date':'first','CM':'first','CY':'first'})
calendarMonthly = calendarMonthly.set_index(calendarMonthly['Date']).sort_index(ascending = False)
calendarMonthly.columns = ['Date','CM','CY']

#######################################
#  Modifying The Inflows Historicals  #
#######################################
## Renaming, reorganizing, sorting, reindexing
inflows.columns = ['Date','Bucket','Inflows']
inflows = inflows.assign(Date = pd.to_datetime(inflows.Date, format='%m/%d/%Y'))
inflows.index = inflows.Date
inflowsNetwork = inflows.groupby(inflows.index).agg({'Date':'first','Inflows':sum})   
inflowsNetwork = inflowsNetwork.assign(Bucket = 5)
inflowsNetwork = inflowsNetwork.assign(CW = calendarDaily.CW)
inflowsNetwork = inflowsNetwork.assign(CY = calendarDaily.CY)
inflowsNetwork = inflowsNetwork.assign(CM = calendarDaily.CM)
inflowsNetwork = inflowsNetwork.drop('Date',axis = 1)
inflows = inflows.assign(CW = calendarDaily.CW)
inflows = inflows.assign(CY = calendarDaily.CY)
inflows = inflows.assign(CM = calendarDaily.CM)
inflows = inflows.drop('Date', axis = 1)
inflows = pd.concat([inflows,inflowsNetwork], axis = 0, join = 'outer', join_axes = None, ignore_index = False, sort = True)
inflows = inflows[['CW','CM','CY','Bucket','Inflows']]

#######################################
#   Modifying The Inflows Forecasts   #
#######################################
inflowsfcst['CY'] = 2019
inflowsfcst['Fcst'] = inflowsfcst['Fcst'].astype(int)
inflowsfcst = pd.merge(inflowsfcst, calendarMonthly, on = ['CY','CM'], how = 'left')
inflowsfcst = inflowsfcst.set_index(inflowsfcst['Date']).sort_index(ascending = False)
inflowsfcst = inflowsfcst[['CY','CM','Bucket','Fcst']]
inflowsFcstNetwork = inflowsfcst.groupby(inflowsfcst.index).agg({'CY':'first','CM':'first','Fcst':sum})   
inflowsFcstNetwork = inflowsFcstNetwork.assign(Bucket = 5)
inflowsfcst = pd.concat([inflowsfcst,inflowsFcstNetwork], axis = 0, join = 'outer', join_axes = None, ignore_index = False, sort = True)
inflowsfcst = inflowsfcst[['CY','CM','Bucket','Fcst']]

#######################################
#  Modifying the Damage Rate Actuals  #
#######################################
drhistoricAll = drhistoricAll.assign(Date = pd.to_datetime(drhistoricAll.CalendarDay, format='%m/%d/%Y'))
drhistoricAll.index = drhistoricAll.Date
drhistoricAll = drhistoricAll.assign(CW = calendarDaily.CW)
drhistoricAll = drhistoricAll.assign(CY = calendarDaily.CY)
drhistoricAll = drhistoricAll[['CY','CW','Date','Bucket','DamageRate']]
drhistoricAll.columns = ['CY','CW','Date','Bucket','Rate']
# In order to combine the buckets to the network bucket, they have to be the same datatype
drhistoricAll.Rate = drhistoricAll.Rate.astype(float)

# Modifying the network level damage rates, same as above
drhistoricNetwork = drhistoricNetwork.assign(Date = pd.to_datetime(drhistoricNetwork.CalendarDay, format='%m/%d/%Y'))
drhistoricNetwork.index = drhistoricNetwork.Date
drhistoricNetwork = drhistoricNetwork.assign(CW = calendarDaily.CW)
drhistoricNetwork = drhistoricNetwork.assign(CY = calendarDaily.CY)
drhistoricNetwork = drhistoricNetwork[['CY','CW','Date','Bucket','DamageRate']]
drhistoricNetwork.columns = ['CY','CW','Date','Bucket','Rate']
drhistoricNetwork.Rate = drhistoricNetwork.Rate.astype(float)
#Combines network to the other buckets
DRHistoric = pd.concat([drhistoricNetwork,drhistoricAll], axis = 0, join = 'outer', join_axes = None, ignore_index = False, sort = True)
DRHistoric = DRHistoric[['CY','CW','Bucket','Rate']]

buckets = DRHistoric.Bucket.unique()
AllBucketsDRFcst = pd.DataFrame([])
for bucket in buckets:
    BucketDRActuals = DRHistoric.loc[DRHistoric.Bucket == bucket]
    BucketInflowsActuals = inflows.loc[inflows.Bucket == bucket].sort_index()
    # I can't seem to get any kind of join to correctly work, so i lazily made this
    BucketDRActuals = BucketDRActuals.assign(Inflows = BucketInflowsActuals.Inflows)
    # Both dataframes arent the same length, this works kind of as an inner join function
    BucketDRActuals = BucketDRActuals.dropna(how='any')
    # Reorganize and remove bucket name for model testing
    BucketDRActuals = BucketDRActuals[['CY','CW','Inflows','Rate']]
    BucketDRActuals = BucketDRActuals.sort_index(ascending = True)
    ###############################################################################
    ##                               Model Testing                               ##
    ###############################################################################
    # Inflows and date fields
    X = BucketDRActuals.iloc[:,:3].values
    # Damage rate only as prediction answers
    y = BucketDRActuals.iloc[:,-1].values
    
    # Split the data, X_test gets the last 20% of the data, X_train gets the first 80%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
    
    regressor= XGBRegressor()
    
#    #Building the grid search                     
#    parameters = {'n_estimators' : [5, 10, 50, 100],
#                  'learning_rate' : [.1, .5, 1],
#                  'max_depth' : [1, 5, 10, 20, 30, 40],
#                  }
#    grid_search = GridSearchCV(estimator = regressor, 
#                               param_grid = parameters,
#                               scoring = 'r2',
#                               cv = 10)
#    grid_search = grid_search.fit(X_train, y_train)
#    best_parameters = grid_search.best_params_
#    best_accuracy = grid_search.best_score_
    
    regressor= XGBRegressor(n_estimators = 10, max_depth = 20, learning_rate = 0.5)
    regressor.fit(X_train,y_train)
    
    y_pred = regressor.predict(X_test)       

    ###############################################################################
    ##               Converting Monthly Inflow forecast to weekly :              ##
    ##    This process uses the week of the year average weekly actual ratio     ##
    ###############################################################################
    BucketInflowsFcst = inflowsfcst.loc[inflowsfcst.Bucket == bucket].sort_index()
    BucketInflowsFcst = BucketInflowsFcst.drop('Bucket', axis = 1)
    ## Expands the data from monthly to weekly
    ## -- Weekly is just the same as monthly for now
    BucketInflowsFcst = BucketInflowsFcst.resample('W').ffill()
    ## When using resample, it places rows in between the dates based on the selection
    ## This does not turn the final month of the forecast to weekly 
    ## -- because it doesnt have a reference
    ## We have to manually make the week
    ## -- * grab the final week from source file and make own object
    ## -- * drop the final week we grabbed in the source
    ## -- * make the date as index in the new object
    ## -- * manually make each date field by doing a date = date+1
    ## -- * join the final month data and forward fill the rest to the other days
    finalweek = BucketInflowsFcst[-1:]
    BucketInflowsFcst = BucketInflowsFcst.drop(BucketInflowsFcst.index[len(BucketInflowsFcst)-1])
    finalweekfcst = pd.DataFrame([])           
    finalweek = finalweek.assign(Date = finalweek.index)
    a = pd.to_datetime(finalweek.iloc[:,-1].values)
    # When making the range of dates, it doesnt include the current day, so you have to go back one day
    a += timedelta(days=-7)
    for i in range(4): 
        a += timedelta(days=7)
        b = pd.DataFrame([a])
        finalweekfcst = pd.concat([finalweekfcst, b],axis = 0, join = 'outer', join_axes = None, ignore_index = False, sort = True)            
    finalweekfcst = finalweekfcst.set_index(finalweekfcst[0])
    finalweekfcst = finalweekfcst.assign(Fcst = finalweek['Fcst'])
    finalweekfcst = finalweekfcst.assign(CY = finalweek['CY'])
    finalweekfcst = finalweekfcst.ffill()
    finalweekfcst = finalweekfcst[['CY','Fcst']] 
    BucketInflowsFcst = pd.concat([BucketInflowsFcst, finalweekfcst],axis = 0, join = 'outer', join_axes = None, ignore_index = False, sort = True)
    
    ## When forward filling and resampling, weeks forward fill so we dont have an accurate week number
    ## -- We have to make our own manual week list using length of the forecast and the first week
    ## -- join the weeks table to the forecast and cut the old one and rename like the old one
    # weeks = pd.DataFrame([list(range(int(BucketInflowFcst.iloc[0:1,1].values),int(BucketInflowFcst.iloc[0:1,1].values+len(BucketInflowFcst))))])
    ### We need to adjust this for future iterations of the code, not all years have same week length of year
    weeks = pd.DataFrame([list(range(1,53-len(BucketInflowsFcst)))])
    weeks = weeks.transpose()
    BucketInflowsFcst=BucketInflowsFcst.assign(Date = BucketInflowsFcst.index) 
    BucketInflowsFcst = BucketInflowsFcst.assign(CM = calendarDaily.CM)
    BucketInflowsFcst = BucketInflowsFcst.reset_index(drop = True)
    BucketInflowsFcst = BucketInflowsFcst.join(weeks)
    BucketInflowsFcst = BucketInflowsFcst.set_index(BucketInflowsFcst['Date']).sort_index(ascending = True)
    BucketInflowsFcst = BucketInflowsFcst[['CY',0,'CM','Fcst']]    
    BucketInflowsFcst.columns = ['CY','CW','CM','Forecast']

    #######################################
    #      Creating the weekly ratio      #
    #######################################
    ## Reformat the date to a date object and set it as the index
    ## Apply a month ID so it knows what to consider a new month
    BucketInflowsActuals = BucketInflowsActuals.assign(Date = BucketInflowsActuals.index)
    BucketInflowsActuals = BucketInflowsActuals.assign(CM2 = calendarDaily.CM2)
    BucketInflowsActuals = BucketInflowsActuals.drop('Date', axis = 1)
    ## Collect the years to iterate through
    ## Iterate through the years and the months of each year
    ## -- calculates the ratio by taking each week and dividing it by the sum of that month
    ## -- in case one week has no data, instead of an NA, it's 0
    years = BucketInflowsActuals.loc[:,'CY'].unique()            
    allratios= pd.DataFrame([])
    monthlyratios= pd.DataFrame([])
    for year in years:
        # To get into the weeks per year, we have to make a list of unique weeks as a list to iterate through
        months = np.unique(BucketInflowsActuals.loc[BucketInflowsActuals.loc[:,'CY']==year,'CM2'].values)
        #Iterating through the weeks to get the weekly ratios (This loop takes up 2secs of the 2.3 g/s process)
        for month in months:
            # Searches through the historical data to get the specific week and specific year
            monthly = BucketInflowsActuals.loc[(BucketInflowsActuals.loc[:,'CM2']==month) & (BucketInflowsActuals.loc[:,'CY']==year)]
            monthly = monthly.reset_index()
            monthly = monthly.groupby(monthly.CW).agg({'Date':'first','CW':'first','CY':'first','CM':'first','Bucket':'first','Inflows':'sum','CM2':'first'})
            # Calculates the sum of all the week's volume to obtain the ratio                  
            monthlyvolumetotal = monthly.loc[:,'Inflows'].sum()             
            # Calculates the ratio for the weekly breakout
            ratios = monthly.loc[:,'Inflows'] / monthlyvolumetotal              
            # Putting all the weeks together
            monthly= monthly.assign(Ratios= ratios)
            # Some days don't have ratios, this puts them to zero
            monthly = monthly.fillna(value=0)  
            # Appends the current week to the previous week
            monthlyratios = pd.concat([monthlyratios, monthly],axis = 0, join = 'outer', join_axes = None, ignore_index = False, sort = True)             
    # I am too lazy to figure out how to logically remove weeks without full months tied to them 
    # - So i decided to just drop all entries with a ratio of greater than .45
    # - this should work becuase no real ratio will be more than this, but we will see *shrug*
    monthlyratios = monthlyratios.reset_index(drop = True)
    monthlyratios = monthlyratios.drop(monthlyratios[monthlyratios['Ratios'].map(lambda x: x >0.32)].index)
    monthlyratios = monthlyratios.groupby(monthlyratios.CW).agg({'CW':'first','Ratios':'mean'})   
    #monthlyratios = monthlyratios.drop('FW', axis = 1)
    #######################################
    #      Applying the weekly ratio      #
    #######################################
    ## Extracting just a list of ratios
    ## joining this list to the forecast of inflows table
    ## Multiplying the monthly forecast column to the ratio to obtain the weekly
    ## Removing extraneous columns 
    BucketInflowsFcst = pd.merge(BucketInflowsFcst, monthlyratios, on = ['CW'], how = 'left')
    BucketInflowsFcst = BucketInflowsFcst.assign(Forecast = BucketInflowsFcst.Forecast * BucketInflowsFcst.Ratios)
#    ratios = monthlyratios.values
    BucketInflowsFcst = BucketInflowsFcst.drop(['Ratios','CM'], axis = 1)
    ###############################################################################
    ##                  Applying the model to create forecast :                  ##
    ###############################################################################
    ## Strips the inflows and date fields of just numbers
    ## Applies the trained model to the stripped table
    ## Appending the forecasted damage rate to the date columns
    ## Dropping the weekly inflow forecast
    BucketInflowFcstNoLbls = BucketInflowsFcst.iloc[:,:3].values
    drfcst = regressor.predict(BucketInflowFcstNoLbls)
    BucketInflowsFcst['Damage Rate'] = drfcst.astype(float)
    BucketDRFcst = BucketInflowsFcst.drop('Forecast', axis = 1)
    BucketDRFcst = BucketDRFcst.assign(Bucket = bucket)
    BucketDRFcst = pd.merge(BucketDRFcst, calendarWeekly, on = ['CY','CW'], how = 'left')
    BucketDRFcst = BucketDRFcst.set_index(BucketDRFcst['Date']).sort_index(ascending = False)
    BucketDRFcst = BucketDRFcst[['Bucket','Damage Rate']]
    AllBucketsDRFcst = pd.concat([AllBucketsDRFcst, BucketDRFcst],axis = 0, join = 'outer', join_axes = None, ignore_index = False, sort = True)             
## Shipping it out as a csv file
AllBucketsDRFcst = AllBucketsDRFcst.sort_index(ascending = True)
AllBucketsDRFcst.to_csv(path_or_buf = "rate.csv", index = True, float_format = "%4f", index_label = None)