# Import packages and read data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import scipy.stats as stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import geopandas as gpd
# !pip install folium is necessary
import folium
from folium.plugins import HeatMap

df = pd.read_csv('kc_house_data.csv')
pd.set_option('display.max_columns', len(df.columns))

# Clean data
##Convert Date to integer days since beginning
time = pd.to_datetime(df.date)
df['days_since_first'] = [time[i]-time.min() for i in range(len(time))]
df['days_since_first'] = [int(str(i).split()[0]) for i in df.days_since_first]

##Convert sqft_basement to integer
df.sqft_basement.replace('?','0.0', inplace = True)
df.sqft_basement = [float(i) for i in df.sqft_basement]

##Replace all missing values with mode
df.waterfront.replace(np.nan, 0, inplace = True)
df.yr_renovated.replace(np.nan, 0, inplace = True)
df.view.replace(np.nan, 0, inplace = True)

##Categorize zipcodes (grouped into cities) - ultimately not used for my model
# seattle = [98178, 98125, 98136, 98198, 98146, 98115, 98107, 98126, 98103, 98133, 98119, 98112, 98117, 98166, 
#            98148, 98105, 98122, 98144, 98116, 98118, 98199, 98102, 98108, 98168, 98177, 98109, 98155, 98106, 98188]
# kenmore = [98028]
# sammamish = [98074, 98075]
# redmond = [98053, 98052]
# federal_way = [98003, 98023]
# maple_valley = [98038]
# bellevue = [98007, 98008, 98004, 98005, 98006]
# duvall = [98019]
# auburn = [98002, 98092, 98001]
# mercer_island = [98040]
# kent = [98030, 98042, 98032, 98031]
# issaquah = [98027, 98029]
# renton = [98058, 98056, 98059, 98055]
# vashon = [98070]
# kirkland = [98034, 98033]
# black_diamond = [98010]
# north_bend = [98045]
# woodinville = [98077, 98072]
# snoqualmie = [98065]
# enumclaw = [98022]
# fall_city = [98024]
# bothell = [98011]
# carnation = [98014]
# medina = [98039]

# cities = []
# for zips in df.zipcode: 
#     if zips in seattle: 
#         cities.append('Seattle')
#     elif zips in kenmore: 
#         cities.append('Kenmore')
#     elif zips in sammamish:
#         cities.append('Sammamish')
#     elif zips in redmond:
#         cities.append('Redmond')
#     elif zips in federal_way:
#         cities.append('Federal Way')
#     elif zips in maple_valley:
#         cities.append('Maple Valley')
#     elif zips in bellevue:
#         cities.append('Bellevue')
#     elif zips in  duvall:
#         cities.append('Duvall')
#     elif zips in auburn:
#         cities.append('Auburn')
#     elif zips in mercer_island:
#         cities.append('Mercer Island')
#     elif zips in kent:
#         cities.append('Kent')
#     elif zips in issaquah:
#         cities.append('Issaquah')
#     elif zips in renton:
#         cities.append('Renton')
#     elif zips in vashon:
#         cities.append('Vashon')
#     elif zips in kirkland:
#         cities.append('Kirkland')
#     elif zips in black_diamond:
#         cities.append('Black Diamond')
#     elif zips in north_bend: 
#         cities.append('North Bend')
#     elif zips in woodinville:
#         cities.append('Woodinville')
#     elif zips in snoqualmie:
#         cities.append('Snoqualmie')
#     elif zips in enumclaw:
#         cities.append('Enumclaw')
#     elif zips in fall_city:
#         cities.append('Fall City')
#     elif zips in bothell:
#         cities.append('Bothell')
#     elif zips in carnation:
#         cities.append('Carnation')
#     elif zips in medina:
#         cities.append('Medina')

# def add_cities(dataframe):
#     cities_series = pd.Series(cities)
#     cat_cities = cities_series.astype('category')
#     dataframe['cities'] = cities_series
#     from sklearn.preprocessing import LabelEncoder
#     label = LabelEncoder()
#     cities_encoded = label.fit_transform(cat_cities)
#     cities_dummies = pd.get_dummies(cat_cities)
#     dataframe = pd.concat([dataframe, cities_dummies], axis = 1)
#     return dataframe

# df_cities = add_cities(df)

# Remove outliers - if I had more time I would have written this as a much more readable and quicker lambda function!
def remove_outliers(dataframe):
    return dataframe[((dataframe.price > (dataframe.price.mean() - dataframe.price.std()*3)) 
        & (dataframe.price < (dataframe.price.mean() + dataframe.price.std()*3)))
      & ((dataframe.bedrooms > (dataframe.bedrooms.mean() - dataframe.bedrooms.std()*3)) 
         & (dataframe.bedrooms < (dataframe.bedrooms.mean() + dataframe.bedrooms.std()*3)))
       & ((dataframe.sqft_living > (dataframe.sqft_living.mean() - dataframe.sqft_living.std()*3)) 
          & (dataframe.sqft_living < (dataframe.sqft_living.mean() + dataframe.sqft_living.std()*3)))
      & ((dataframe.sqft_lot > (dataframe.sqft_lot.mean() - dataframe.sqft_lot.std()*3)) 
        & (dataframe.sqft_lot < (dataframe.sqft_lot.mean() + dataframe.sqft_lot.std()*3)))
      & ((dataframe.sqft_above > (dataframe.sqft_above.mean() - dataframe.sqft_above.std()*3)) 
        & (dataframe.sqft_above < (dataframe.sqft_above.mean() + dataframe.sqft_above.std()*3)))
      & ((dataframe.sqft_living15 > (dataframe.sqft_living15.mean() - dataframe.sqft_living15.std()*3)) 
        & (dataframe.sqft_living15 < (dataframe.sqft_living15.mean() + dataframe.sqft_living15.std()*3)))
      & ((dataframe.sqft_lot15 > (dataframe.sqft_lot15.mean() - dataframe.sqft_lot15.std()*3)) 
        & (dataframe.sqft_lot15 < (dataframe.sqft_lot15.mean() + dataframe.sqft_lot15.std()*3)))
      & ((dataframe.days_since_first > (dataframe.days_since_first.mean() - dataframe.days_since_first.std()*3)) 
        & (dataframe.days_since_first < (dataframe.days_since_first.mean() + dataframe.days_since_first.std()*3)))]

df_or = remove_outliers(df)

df_clean = df_or.drop(['id', 'date', 'zipcode'], axis = 1)

# visualize qqplots and linearity
        
def hist(dataframe):
    dataframe.hist(bins = 100, figsize = (20, 20))
    
        
def scatter(dataframe, dependent):
    sns.set_context('poster')
    fig, axes = plt.subplots(10,2, figsize = (40,200))
    plt.subplots_adjust(hspace = 0.4)
    for index, item in enumerate(dataframe):
        row = index//2
        col = index%2
        ax = axes[row][col]
        ax.scatter(dataframe[item], dataframe[dependent], color = np.random.rand(3,), marker = '.')
        ax.set_title(item)

#Test for correlation between variables and drop variables where correlation >0.75 (and id and date): 
def correlationcheck(dataframe):
    return abs(dataframe.corr()) > 0.75

copy = df_clean.copy()

# transform any variable to log
def logtransform(dataframe, columns):
    for column in columns:
        dataframe['log_' + str(column)] = np.log(dataframe[column])
    return dataframe

columns = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15', 'sqft_lot15']
logtransform(copy, columns)
clean_log = copy.drop(['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15', 'sqft_lot15'], axis = 1)


# Check models for linearity - you can use the scatter comparison to visualize changes in linearity

def formula(dataframe, interest):
    formulas = []
    for column in dataframe: 
        formulas.append(interest + '~' + column)
    return formulas
    
clean_formula = formula(df_clean, 'price')
log_formula = formula(clean_log, 'log_price')
# This can be used to find the qq plots for our cleaned data

def model(dataframe, variable, interest):
    formulas = formula(dataframe, interest)
    model = None
    for f in formulas: 
        if variable in f:
            model = (smf.ols(formula = f, data = dataframe).fit())
    return model

def resid(dataframe, variable, interest):
    resid = model(dataframe, variable, interest).resid
    return resid

def jbtest(dataframe, variable, interest):
    test_name = ['JB: ', 'p: ', 'skew: ', 'kurtosis: ']
    to_test = model(dataframe, variable, interest).resid
    jb = sms.jarque_bera(to_test)
    return list(zip(test_name, jb))

# Check for heteroscedasticity
def qqplots(dataframe, formulas, interest):
    for variable in formulas:
        sm.graphics.qqplot(resid(dataframe, variable, interest), dist=stats.norm, line='45', fit=True)
        plt.title(variable)
        plt.show()

# check heatmap to decide what we want to include and what we want to drop before starting our model

def correlationhm(dataframe):
    sns.set_context('poster', font_scale = 2.5)
    plt.figure(figsize = (100, 100))
    sns.heatmap(dataframe.corr(), annot = True)
    
def correlationcheck(dataframe):
    return abs(dataframe.corr()) > 0.75
    
model_df = clean_log.drop('log_sqft_above', axis = 1)

of_interest = clean_log.drop(['log_sqft_above', 'waterfront', 'condition', 'yr_built', 'yr_renovated', 'long', 'days_since_first', 'log_price', 'log_sqft_lot', 'log_sqft_lot15'], axis = 1)

# Here is our first model
dependent = 'log_price'
predictor_sum = "+".join(of_interest.columns)
formula = dependent + "~" + predictor_sum
model = ols(formula= formula, data=model_df).fit()
model.summary()

# Feature Selection
independent = model_df.drop('log_price', axis =1)
linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select = 7)
selector = selector.fit(independent, model_df['log_price'])

ranking = selector.ranking_
estimators = selector.estimator_
coef = estimators.coef_
intercept = estimators.intercept_

feature_selection = independent.drop(['bedrooms', 'bathrooms', 'floors', 'view', 'sqft_basement', 'yr_built', 'yr_renovated', 'days_since_first', 'log_sqft_lot', 'log_sqft_lot15'], axis =1)

# model summary for our new feature selections
dependent = 'log_price'
predictors = "+".join(feature_selection.columns)
formula = dependent + "~" + predictors
feature = ols(formula= formula, data=model_df).fit()
feature.summary()

# qq plot to see how our feature selection works
featureqq = sm.graphics.qqplot(feature.resid, dist=stats.norm, line='45', fit=True)


# Checking MSE for our feature selection - with lots of help from Learn!
x = feature_selection
y = model_df[['log_price']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
linreg = LinearRegression()
linreg.fit(x_train, y_train)
y_hat_train = linreg.predict(x_train)
y_hat_test = linreg.predict(x_test)
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
trainmse = 'train MSE: ' + str(train_mse)
testmse = 'test MSE: ' + str(test_mse)


# heatmap showing how price changes based on location - using to answer some questions from our data
# Special thanks to Michael Cunha from Alcid Analytics for this awesome feature (https://alcidanalytics.com/p/geographic-heatmap-in-python).

hmap = folium.Map(location=[47.5480, -121.9836], zoom_start=9)
mymap = model_df[['lat', 'long']]
myprice = df[['price']]
max1 = float(myprice['price'].max())
hm_wide = HeatMap(list(zip(mymap.lat.values, mymap.long.values, myprice.price.values)),
                   min_opacity=0.2,
                   max_val=max1,
                   radius=17, blur=15, 
                   max_zoom=1, 
                 )
price_map = hmap.add_child(hm_wide)


df_duplicated1 = df_or[df_or.id.duplicated(keep = 'first')]
df_duplicated1 = df_duplicated1[['id', 'price', 'days_since_first']]
df_duplicated2 = df_or[df_or.id.duplicated(keep = 'last')]
df_duplicated2 = df_duplicated2[['id', 'price', 'days_since_first']]