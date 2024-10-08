import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import joblib

warnings.filterwarnings("ignore")   

file_path = './Carbon Emission.csv'
data = pd.read_csv(file_path)

data.replace(np.nan, 'None', inplace=True)

# sdasdas
df= pd.DataFrame(data)
df = df.drop(columns=["Cooking_With","Recycling","Sex","Body Type"])

diet_mapping ={'pescatarian':0, 'vegetarian':1, 'omnivore':2, 'vegan':3}
shower_mapping={'daily':0,'less frequently':1, 'more frequently':2, 'twice a day':3}
heating_mapping={'coal':0, 'natural gas':1, 'wood':2, 'electricity':3}
transport_mapping={'public':0, 'walk/bicycle':1, 'private':2}
transport_type_mapping={'None':0, 'petrol':1, 'diesel':2, 'hybrid':3, 'lpg':4, 'electric':5}
travel_by_air_mapping={'frequently':2, 'rarely':1, 'never':0, 'very frequently':3}
waste_mapping={'large':2, 'extra large':3, 'small':0, 'medium':1}
energy_mapping={'No':2, 'Sometimes':1, 'Yes':0}
social_mapping={'often':0, 'never':1, 'sometimes':2}

df["diet"] = df["diet"].map(diet_mapping)
df["showerFrequency"] = df["showerFrequency"].map(shower_mapping)
df["heatingSource"] = df["heatingSource"].map(heating_mapping)
df["transportation"] = df["transportation"].map(transport_mapping)
df["vehicleType"] = df["vehicleType"].map(transport_type_mapping)
df["travelFrequency"] = df["travelFrequency"].map(travel_by_air_mapping)
df["wasteBagSize"] = df["wasteBagSize"].map(waste_mapping)
df["energyEfficiency"] = df["energyEfficiency"].map(energy_mapping)
df["socialActivity"] = df["socialActivity"].map(social_mapping)

print(df.head())

X = df.drop(columns=['CarbonEmission'])
y = df['CarbonEmission']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)

regressor.fit(X, y)

y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R-squared (Accuracy): {r2}')

# with open('regressor.pkl', 'wb') as file:
#     pickle.dump(regressor, file)


joblib.dump(regressor,'random_forest_model.joblib')



# print(df.shape)
# new_data = [[1,1,2,1,2,1,66,1,2000,3,2,8,2,3,4]]
# carbon = regressor.predict(new_data)
# print(carbon)

