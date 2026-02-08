import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from ydata_profiling import ProfileReport
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV


df=pd.read_csv("Boston_house.csv")
# print(df.head())
# print(df.info())
# print(df.describe())
df.drop(columns=['CHAS','B'],axis=1,inplace=True)



x=df.drop('PRICE',axis=1)
y=df['PRICE']

# train test split 
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.33,random_state=42)

preprocessing = ColumnTransformer( transformers=[
                                                  ('log_trans', Pipeline([
                                                                          ('scale',RobustScaler()),
                                                                          ('log',FunctionTransformer(np.log1p))
                                                                          
                                                                          ]),['CRIM','ZN']),
                                                  ('numerical', Pipeline([
                                                                          ('scale',StandardScaler())
                                                                         ]) ,['INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','LSTAT'])
                                                ])

final_model = Pipeline([
    ('preprocessing', preprocessing),
    ('model_selection', RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        random_state=42,
        n_jobs=-1
    ))
])

final_model.fit(X_train, Y_train)

y_pred = final_model.predict(X_test)
accuracy = r2_score(Y_test, y_pred)
print("Final Model Accuracy:", accuracy)


# param_grid = {
#     "model_selection__n_estimators": [100, 200, 300, 500],
#     "model_selection__max_depth": [None, 10, 15, 20],
#     "model_selection__min_samples_split": [2, 5, 10],
#     "model_selection__min_samples_leaf": [1, 2, 4],
#     "model_selection__max_features": ["sqrt", "log2"]
# }


# search = RandomizedSearchCV(
#     model,
#     param_grid,
#     n_iter=20,
#     cv=5,
#     scoring="r2",
#     n_jobs=-1,
#     random_state=42
# )

# search.fit(X_train, Y_train)
# best_model = search.best_estimator_


# print("Best Parameters:", search.best_params_)
# print("Best CV Score:", search.best_score_)



# # on which parameters the model has been trained 
# # model.gets_params()

# y_pred = best_model.predict(X_test)


# accuracy = r2_score(Y_test,y_pred)
# print("Accuracy :",accuracy)

import pickle 

# save model 
pickle.dump(final_model,open("boston.pkl","wb"))

# load model 
model = pickle.load(open("boston.pkl", "rb"))

# Prediction ke liye DataFrame banana compulsory hai
# (columns same hone chahiye jo training me the) 

input_data = pd.DataFrame([[0.2, 12.5, 5.3, 0.4, 6.5, 45, 4.2, 2, 300, 18.7, 10.5]],
columns=['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','LSTAT'])


# prediction karna 
price = model.predict(input_data)
print("Predicted House Price:", price[0])





