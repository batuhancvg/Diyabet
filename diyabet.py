# import libreary
import pandas as pd #data science library
import numpy as np #numerical python library
import matplotlib.pyplot as plt #data visualization library
import seaborn as sns #data visualization library

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

import warnings
warnings.filterwarnings('ignore') 

# import data and EDA 
    #load data
df= pd.read_csv('diyabet\\diabetes.csv')
df_name = df.columns

#sutun isimleri(buyuk kucuk harf, bosluk, ingilizce olmayan karakter)
#sample sayisi ve kayip veri olup olmadigi
#veri tipleri
df.info() #data information 

decsribe = df.describe() #veri aciklamasi(ortalama, standart sapma, min, max, ceyreklikler)
print(decsribe)


sns.pairplot(df, hue='Outcome') 
plt.show()

def plot_corelation_heatmap(df):

    corr_matrix = df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cmap='coolwarm')
    plt.title('Corelation of Features')
    plt.show()

plot_corelation_heatmap(df)

# outlier detection 
def detect_outliers_iqr(df):

    outlier_indices = []
    outliers_df = pd.DataFrame()

    for col in df.select_dtypes(include=["float64","int64"]).columns:
        Q1 = df[col].quantile(0.25) #1. ceyreklik
        Q3 = df[col].quantile(0.75) #3. ceyreklik

        IQR = Q3 - Q1 #interquartile range

        lower_bound = Q1 - 1.5*IQR  #alt sinir
        upper_bound = Q3 + 1.5*IQR  #ust sinir

        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_indices.extend(outliers_in_col.index)
        outliers_df = pd.concat([outliers_df, outliers_in_col], axis=0)

    #remove duplicates indices
    outlier_indices = list(set(outlier_indices))
    #remove duplicates rows in the outlier dataframe
    outliers_df = outliers_df.drop_duplicates()

    return outlier_indices, outliers_df

outlier_indices, outliers_df = detect_outliers_iqr(df)
# print(outliers_df)
# print(outlier_indices)

#remove outliers from the dataframe
df_cleaned = df.drop(outlier_indices).reset_index(drop=True)

# Train Test Split
X = df_cleaned.drop('Outcome', axis=1)
y = df_cleaned['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# standarization
scaler = StandardScaler()

X_train_sclaed = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model training and evaluation
"""
LogisticRegression
DecisionTreeClassifier
KNeighborsClassifier
GaussianNB
SVC
RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
"""

def getBasedModel():
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier()))
    basedModels.append(('DT' , DecisionTreeClassifier()))
    basedModels.append(('NB'   , GaussianNB()))
    basedModels.append(('SVM'  , SVC(probability=True)))
    basedModels.append(('RF'   , RandomForestClassifier()))
    basedModels.append(('AdaB'  , AdaBoostClassifier()))
    basedModels.append(('GBM'   , GradientBoostingClassifier()))
    return basedModels

def baseModelsTraning(X_train, y_train,models):
    
    results = []
    names = []
    for name , model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name} : accuracy: {cv_results.mean()}, std: ({cv_results.std()})")
    return names, results

def plotBox(names,results):
    df = pd.DataFrame({names[i]: results[i] for i in range(len(names))})
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.show()

models = getBasedModel()
names,results = baseModelsTraning(X_train_sclaed, y_train,models)
plotBox(names,results)

# hypirparameter tuning
    # DT hyperparameter set
param_grid ={
    "criterion": ["gini", "entropy"],
    "max_depth": [10,20,30,40,50],
    "min_samples_split": [2,5,10,],
    "min_samples_leaf": [1,2,4]    
}

dt = DecisionTreeClassifier()
    #grid search cv 
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring="accuracy")
    #tranining
grid_search.fit(X_train, y_train)
print("En iyi parametreler:", grid_search.best_params_)

best_dt_model = grid_search.best_estimator_
y_pred = best_dt_model.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
"""
[[95 14]  etiketi 0 olan 95 veriden 95 tanesi dogru tahmin edilmis, 14 tanesi yanlis tahmin edilmis
 [29 22]] etiketi 1 olan 51 veriden 22 tanesi dogru tahmin edilmis, 29 tanesi yanlis tahmin edilmis
"""
print("Classification_report")
print(classification_report(y_test, y_pred))

# model testin with real data   
new_data = np.array([[6,148,72,35,0,34.6,0.627,51]])

best_dt_model.predict(new_data)
new_prediction = best_dt_model.predict(new_data)
print("New Prediction:", new_prediction)

