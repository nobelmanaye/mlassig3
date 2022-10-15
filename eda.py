### Nobel Manaye 
### Oct 2nd 2022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.preprocessing import StandardScaler
#df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 #'python-machine-learning-book-3rd-edition/'
                 #'master/ch10/housing.data.txt',
                 #header=None,
                 #sep='\s+')

# Total bedrooms is missing


df = pd.read_csv('covid.csv')

### Digitizing proximity to the ocean


df.columns = ['id','sex','patient_type','entry_date','date_symptoms','date_died','intubed','pneumonia','age','pregnancy','diabetes','copd','asthma','immsupr','hypertension','other_disease','cardiovascular','obesity','renal_chronic','tobacco','contact_other_covid','covid_res','icu']




cols = ['id','sex','patient_type','icu']




scatterplotmatrix(df[cols].values, figsize=(5, 8),names=cols,
alpha=0.5)



#plt.show()
#                   names=cols, alpha=0.5)
# df.head()

# print(df)
# cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
# scatterplotmatrix(df[cols].values, figsize=(10, 8), 
#                   names=cols, alpha=0.5)
# plt.tight_layout()
#plt.show()
#cm = np.corrcoef(df[cols].values.T)
#hm = heatmap(cm, row_names=cols, column_names=cols)
#plt.show()

# training = df.sample(frac = 0.8)
# testing = df.drop(training.index)


# Xtrain = training[cols].values
# ytrain = training['median_house_value'].values

# Xtest = testing[cols].values
# ytest = testing['median_house_value'].values


# scaler = StandardScaler()
# scaler2 = StandardScaler()
# scaler3 = StandardScaler()
# scaler4 = StandardScaler()
# scaler5 = StandardScaler()

# Xtrainscaled = scaler.fit_transform(Xtrain)
# ytrainscaled = scaler2.fit_transform(ytrain[:, np.newaxis]).flatten()

# Xtestscaled = scaler3.fit_transform(Xtest)
# ytestscaled = scaler4.fit_transform(ytest[:, np.newaxis]).flatten()























# Ytrain = training['median_house_value']
# Ytest  = testing['median_house_value']







