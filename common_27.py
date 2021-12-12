import pandas as pd
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import numpy as np

df_27 = pd.read_csv("BetaData_27K_SimpleImpute_Mean_1.csv")
df_450 = pd.read_csv("BetaData_450K_SimpleImpute_Zero_1.csv")
markers_27=list(df_27.columns)[2:-1]
print("27k total markers:",len( markers_27))
markers_450=list(df_450.columns)[2:-1]
print("450k total markers:", len(markers_450))
common_merkers=set(markers_27).intersection(markers_450)
print(len(common_merkers))

df = df_27.iloc[: , 1:]  #deleting 1st column 
df = df.drop(['Donor_Sample'], axis=1) #deleting this column
df = df.fillna(0)
#target = df['is_tumor']
df.is_tumor = df.is_tumor.astype(int) #converted is_tumor value float to int
print("Number of markers/ features", len(df.columns)-1)
print("Number of samples", len(df))
print(df['is_tumor'].value_counts())
#print(df)
#split dataset into features and target
X = df.iloc[:,0:len(df.columns)-1].values
print("X = ", X.shape)
Y = df.iloc[:,len(df.columns)-1].values
print("Y = ",Y.shape)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#print(X)
print(X.shape)
print(Y.shape)
Y_original = Y

oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)
print(X.shape)
print(Y.shape)

pca = PCA(10)
pca.fit(X) #calculates loading scores and variations each PC has
pca_data = pca.transform(X)
n_pcs= pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

most_important_names = [markers_27[most_important[i]] for i in range(n_pcs)]
print(most_important_names)
common_merkers_2=set(most_important_names).intersection(common_merkers)
print(common_merkers_2)
