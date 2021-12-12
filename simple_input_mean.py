#!/usr/bin/env python
# coding: utf-8

# Importing dataset

# In[1]:


import random
random.seed(0)


# In[2]:


from platform import python_version
print(python_version())


# In[3]:


import pandas as pd

df = pd.read_csv("BetaData_450K_SimpleImpute_Zero_1.csv")


# In[4]:


df = df.iloc[: , 1:]  #deleting 1st column 
df = df.drop(['Donor_Sample'], axis=1) #deleting this column
df = df.fillna(0)


# In[5]:


#target = df['is_tumor']
df.is_tumor = df.is_tumor.astype(int) #converted is_tumor value float to int


# In[6]:


print("Number of markers/ features", len(df.columns)-1)
print("Number of samples", len(df))
print(df['is_tumor'].value_counts())
#print(df)


# Splitting data into features and target

# In[7]:


#split dataset into features and target
X = df.iloc[:,0:len(df.columns)-1].values
print(type(X))
print("X = ", X.shape)
Y = df.iloc[:,len(df.columns)-1].values
print("Y = ",Y.shape)


# Feature scaling

# In[8]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X.shape)
print(Y.shape)
Y_original = Y


# Handling oversampling with SMOTE

# In[9]:


'''from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)
#print(X.shape)
#print(Y.shape)
#print(target.shape)'''


# In[10]:


target = pd.DataFrame(Y)  #This is the target (Y) after oversampling


# Reduced features

# In[11]:


from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


fvals, pvals = f_classif(X, Y)
#print(pvals)
to_keep = pvals < (0.05/X.shape[1])
to_remove = pvals >= (0.05/X.shape[1])
#print(to_remove)
#p value less than .05/X is removed 
X_anova_remove = np.delete(X, obj=to_keep, axis=1)
#p value greater than .05/X is removed
X_anova = np.delete(X, obj=to_remove, axis=1)
print("Total number of features:", X.shape[1])
print("Only ANOVA")
print("Features with p value less than this threshold (0.05/ total number of features) was kept")
print("Number of selected features using ANOVA F-test:", X_anova.shape[1])
print("reduced features using anova", X_anova_remove.shape[1])
selected_feat_anova= df.iloc[:,:-1].columns[(to_keep)]
#print("Selected features using ANOVA F-test:", selected_feat_anova)
#print(X_anova)

print("\n")
print("Only random forest")
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X, Y)
rf_sel_features=sel.get_support()
X_rf = np.delete(X, obj=~np.array(rf_sel_features), axis=1)
X_rf_remove = np.delete(X, obj=np.array(rf_sel_features), axis=1)
selected_feat_rf= df.iloc[:,:-1].columns[(sel.get_support())]
print("Number of selected features using RF:", X_rf.shape[1])
print("reduced features using rf", X_rf_remove.shape[1])
#print("Selected features using ANOVA F-test:", selected_feat_rf)

print("\n")
print("Both ANOVA and random forest")
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_anova, Y)
anova_rf_sel_features=sel.get_support()
X_anova_rf = np.delete(X_anova, obj=~np.array(anova_rf_sel_features), axis=1)
X_anova_rf_remove = np.delete(X_anova, obj=np.array(anova_rf_sel_features), axis=1)
#selected_feat_anova_rf= df.iloc[:,:-1].columns[(sel.get_support())]
print("Number of selected features using anova and RF:", X_anova_rf.shape[1])
print("reduced features using anova and rf", X_anova_rf_remove.shape[1])
#print("Selected features using ANOVA F-test:", selected_feat_anova_rf)

print("\n")
print("Only Linear Regression")
sel = SelectFromModel(LinearRegression())
sel.fit(X, Y)
LR_sel_features=sel.get_support()
X_LR = np.delete(X, obj=~np.array(LR_sel_features), axis=1)
X_LR_remove = np.delete(X, obj=np.array(LR_sel_features), axis=1)
selected_feat_LR= df.iloc[:,:-1].columns[(sel.get_support())]
print("Number of selected features using LR:", X_LR.shape[1])
print("reduced features using LR", X_LR_remove.shape[1])
#print("Selected features using LR F-test:", selected_feat_LR)


# Train test validation split on total features

# In[12]:


#import imblearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, cohen_kappa_score, f1_score, roc_auc_score



print("train and test split on total dataset", len(X))
X_train0, X_test, y_train0, y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
print("train and validation split on training dataset", len(X_train0))
X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.2, random_state = 0)
print("Training data:",X_train.shape)
print("Validation data", X_val.shape)
print("test data",X_test.shape)


# Random forest model on total features

# In[13]:


RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print("on validation dataset")
y_pred = RF.predict(X_val)
print("tn, fp, fn, tp=", confusion_matrix(y_val, y_pred).ravel())
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
print("Recall/Sensitivity : " + str(round(tp / (tp + fn), 2)))
print("Specificity : " + str(round(tn / (tn + fp), 2)))
print("F1 score : " + str(round(f1_score(y_val, y_pred))))
print("Cohens kappa : " + str(round(tn / (tn + fp), 2)))
print("ROC AUC : " + str(round(roc_auc_score(y_val, y_pred))))

print("\n")
print("on test dataset")
y_pred = RF.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tn, fp, fn, tp=", confusion_matrix(y_test, y_pred).ravel())
print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
print("Recall/Sensitivity : " + str(round(tp / (tp + fn), 2)))
print("Specificity : " + str(round(tn / (tn + fp), 2)))
print("F1 score : " + str(round(f1_score(y_test, y_pred))))
print("Cohens kappa : " + str(round(tn / (tn + fp), 2)))
print("ROC AUC : " + str(round(roc_auc_score(y_test, y_pred))))


# Train test validation split on selected features

# In[14]:


print("train and test split on total dataset", len(X))
X_train0, X_test, y_train0, y_test = train_test_split(X_anova_rf, Y, test_size=0.2, random_state = 0)
print("train and validation split on training dataset", len(X_train0))
X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.2, random_state = 0)
print("Training data:",X_train.shape)
print("Validation data", X_val.shape)
print("test data",X_test.shape)


# Random forest model on reduced features

# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, cohen_kappa_score, f1_score, roc_auc_score

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
print("on validation dataset")
y_pred = RF.predict(X_val)
print("tn, fp, fn, tp=", confusion_matrix(y_val, y_pred).ravel())
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
print("Recall/Sensitivity : " + str(round(tp / (tp + fn), 2)))
print("Specificity : " + str(round(tn / (tn + fp), 2)))
print("F1 score : " + str(round(f1_score(y_val, y_pred))))
print("Cohens kappa : " + str(round(tn / (tn + fp), 2)))
print("ROC AUC : " + str(round(roc_auc_score(y_val, y_pred))))

print("\n")
print("on test dataset")
y_pred = RF.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tn, fp, fn, tp=", confusion_matrix(y_test, y_pred).ravel())
print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
print("Recall/Sensitivity : " + str(round(tp / (tp + fn), 2)))
print("Specificity : " + str(round(tn / (tn + fp), 2)))
print("F1 score : " + str(round(f1_score(y_test, y_pred))))
print("Cohens kappa : " + str(round(tn / (tn + fp), 2)))
print("ROC AUC : " + str(round(roc_auc_score(y_test, y_pred))))


# Output selected features to CSV

# In[18]:


anovaRan = pd.DataFrame(X_anova_rf)
anovaRan['is_tumor'] = target
print(anovaRan)
anovaRan.to_csv("450_sim_anova_RF_woSMOTE.csv")


# Common features using different methods

# In[19]:


df1=df.drop(["is_tumor"], axis=1)
print("ANOVA test")
bestfeatures = SelectKBest(score_func=f_classif, k=X_anova.shape[1])
#df1 = pd.DataFrame(X)
anova = bestfeatures.fit(df1, Y_original)
dfscores = pd.DataFrame(anova.scores_)
dfcolumns = pd.DataFrame(df1.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['fFeature', 'fScore']  # naming the dataframe columns
best_features = featureScores.nlargest(X_anova.shape[1], 'fScore')
#print(best_features)
anova_bestfeatures = best_features['fFeature'].tolist()
print(len(anova_bestfeatures))

print("Random forest")
RF = RandomForestClassifier()
RF.fit(X, Y)
# feature importance
importance = RF.feature_importances_
dfscores = pd.DataFrame(RF.feature_importances_)
dfcolumns = pd.DataFrame(df1.columns)
# concat two dataframes for better visualization
fs = pd.concat([dfcolumns, dfscores], axis=1)
fs.columns = ['rfFeature', 'rfScore']  # naming the dataframe columns
best_features = fs.nlargest(X_rf.shape[1], 'rfScore')
#print(best_features)
rf_bestfeatures = best_features['rfFeature'].tolist()
print(len(rf_bestfeatures))


print("ANOVA and Random forest")
#df_anova = 
RF = RandomForestClassifier()
RF.fit(X_anova, Y)
# feature importance
importance = RF.feature_importances_
dfscores = pd.DataFrame(importance)
dfcolumns = pd.DataFrame(df1.columns)
# concat two dataframes for better visualization
fs = pd.concat([dfcolumns, dfscores], axis=1)
fs.columns = ['rfFeature', 'rfScore']  # naming the dataframe columns
best_features = fs.nlargest(X_anova_rf.shape[1], 'rfScore')
anova_rf_bestfeatures = best_features['rfFeature'].tolist()
print(len(anova_rf_bestfeatures))
anova_rf_markers = pd.DataFrame (anova_rf_bestfeatures,columns=['Reduced_Markers_AnovaRF'])
anova_rf_markers.to_csv("450_sim_AnovaRF_markers_woSMOTE.csv")

print("Linear regression")
LnR = LinearRegression()
LnR.fit(X, Y)
dfscores = pd.DataFrame(LnR.coef_)
dfcolumns = pd.DataFrame(df1.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['lFeature', 'lScore']  # naming the dataframe columns
best_features = featureScores.nlargest(X_LR.shape[1], 'lScore')
LnR_bestfeatures = best_features['lFeature'].tolist()


# In[20]:


print("Total features:", X.shape[1])
print("Total anova features:", len(anova_bestfeatures))
print("Total RF features:", len(rf_bestfeatures))
print("Total LR features:", len(LnR_bestfeatures))
common_all = list(set.intersection(*map(set, [anova_bestfeatures, rf_bestfeatures, LnR_bestfeatures])))
print("anova, rf, lr:",len(common_all))
common_ar = list(set.intersection(*map(set, [anova_bestfeatures, rf_bestfeatures])))
print("anova, rf:",len(common_ar))
common_al = list(set.intersection(*map(set, [anova_bestfeatures, LnR_bestfeatures])))
print("anova, LR:",len(common_al))
common_lr = list(set.intersection(*map(set, [rf_bestfeatures, LnR_bestfeatures])))
print("LR, RF:",len(common_lr))


# Draw venn diagram to display common features

# In[21]:


#Venn diagram
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

fig = plt.figure()
plt.figure(figsize=(8,8))
venn3(subsets=(len(anova_bestfeatures), len(rf_bestfeatures), len(common_ar), len(LnR_bestfeatures), len(common_al), len(common_lr), len(common_all)), 
      set_labels = ('ANOVA', 'Random Forest', 'Linear Regression'),
      set_colors=('#c4e6ff', '#F4ACB7','#9D8189'),
 alpha = 1)
plt.show()
#fig.savefig('test.png', bbox_inches='tight')
plt.clf()
plt.close()


# Draw venn diagram to display common features for only RF and ANOVA 

# In[22]:


#Venn diagram
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

fig = plt.figure()
plt.figure(figsize=(6,6))
print(len(common_ar))
venn2(subsets=(len(anova_bestfeatures), len(rf_bestfeatures), len(common_ar)), 
      set_labels = ('ANOVA', 'Random Forest'),
      set_colors=('#c4e6ff', '#F4ACB7'), alpha = 1)
plt.show()
fig.savefig('test.png', bbox_inches='tight')
plt.clf()
plt.close()

