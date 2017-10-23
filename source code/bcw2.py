
# coding: utf-8

# In[36]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


# ## **Data Analysis**

# In[37]:

# load data
data = pd.read_csv("../dataset/breast_cancer.csv")
data.drop('id', axis=1, inplace=True)    # remove the 'Id' column
data.head(3)


# In[38]:

data.describe()


# In[39]:

YCounts = data['diagnosis'].value_counts()
print("There are {0} benign instances, and {1} malignant instances.".format(YCounts[0], YCounts[1]))
YCounts.plot(kind='bar', figsize=(3,3))


# ### **Data Processing**

# In[40]:

# split dataset into training set and test set
train, test = train_test_split(data, test_size = 0.3)
print("There are {} instances in the training set.".format(train.shape[0]))
print("There are {} instances in the testing set.".format(test.shape[0]))


# In[41]:

featureSize = train.shape[1] - 1;
XTrain = train.iloc[:,1:31]    # features
YTrain = train.iloc[:,0]    # label

XTest = test.iloc[:,1:31]    # features
YTest = test.iloc[:,0]    # label


# ### Feature Selection (feature correlations)

# In[42]:

featureCorr = XTrain.corr()    # correlation between features
plt.subplots(figsize=(8,8))
# print(featureCorr.columns)
sns.heatmap(featureCorr, annot=False, fmt='.2f', linecolor='white', linewidths=0.5, cbar=False)


# In[43]:

plt.subplots(figsize=(8,8))
mask = featureCorr < 0.9
sns.heatmap(featureCorr, annot=False, fmt='.2f', linecolor='white', linewidths=0.1, cbar=False, mask=mask)


# In[44]:

plt.subplots(figsize=(8,8))
mask = featureCorr.iloc[0:10,0:10] < 0.9
sns.heatmap(featureCorr.iloc[0:10,0:10], annot=True, fmt='.2f', linecolor='white', linewidths=0.5, cbar=False, mask=mask)


# In[45]:

# high correlation feature groups: 
# (radius_mean, perimeter_mean, area_mean, radius_worst, perimeter_worst, area_worst)
# (concavity_mean, concave points_mean)
# (concave points_mean, concave points_worst)
# (radius_se, perimeter_se, area_se)
featureList = XTrain.columns
featureIgnored = ["radius_mean", "perimeter_mean", "area_mean", "perimeter_worst", "area_worst", "concave points_mean", "perimeter_se", "area_se"]
featureSelectedCol = [f for f in featureList if f not in featureIgnored]
print("selected features: {}.".format(featureSelectedCol))
# len(featureSelectedCol)


# ### Random Forest

# In[46]:

# train with all features
rf = RandomForestClassifier()
rf.fit(XTrain, YTrain)
YPredictRFTest = rf.predict(XTest)
print ("Accuracy of Random Forest (trained with all features) on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTest, YPredictRFTest)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTest, YPredictRFTest)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTest, YPredictRFTest, pos_label='M')))


# In[47]:

# get feature importance list in descending order
featureImportanceList = pd.Series(rf.feature_importances_, index=XTrain.columns).sort_values(ascending=False)
print("feature importance list in descending order:\n{}".format(featureImportanceList))


# In[48]:

# feature selected with random forest
featureSelectedRF = list(featureImportanceList[:10].index)    #######################TODO
print("Top 10 important features selected by Random Forest:\n{}".format(featureSelectedRF))


# In[49]:

# train with features selected by correlation
rf2 = RandomForestClassifier()
rf2.fit(XTrain[featureSelectedCol], YTrain)
YPredictRFTest2 = rf2.predict(XTest[featureSelectedCol])
print ("Accuracy of Random Forest on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTest, YPredictRFTest2)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTest, YPredictRFTest2)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTest, YPredictRFTest2, pos_label='M')))


# In[50]:

# train with features selected by Random Forest
rf3 = RandomForestClassifier()
rf3.fit(XTrain[featureSelectedCol], YTrain)
YPredictRFTest3 = rf3.predict(XTest[featureSelectedCol])
print ("Accuracy of Random Forest on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTest, YPredictRFTest3)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTest, YPredictRFTest3)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTest, YPredictRFTest3, pos_label='M')))


# ### **Gaussian Mixture Model**

# In[51]:

k = 2
# train with all features
gmm1 = GaussianMixture(n_components=k)
gmm1.fit(XTrain)
YPredictGMMTrain = gmm1.predict(XTrain)
# YPredictGMMProb = gmm1.predict_proba(XTrain)
# print(YPredictGMM[0:5])
# print(YPredictGMMProb[0:5])
YPredictGMMTest = gmm1.predict(XTest)
YPredictGMMTestProb = gmm1.predict_proba(XTest)

# check accuracy of GMM (trained with all features)
YTrainNumerical = YTrain.map({'M': 1, 'B': 0})
YTestNumerical = YTest.map({'M': 1, 'B': 0})
print ("Accuracy of GMM clustering (trained with all features) on training set is {0:.2f}."
       .format(metrics.accuracy_score(YTrainNumerical, YPredictGMMTrain)))
print ("Accuracy of GMM clustering (trained with all features) on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTestNumerical, YPredictGMMTest)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTestNumerical, YPredictGMMTest)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTestNumerical, YPredictGMMTest)))

# #######################
# print(YTest)
# print(YTestNumerical)
# print(YPredictGMMTest)
# print(YPredictGMMTestProb[:,1])
# print(YPredictGMMTestProb)


# In[57]:

# train with features selected by correlation
# gmm2 = GaussianMixture(n_components=k, tol=0.00000001, max_iter=10000)
gmm2 = GaussianMixture(n_components=k)
gmm2.fit(XTrain[featureSelectedCol])
YPredictGMMTrain2 = gmm2.predict(XTrain[featureSelectedCol])
# YPredictGMMProb = gmm2.predict_proba(XTrain[featureSelectedCol])
YPredictGMMTest2 = gmm2.predict(XTest[featureSelectedCol])
YPredictGMMTestProb2 = gmm2.predict_proba(XTest[featureSelectedCol])

# check accuracy of GMM (trained with selected features)
print ("Accuracy of GMM clustering (trained with selected features) on training set is {0:.2f}."
       .format(metrics.accuracy_score(YTrainNumerical, YPredictGMMTrain2)))
print ("Accuracy of GMM clustering (trained with selected features) on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTestNumerical, YPredictGMMTest2)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTestNumerical, YPredictGMMTest2)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTestNumerical, YPredictGMMTest2)))


# In[58]:

# train with features selected by Random Forest
gmm3 = GaussianMixture(n_components=k)
gmm3.fit(XTrain[featureSelectedRF])
YPredictGMMTrain3 = gmm3.predict(XTrain[featureSelectedRF])
YPredictGMMTest3 = gmm3.predict(XTest[featureSelectedRF])
YPredictGMMTestProb3 = gmm3.predict_proba(XTest[featureSelectedRF])

# check accuracy of GMM (trained with selected features)
print ("Accuracy of GMM clustering (trained with selected features) on training set is {0:.2f}."
       .format(metrics.accuracy_score(YTrainNumerical, YPredictGMMTrain3)))
print ("Accuracy of GMM clustering (trained with selected features) on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTestNumerical, YPredictGMMTest3)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTestNumerical, YPredictGMMTest3)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTestNumerical, YPredictGMMTest3)))


# ### **Naive Bayes**

# In[59]:

# train with all features
nb = GaussianNB()
nb.fit(XTrain, YTrain)
YPredictNBTest = nb.predict(XTest)
print ("Accuracy of Naive Bayes (trained with all features) on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTest, YPredictNBTest)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTest, YPredictNBTest)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTest, YPredictNBTest, pos_label='M')))


# In[60]:

# train with features selected by correlation
nb2 = GaussianNB()
nb2.fit(XTrain[featureSelectedCol], YTrain)
YPredictNBTest2 = nb2.predict(XTest[featureSelectedCol])
print ("Accuracy of Naive Bayes (trained with all features) on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTest, YPredictNBTest2)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTest, YPredictNBTest2)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTest, YPredictNBTest2, pos_label='M')))


# In[61]:

# train with features selected by Random Forest
nb3 = GaussianNB()
nb3.fit(XTrain[featureSelectedRF], YTrain)
YPredictNBTest3 = nb3.predict(XTest[featureSelectedRF])
print ("Accuracy of Naive Bayes (trained with all features) on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTest, YPredictNBTest3)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTest, YPredictNBTest3)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTest, YPredictNBTest3, pos_label='M')))


# In[62]:

# g = sns.FacetGrid(test, hue='diagnosis', size=4).map(plt.scatter, "radius_mean", "texture_mean")
# g.add_legend();


# ### PCA

# In[63]:

dim = 2
pca = PCA(n_components=dim)
pca.fit(XTest)
XPCA = pd.DataFrame(pca.transform(XTest), columns=['c1','c2'])


# In[64]:

pcaData = XPCA.copy()
pcaData['Y'] = list(YTestNumerical)

pcaData['YPredictGMM'] = YPredictGMMTest
pcaData['YPredictGMM2'] = YPredictGMMTest2
pcaData['YPredictGMM3'] = YPredictGMMTest3
pcaData['YPredictGMMProb'] = YPredictGMMTestProb[:,1]
pcaData['YPredictGMMProb2'] = YPredictGMMTestProb2[:,1]
pcaData['YPredictGMMProb3'] = YPredictGMMTestProb3[:,1]

pcaData['YPredictNB'] = YPredictNBTest
pcaData['YPredictNB2'] = YPredictNBTest2
pcaData['YPredictNB3'] = YPredictNBTest3
pcaData['YPredictNB'] = pcaData['YPredictNB'].map({'M': 1, 'B': 0})
pcaData['YPredictNB2'] = pcaData['YPredictNB2'].map({'M': 1, 'B': 0})
pcaData['YPredictNB3'] = pcaData['YPredictNB3'].map({'M': 1, 'B': 0})
pcaData.head(5)


# In[65]:

pcaData['YPredictRF'] = YPredictRFTest
pcaData['YPredictRF'] = pcaData['YPredictRF'].map({'M': 1, 'B': 0})
pcaData.head(5)


# In[66]:

def pcaPlot(df, groupby1, groupby2):
    fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    cmap = {0:'royalblue', 1:'crimson'}

    for i, cluster in df.groupby(groupby1):
        cluster.plot(x='c1', y='c2', kind='scatter', ax = ax1, color=cmap[i],
                     label="{0} {1}".format(groupby1, i), s=40, edgecolor="white")
        
    for i, cluster in df.groupby(groupby2):
        cluster.plot(x='c1', y='c2', kind='scatter', ax = ax2, color=cmap[i],
                     label="{0} {1}".format(groupby2, i), s=40, edgecolor="white")


# In[67]:

def pcaPlotWithProb(df, y1, y2):
    fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    mycmap = LinearSegmentedColormap.from_list('mycmap', ['royalblue', 'orchid','crimson'])
    
    df.plot(x='c1', y='c2', kind='scatter', ax = ax1, cmap=mycmap, c=y1, s=40, edgecolor="white")
    df.plot(x='c1', y='c2', kind='scatter', ax = ax2, cmap=mycmap, c=y2, s=40, edgecolor="white")


# In[68]:

# GMM
pcaPlot(pcaData, "Y", "YPredictGMM")
pcaPlot(pcaData, "Y", "YPredictGMM2")
pcaPlot(pcaData, "Y", "YPredictGMM3")


# In[69]:

pcaPlotWithProb(pcaData, "Y", "YPredictGMMProb")
pcaPlotWithProb(pcaData, "Y", "YPredictGMMProb2")
pcaPlotWithProb(pcaData, "Y", "YPredictGMMProb3")


# In[70]:

# NB
# pcaPlot(pcaData, "Y", "YPredictNB")
# pcaPlot(pcaData, "Y", "YPredictNB2")
# pcaPlot(pcaData, "Y", "YPredictNB3")
pcaPlotWithProb(pcaData, "Y", "YPredictNB")
pcaPlotWithProb(pcaData, "Y", "YPredictNB2")
pcaPlotWithProb(pcaData, "Y", "YPredictNB3")


# In[71]:

# random forest
pcaPlotWithProb(pcaData, "Y", "YPredictRF")


# ### Bagging of Models

# In[85]:

# baggingData = pcaData.loc[:,["YPredictGMM", "YPredictNB", "YPredictRF"]]
# baggingData = pcaData.loc[:,["YPredictGMM2", "YPredictNB2", "YPredictRF2"]]
baggingData = pcaData.loc[:,["YPredictGMM3", "YPredictNB3", "YPredictRF3"]]
baggingSum = baggingData.sum(1)
baggingResult = baggingSum > 1
baggingResult = baggingResult.astype(int)


# In[86]:

pcaData['YPredictBagging'] = baggingResult
pcaData.head(5)


# In[87]:

print ("Accuracy of bagging model on testing set is {0:.2f}."
       .format(metrics.accuracy_score(YTestNumerical, baggingResult)))

print("Confusion Matrix: {}".format(metrics.confusion_matrix(YTestNumerical, baggingResult)))   
print("Recall: {0:.2f}".format(metrics.recall_score(YTestNumerical, baggingResult)))


# In[88]:

# bagging
pcaPlotWithProb(pcaData, "Y", "YPredictBagging")

