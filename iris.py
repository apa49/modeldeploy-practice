from re import S
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

st.title("First Model Deployment")
image=Image.open("iris.jpg")
st.image(image,use_column_width=True)
st.write("A simple data App with streamlit")
st.write("Let's explore some different classifiers and datasets")
dataset_name=st.sidebar.selectbox('Select-Datasets',('Breast Cancer','Iris','Wine'))
classifier=st.sidebar.selectbox('Select Classifier',('SVM','KNN'))
def get_dataset(name):
    data=None
    if name=='Iris':
        data=datasets.load_iris()
    elif name=='Wine':
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target
    return x,y

x,y=get_dataset(dataset_name)
st.dataframe(x)
st.write("shape of dataset is",x.shape)  
st.write('Unique target values',len(np.unique(y)))  
fig=plt.figure()
sns.boxplot(data=x,orient='h')
st.pyplot(fig)

plt.hist(x)
st.pyplot(fig)
def add_parameter(name_of_clf):
    params=dict()
    if name_of_clf=='SVM':
        C=st.sidebar.slider('C',0.01,15.0)
        params['C']=C
    else:
        name_of_clf=='KNN' 
        K=st.sidebar.slider('K',1,15)
    return params 
params=add_parameter(classifier)

def get_classifier(name_of_clf,params):
    clf=None
    if name_of_clf=='SVM':
        clf=SVC(C=params['C'])
    else:
        clf=KNeighborsClassifier(n_neighbors=params['K'])  
    return clf

clf=get_classifier(classifier,params) 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
st.write(y_pred)

accuracy=accuracy_score(y_test,y_pred)
st.write("Accuracy for your model is",accuracy)
st.write('Classifier name:',classifier)


