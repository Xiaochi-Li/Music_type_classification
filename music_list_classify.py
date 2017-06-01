#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:43:30 2017

@author: lixiaochi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
import seaborn as sn
import pydotplus 

from sklearn.utils import shuffle
from pandas import DataFrame
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from IPython.display import Image 


#an mtrix contain 1476 song information in 18 features.
musiclist_df = pd.read_csv('/Users/lixiaochi/Documents/study/semester4/DECO7380/sem1_project/songData.csv')
#print musiclist_df.isnull().any()
musiclist_df = musiclist_df.fillna(method='ffill')
#a datafram stores randomly reordered data from musiclist dataframe
shuffled_mc_df = shuffle(musiclist_df)
song_id_name = musiclist_df [['name','id','uri','artist']]
music_lables = ['dinner','sleep','workout','party']

#delete inrrelated features
del  shuffled_mc_df ['name']
del  shuffled_mc_df ['id']
del  shuffled_mc_df ['uri']
del  shuffled_mc_df ['artist']

music_np_matrix = shuffled_mc_df.as_matrix()
#the training features of songs
music_features = music_np_matrix[:,:13]
#the lables of songs
music_list_lables = music_np_matrix[:,13]

#seperate data.
X_train, X_test, y_train, y_true = train_test_split(music_features, music_list_lables, test_size=0.3)

#music list decition tree classifier
clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=60)
clf = clf.fit(X_train,y_train)

#make prediction
y_predict = clf.predict(X_test)
#generate confusion matrix
cnf_matrix = confusion_matrix(y_true, y_predict, labels = music_lables)
cnf_matrix = pd.DataFrame(cnf_matrix, index = [i for i in music_lables],
                          columns = [i for i in music_lables])
# Plot non-normalized confusion matrix
sn.heatmap(cnf_matrix, annot= True, cmap="YlGnBu",fmt="d")
#calulate overall Scoring scoring.
precision, recall, fscore, support = score(y_true, y_predict,labels=music_lables)
print('precision: {}'.format(precision))
print precision_score(y_true, y_predict, average='weighted') 
print accuracy_score(y_true, y_predict)


#predict
features_list = list(shuffled_mc_df)
features_list.remove('category')
dot_data = tree.export_graphviz(clf,out_file=None,
                                feature_names=features_list,  
                         class_names=['dinner','sleep','workout','party'],  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("music_class_tree2.pdf") 