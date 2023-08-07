#!python
import os
import _pickle as cPickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

train_mode=0

filesdir=os.path.dirname(os.path.abspath(__file__))
out_saved_model_file = filesdir+"/msu_hs_model.pkl"

hs_sample_file  = filesdir+"/pos_sample.csv"
nhs_sample_file = filesdir+"/neg_sample.csv"

df_hot = pd.read_csv(hs_sample_file,delimiter="\t")
df_cold = pd.read_csv(nhs_sample_file,delimiter="\t")

df_hot['sign_hs']  = True #'hot'#'hot'
df_cold['sign_hs'] = False #'no'#'cold'

print("len(df_hot)={}".format(len(df_hot)))
print("len(df_cold)={}".format(len(df_cold)))

df_all = pd.concat([df_cold,df_hot],ignore_index=True)
# df_all = df_all.dropna
df_all.dropna(inplace=True)

y_all = df_all['sign_hs'].values

######
df_all['test_i41'] =df_all['4']-df_all['4_mean']
df_all['test_iDT1']=df_all['DT']-df_all['DT_mean']
df_all['test_i43'] =df_all['4']-df_all['4_mean']-df_all['4_mad']
df_all['test_iDT3']=df_all['DT']-df_all['DT_mean']-df_all['DT_mad']
df_all['test_i53'] =df_all['5']-df_all['5_mean']-df_all['5_mad']
list_test = ['test_i41','test_iDT1','test_i43','test_iDT3','test_i53','2','4','4_mean','4_mad','5','5_mean','5_mad','DT','DT_mean','DT_mad','BkgFire_mad'] #'day',

if 'list_test' in locals():
    print("small test mode {}".format(list_test))
    X_all = df_all[list_test].values.astype('float32')
else:
    X_all = df_all.drop('sign_hs',axis=1).values.astype('float32')
    list_test = df_all.drop('sign_hs',axis=1).columns
    print("all features test mode {}".format(list_test))


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=42)
# X_train = X_test = X_all
# y_train = y_test = y_all

classifier = RandomForestClassifier(random_state=42)
if(train_mode==1):
    print("\nstart fit")
    classifier.fit(X_train, y_train)
else:
    with open(out_saved_model_file, 'rb') as fid:
        classifier = cPickle.load(fid)

print("start predict (only to test)")
y_predict = classifier.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_predict).ravel()
print( "confusion_matrix:\n{} (tp)\t{} (fn)\n{} (fp)\t{} (tn)".format(tp, fn, fp, tn) )
print( "\nclassification_report:\n", classification_report(y_test, y_predict, target_names=['no','hot']) )
# precision,recall,fscore,support = map(lambda x: round(x[0]*1000)/10, precision_recall_fscore_support(y_test,y_predict,average=None,labels=['hot']))
# print( "{} {} {} {}".format(precision,recall,fscore,support))

# save the classifier
if(train_mode==1):
    with open(out_saved_model_file, 'wb') as fid:
        cPickle.dump(classifier, fid)    
print("finish\n")


