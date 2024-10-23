import pandas as pd
import numpy as np
from sklearn import metrics 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, KFold, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import seaborn as sns

'''
read in the csv -- make it into a pandas dataframe
'''
def process_gene_expression_data(file_path, metadata_path):
    matrix = pd.read_csv(file_path)
    
    matrix = matrix.set_index('gene')
    matrix = matrix.T
    
    processed_data = pd.DataFrame(matrix)
    
    features_df = pd.read_csv(metadata_path, sep='\t', index_col=0)
    
    return processed_data, features_df

train_file_path = 'ArrayExpress-normalized.csv'
metadata_file_path = 'E-MTAB-11349.sdrf.txt'
train_data, train_features = process_gene_expression_data(train_file_path, metadata_file_path)
train_data = train_data.drop(train_data.index[:2])

# print(train_data[0:6])

columns_to_keep = ['Characteristics[age]', 'Characteristics[sex]', 'Characteristics[disease]']
filtered_train_features = train_features[columns_to_keep]
filtered_train_features = filtered_train_features.rename(columns={
    'Characteristics[age]': 'Age',
    'Characteristics[sex]': 'Sex',
    'Characteristics[disease]': 'Disease'
})

disease_mapping = {
    'normal': 0,
    "Crohn's disease": 1,
    'ulcerative colitis': 2
}

filtered_train_features['Disease_Label'] = filtered_train_features['Disease'].map(disease_mapping)

#print(filtered_train_features[0:6])

'''
join the two tables to get the full table of data
'''
all_data = train_data.join(filtered_train_features)
all_data['sick_status'] = all_data['Disease_Label'].apply(lambda x: 1 if (x==1 or x == 2) else 0)
all_data['Sex_Labels'] = all_data['Sex'].apply(lambda x: 1 if x == 'female' else 0)
#print(all_data[0:6])
columns = list(all_data.columns.values)
genes =[]
for i in range(len(columns)-6):
    if type(columns[i]) == str:
        genes.append(columns[i])

#print(columns[len(columns)-5:])
#print(all_data)

'''
logistic regression models for the prepared data
    Age/Sex ==> DiseaseLabel = 0 (Normal), 1 (Chrones), 2 (Ulcerative Colitis)
'''
X = all_data[['Age', 'Sex_Labels']]
y = all_data['Disease_Label']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=0)

log_reg1 = LogisticRegression()
log_reg1.fit(X_train1, y_train1)
y_pred1 = log_reg1.predict(X_test1)

#Visualize Performance
cnf_matrix1 = metrics.confusion_matrix(y_test1, y_pred1)
accuracy1 = metrics.accuracy_score(y_test1, y_pred1)

print("------------ Age/Sex ==> Disease Label (0, 1, 2) --------------------")
print("CNF Matrix: \n", cnf_matrix1)
print("Accuracy: ", accuracy1)

'''
Age/Sex ==> Sick/Not Sick
'''
X = all_data[['Age', 'Sex_Labels']]
y = all_data['sick_status']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=0)

log_reg2 = LogisticRegression()
log_reg2.fit(X_train2, y_train2)
y_pred2 = log_reg2.predict(X_test2)

#Visualize Performance
cnf_matrix2 = metrics.confusion_matrix(y_test2, y_pred2)
accuracy2 = metrics.accuracy_score(y_test2, y_pred2)
#define metrics
y_pred_proba = log_reg2.predict_proba(X_test2)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test2,  y_pred_proba)
auc = metrics.roc_auc_score(y_test2, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.title("Age/Sex ==> Sick Status (0, 1) ROC Curve")
plt.legend(loc=4)
#plt.show()

print("\n------------ Age/Sex ==> Sick Status (0/1) --------------------")
print("CNF Matrix: [[TP, FN],[FP, TN]]\n", cnf_matrix2)
print("Accuracy: ", accuracy2)

'''
Genes ==> Disease Label = 0 (normal), 1(chrones), 2 (UC)
'''
X = all_data[genes]
y = all_data['Disease_Label']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=0)

log_reg2 = LogisticRegression(max_iter = 3000)
log_reg2.fit(X_train2, y_train2)
y_pred2 = log_reg2.predict(X_test2)

#Visualize Performance
cnf_matrix2 = metrics.confusion_matrix(y_test2, y_pred2)
accuracy2 = metrics.accuracy_score(y_test2, y_pred2)

print("\n------------ Genes ==> Disease Label (0/1/2) --------------------")
print("CNF Matrix: \n", cnf_matrix2)
print("Accuracy: ", accuracy2)

'''
Genes ==> Sick/NotSick
'''
X = all_data[genes]
y = all_data['sick_status']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=0)

log_reg2 = LogisticRegression(max_iter = 3000)
log_reg2.fit(X_train2, y_train2)
y_pred2 = log_reg2.predict(X_test2)

#Visualize Performance
cnf_matrix2 = metrics.confusion_matrix(y_test2, y_pred2)
accuracy2 = metrics.accuracy_score(y_test2, y_pred2)
#define metrics
y_pred_proba = log_reg2.predict_proba(X_test2)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test2,  y_pred_proba)
auc = metrics.roc_auc_score(y_test2, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.title("Genes ==> Sick Status (0, 1) ROC Curve")
plt.legend(loc=4)
plt.show()

print("\n------------ Genes ==> Sick Status (0/1) --------------------")
print("CNF Matrix: \n", cnf_matrix2)
print("Accuracy: ", accuracy2)