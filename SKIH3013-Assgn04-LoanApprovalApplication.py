#SKIH 3013 - Pattern Recognition
#Assignment 04 - Comparison on Classifiers' Performances based on Hyperparameter Tuning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

#initiate data access and descriptions
data = pd.read_csv("LoanApprovalPrediction.csv")
print("\n-------Database Structure:-------")
print(data.head(10))
obj = (data.dtypes == 'object')
print("Categorical variables:",len(list(obj[obj].index)))
print("\n-------Database information:-------")
print(data.info())
print("\n-------Missing data ---------------")
print(data.isnull().sum())

# Dropping Loan_ID column
data.drop(['Loan_ID'],axis=1,inplace=True)
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
index = 1

# --------Exploratory Data Analysis---------------

# To show the mean amount of the loan granted to males as well as females
print("\n---- Mean amount of loan granted based on gender ------")
print(data.groupby('Gender').mean()['LoanAmount'])

# joint plot - for applicant income vs co-applicant income
sns.jointplot(x = "ApplicantIncome", y= "CoapplicantIncome", data = data, hue="Loan_Status")
plt.show()

# joint plot - for applicant income vs loan ammount
sns.jointplot(x = "ApplicantIncome", y= "LoanAmount", data = data, hue="Loan_Status")
plt.show()

# data visualization 'Credit_History', 'Property_Area', 'Education', 'Self_Employed','Gender', 'Married'
features = ['Credit_History', 'Property_Area', 'Education', 'Self_Employed','Gender', 'Married']
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.countplot(data[col], hue=data['Loan_Status'])
plt.tight_layout()
plt.show()

# visualization - distribution density - applicant income / loan ammount
plt.subplots(figsize=(10, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
	plt.subplot(1, 2, i+1)
	sns.distplot(data[col])
plt.tight_layout()
plt.show()

# visualization - outliers among applicant income / loan ammount
plt.subplots(figsize=(10, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
	plt.subplot(1, 2, i+1)
	sns.boxplot(data[col])
plt.tight_layout()
plt.show()

# visualization - scatterplots - Cooapplicants income, loan ammount vs. applicant income by credit history
features = ['CoapplicantIncome', 'LoanAmount']
plt.subplots(figsize=(17, 7))
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sns.scatterplot(data=data, x=col,y='ApplicantIncome', hue='Credit_History')
plt.show()

# visualization - loan ammount term (months)
data['Loan_Amount_Term'].value_counts(normalize=True).plot.bar(title= 'Loan_Amount_Term')
plt.title("Loan Ammount Term")
plt.show()

#visualization - heatmap and correlation
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),cmap='jet',fmt='.2f',linewidths=2,annot=True)
plt.title("Correlation on Heatmap")
plt.show()



# -----------pre processing --------------
# Import label encoder
from sklearn import preprocessing
	
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])
# To find the number of columns with datatype==object
obj = (data.dtypes == 'object')
print("\nCategorical variables:",len(list(obj[obj].index)))

#replacing missing values with mean data
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

# ---------- data splitting ------------------
#data partioning - train/test 
from sklearn.model_selection import train_test_split
X = data.drop(['Loan_Status'],axis=1)
Y = data['Loan_Status']
X.shape,Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=1)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

#data preprocessing - normalization (normalized_x = (x – min) / (max – min))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print("-\n------before normalization------")
print(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("-\n------after normalization------")
print(X_train)

# -------- model development ----------------
#libaries for respective classifiers
#classifiers: K-NN, Random Forest, Support Vector Machine, Logistic Regression, Naive Bayes, Decision Tree, MLP Neural Nets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# class - classifiers
# Classifiers with hyperparameter tuning
classifiers = {
    'K-Nearest Neighbours': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'euclidean', 'manhattan']
        }
    },
    'Decision Tree': {
        'model': tree.DecisionTreeClassifier(),
        'params': {
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100],
            'criterion': ["gini", "entropy"]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [25, 50, 100, 150],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [3, 6, 9],
            'max_leaf_nodes': [3, 6, 9]
        }
    },
    
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {
            'penalty': ['l1', 'l2', 'none'],
            'solver': ['lbfgs', 'newton-cg'],
            'max_iter': [100, 1000, 300]
        }
    },
    'Support Vector Machine': {
    'model': SVC(probability=True), 
    'params': {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.5, 0.1, 0.001],
        'kernel': ['rbf', 'linear']
    }
},

    'Multilayer Perceptron': {
    'model': MLPClassifier(),
    'params': {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'learning_rate': ['constant'],
        'max_iter': [1000]
    }
}

}

# making predictions on the training set
# Grid Search Hyperparameter tuning
for model_name, model_params in classifiers.items():
    clf = GridSearchCV(model_params['model'], model_params['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, Y_train)
    print(f"\n------------------ {model_name} ------------------")
    print("Best Hyperparameters:", clf.best_params_)
    Y_pred = clf.predict(X_test)
    print("Accuracy score on test set: %.2f%%" % (100 * metrics.accuracy_score(Y_test, Y_pred)))
    print("Metric classification report:\n", metrics.classification_report(Y_test, Y_pred))
    print("Confusion Matrix:\n", metrics.confusion_matrix(Y_test, Y_pred))

# Loop through classifiers
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
for model_name, model_params in classifiers.items():
    clf = GridSearchCV(model_params['model'], model_params['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, Y_train)
    
    # Predict probabilities for positive class (class 1)
    Y_prob = clf.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

#predict new classes
Xnew = np.array([1,0,0,1,0,5849,0,0,360,1,1])
normXnew = scaler.transform(Xnew.reshape(1,-1))
print("\nnew classification for:", Xnew)
for model_name, model_params in classifiers.items():
    clf = GridSearchCV(model_params['model'], model_params['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(normXnew.reshape(1, -1))
    print(f"\nPrediction using {model_name}: {Y_pred}")


#initializing the majority voting based classifier	
case_yes = 0
case_no = 0
total_classifier =0

print("\n ++++++++++++++++++ Predicting a new case ++++++++++++++++++++++")
print("\nnew classification for:\n", Xnew)
# Predict new classes using the best-performing models
Xnew = np.array([[1, 1, 2, 0, 0, 11417, 1126, 225, 360, 1, 2, 0]])
normXnew = scaler.transform(Xnew.reshape(1, -1))

# Store the best-performing models in a dictionary
best_models = {}

for model_name, model_params in classifiers.items():
    clf = GridSearchCV(model_params['model'], model_params['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, Y_train)
    
    # Store the best-performing model in the dictionary
    best_models[model_name] = clf.best_estimator_

    Y_pred = clf.predict(normXnew.reshape(1, -1))
    print(f"\nPrediction using {model_name}: {Y_pred}")

    # Compute majority voting
    case_yes = sum(model.predict(normXnew.reshape(1, -1)) == 1 for model in best_models.values())
    total_classifier = len(best_models)
    case_no = total_classifier - case_yes
    
#count percentage votes
percent_yes = (case_yes/total_classifier)*100
percent_no = (case_no/total_classifier)*100

#finalized based on majority voting, we will decide either a person will/not be approved 
print("\n--------------------- Final Result ---------------------------------")
if case_yes > case_no:
    print("The applicant loan will be approved with overall %.2f%% vote" % percent_yes )
else:
    print("he applicant loan will not be approved with overall(%)%.2f%% vote" % percent_no)       

# Store the best-performing models in a dictionary
best_models = {}

for model_name, model_params in classifiers.items():
    clf = GridSearchCV(model_params['model'], model_params['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, Y_train)
    
    # Store the best-performing model in the dictionary
    best_models[model_name] = clf.best_estimator_

    Y_pred = clf.predict(normXnew.reshape(1, -1))
    print(f"\nPrediction using {model_name}: {Y_pred}")

    # Compute majority voting
    case_yes = sum(model.predict(normXnew.reshape(1, -1)) == 1 for model in best_models.values())
    total_classifier = len(best_models)
    case_no = total_classifier - case_yes

# New datasets
new_cases = {
    'A': np.array([1, 1, 2, 0, 0, 11417, 1126, 225, 360, 1, 2, 0]),
    'B': np.array([1, 1, 3, 1, 1, 5703, 0, 130, 360, 1, 0, 1]),
    'C': np.array([0, 1, 0, 0, 0, 4333, 2451, 110, 360, 1, 2, 0])
}

# Normalize new datasets
norm_new_cases = {case: scaler.transform(data.reshape(1, -1)) for case, data in new_cases.items()}

# Make predictions using majority voting
predictions = []

for case, norm_data in norm_new_cases.items():
    case_yes = sum(model.predict(norm_data) == 1 for model in best_models.values())
    total_classifier = len(best_models)
    case_no = total_classifier - case_yes

    # Compute majority voting
    if case_yes > case_no:
        predictions.append(1)
    else:
        predictions.append(0)

# Display the results
for case, prediction in zip(new_cases.keys(), predictions):
    print(f"\nPrediction for Case {case}: {'Approved' if prediction == 1 else 'Not Approved'}")









