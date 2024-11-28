# Import necessary libraries
from sklearn.model_selection import GridSearchCV

# Initialize data access and descriptions
data = pd.read_csv("LoanApprovalPrediction.csv")

# Drop unnecessary columns
data.drop(['Loan_ID'], axis=1, inplace=True)

# Encode categorical variables
label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

# Replace missing values with mean data
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

# Data splitting
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Data preprocessing - normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.5, 0.1, 0.001],
            'kernel': ['rbf', 'linear']
        }
    },
    'Multilayer Perceptron': {
        'model': MLPClassifier(),
        'params': {
            'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
            'max_iter': [100, 500, 1000],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive']
        }
    }
}

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

# Predict new classes using the best-performing models
Xnew = np.array([1, 0, 0, 1, 0, 5849, 0, 0, 360, 1, 1])
normXnew = scaler.transform(Xnew.reshape(1, -1))

for model_name, model_params in classifiers.items():
    clf = GridSearchCV(model_params['model'], model_params['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(normXnew.reshape(1, -1))
    print(f"\nPrediction using {model_name}: {Y_pred}")

# Initialize the majority voting based classifier
case_yes = 0
case_no = 0
total_classifier = 0

print("\n ++++++++++++++++++ Predicting a new case ++++++++++++++++++++++")
print("\nNew classification for:\n", Xnew)

for model_name, model_params in classifiers.items():
    clf = GridSearchCV(model_params['model'], model_params['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(normXnew.reshape(1, -1))

    # Compute majority voting
    if Y_pred == 1:
        case_yes += 1
    else:
        case_no += 1

    total_classifier += 1

# Count percentage votes
percent_yes = (case_yes / total_classifier) * 100
percent_no = (case_no / total_classifier) * 100

# Finalized based on majority voting
print("\n--------------------- Final Result ---------------------------------")
if case_yes > case_no:
    print("The applicant loan will be approved with overall %.2f%% vote" % percent_yes)
else:
    print("The applicant loan will not be approved with overall %.2f%% vote" % percent_no)
