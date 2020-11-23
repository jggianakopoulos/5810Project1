import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("default of credit card clients.xls", index_col=0, skiprows=1)
# Rename the default feature for simplicity
data = data.rename(columns={'default payment next month': "default"})

# Add a categorical column for default. This is only used for the class split, then dropped
data["default_cat"] = data.apply(lambda row: "Yes" if (row['default'] == 1) else "No", axis=1)

# Visualize the class split
graph = data['default_cat'].value_counts().plot(kind='bar', title="Default Values")
graph.set_xlabel("Default next month?")
graph.set_ylabel("Total customers")
plt.show()

# Drop categorical column for default
data = data.drop(["default_cat"], axis=1)

# Drop duplicate entries
data = data.drop_duplicates()
# Drop rows with null values
data = data.dropna()
# # Correlation
corrmatrix = data.corr()
ax = plt.axes()
ax.set_title("Correlation of Features")
sns.heatmap(corrmatrix, annot=False, ax=ax)
plt.show()

# Create 'average' columns for each of the redundant features

# For some reason, this feature set includes PAY_0 but not PAY_1. That is not a typo
data["pay_avg"] = data[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)

data["bill_amt_avg"] = data[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)
data["pay_amt_avg"] = data[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis=1)

clean_data = data.drop(['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], axis=1)

# Scale bill amount and pay amount averages

from sklearn import preprocessing
arr = clean_data.values
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(arr)
clean_data = pd.DataFrame(scaled, columns=clean_data.columns)

print(clean_data)
# # Correlation
corrmatrix = clean_data.corr()
ax = plt.axes()
ax.set_title("Correlation of Features")
sns.heatmap(corrmatrix, annot=False, ax=ax)
plt.show()

# # # # # # #
#
# KNN
#
# # # # # # #

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score

X = clean_data.drop(["default"], axis=1)
# print(X)
X_train, X_test, y_train, y_test = train_test_split(X, clean_data["default"])

# Test different K Values
for x in range(2, 11):
    if x % 2 == 1:
        continue

    print("k value:", x)
    knn = KNeighborsClassifier(x)
    knn.fit(X_train, y_train)
    result = knn.score(X_test, y_test)
    print("Average score:", cross_val_score(knn, X_test, y_test, cv=5).mean())
    print(result)
    print(confusion_matrix(y_test, knn.predict(X_test)))

# # # # # # #
#
# Logistic Regression
#
# # # # # # #

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

for solver in ["newton-cg", "lbfgs", "liblinear"]:
    print("SOLVER:", solver)
    for z in [1, 1000, 100000]:
        reg = LogisticRegression(C=z, solver=solver)
        reg.fit(X_train, y_train)
        print("C_value:", z)
        # Get a coefficient of determination for predictions
        result = reg.score(X_test, y_test)
        print(result)
        # Mean square error
        prediction = reg.predict(X_test)
        result = mean_squared_error(prediction, y_test)
        print("Average score:", cross_val_score(reg, X_test, y_test, cv=5).mean())


#
# Final Models
#


# KNN
knn = KNeighborsClassifier(8)
knn.fit(X_train, y_train)

# Confusion Matrix
matrix = plot_confusion_matrix(knn, X_test, y_test)
matrix.ax_.set_title("Confusion Matrix")
plt.show()


# Logistic Regression
reg = LogisticRegression(C=1000, solver="liblinear")
reg.fit(X_train, y_train)
prediction = reg.predict(X_test)
result = mean_squared_error(prediction, y_test)
print("MSE:", result)
print("Average score:", cross_val_score(reg, X_test, y_test, cv=5).mean())

#Confusion Matrix
matrix = plot_confusion_matrix(knn, X_test, y_test)
matrix.ax_.set_title("Confusion Matrix")
plt.show()






