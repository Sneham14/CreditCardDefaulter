import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, f1_score, accuracy_score

dataset=pd.read_csv("UCI_Credit_Card.csv")
#check if any data missing 
print(dataset)
dataset.info()

X=dataset.iloc[:, 1:-1].values
Y=dataset.iloc[:, -1].values

corr_data=dataset.corr()
plt.figure(figsize=(25, 20))
sns.heatmap(corr_data, annot=True, vmin=-1.0, cmap='mako')
plt.show()

corr_data.to_csv("feature_imp.csv")
print(corr_data)


def onehot_encode(df, column_dict):
    df= df.copy()
    for column, prefix in column_dict.items():
        dummies=pd.get_dummies(df[column], prefix=prefix)
        df=pd.concat([df,dummies], axis=1)
        df=df.drop(column, axis=1)
    return df

def preprocess_inputs(df):
    df=df.copy()
    df=df.drop("ID", axis=1)
    
    df= onehot_encode(df, 
                    {
                        "EDUCATION":'Edu',
                        "MARRIAGE":'Mar'
                    })
    
    #Split df into X and Y
    Y=df['default.payment.next.month'].copy()
    X=df.drop('default.payment.next.month', axis=1).copy()


    scaler=StandardScaler()
    X=pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, Y
    
    
X, Y= preprocess_inputs(dataset)
print(X)
print(Y)

defs = dataset["default.payment.next.month"].sum()
non_defs = len(dataset) - defs

# Percentage
def_perc = round(defs / len(dataset) * 100, 1)
non_def_perc = round(non_defs / len(dataset) * 100, 1)

# Create a bar chart
x = [0, 1]
y = [non_defs, defs]
labels = ["Non-defaulters", "Defaulters"]
plt.bar(x, y, tick_label=labels, width=0.6, color=["blue", "red"])

# Add annotations
plt.annotate(
    str(non_def_perc) + " %",
    xy=(0, non_defs),
    xytext=(0, non_defs * 0.75),
    ha="center",
    size=12,
)
plt.annotate(
    str(def_perc) + " %",
    xy=(1, defs),
    xytext=(1, defs * 0.75),
    ha="center",
    size=12,
)
plt.title("Distribution of defaulters vs non-defaulters", size=14)
plt.ylabel("Number of customers")
plt.show()

#This code creates a bar chart using the plt.bar() function, 
# where the height of each bar represents the number of customers in each category.
# It also adds annotations to the bars indicating the percentage of customers in each category. 
# Finally, it sets the title and y-axis label for the plot, and displays it using the plt.show() function.


X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3, random_state=123)
log_model=LogisticRegression()
log_model.fit(X_train, Y_train)
y_pred_log = log_model.predict(X_test)

SVC_model=SVC()
SVC_model.fit(X_train, Y_train)
y_pred_SVC = SVC_model.predict(X_test)

DTmodel=DecisionTreeClassifier(criterion="entropy", random_state=0)
DTmodel.fit(X_train, Y_train)
y_pred_DT=DTmodel.predict(X_test)

RFT_model=RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
RFT_model.fit(X_train, Y_train)
y_pred_RFT= RFT_model.predict(X_test)

plot_confusion_matrix(DTmodel, X_test, Y_test)
plt.title("Decision Tree confusion Matrix")

plot_confusion_matrix(log_model, X_test, Y_test)
plt.title("Logistic Regression confusion Matrix")

plot_confusion_matrix(SVC_model, X_test, Y_test)
plt.title("SVC confusion Matrix")


plot_confusion_matrix(RFT_model, X_test, Y_test)
plt.title("Random Forest Classifier confusion Matrix")
plt.show()

models={LogisticRegression(): 'Logistic Regression', 
        SVC(): 'Support Vector Machine',
        DecisionTreeClassifier(criterion="entropy", random_state=0):'Decision Tree Classifier', 
        RandomForestClassifier(n_estimators=10 ,criterion="entropy", random_state=0 ): 'Random Forest Classifier'}

for model in models.keys():
    model.fit(X_train, Y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get confusion matrix
    test_matrix=confusion_matrix(Y_test, y_test_pred)
    print(test_matrix)

    # Get F1 score
    f1_test = f1_score(Y_test, y_test_pred)
    print(f1_test)

for model,name in models.items():
    print(name+ ": {: .4f}%".format(model.score(X_test, Y_test)*100))

import pickle
#save the model
f=None
try:
    f=open("CCdefaulter_model", "wb")
    pickle.dump(log_model, f)
except Exception as e:
    print("issue", e)

finally:
    if f is not None:
        f.close()
