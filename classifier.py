import pandas as pd
from sklearn.metrics.cluster import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv('/content/mlops1/data/iris.csv')
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'
x_train, x_test, y_train, y_test = train_test_split(df[features],df[target],test_size=0.3,shuffle=True)
classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model is {accuracy*100}")