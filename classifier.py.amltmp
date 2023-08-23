import pandas as pd
from sklearn.metrics.cluster import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from azureml.core import Workspace, Dataset

subscription_id = 'd5d1b5f7-5210-4d65-8f0c-730767acb30b'
resource_group = 'imdrkapil-rg'
workspace_name = 'mlworkspaceaug'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='iris1')
df= dataset.to_pandas_dataframe()
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'
x_train, x_test, y_train, y_test = train_test_split(df[features],df[target],test_size=0.1,shuffle=True)
classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model is {accuracy*100}")