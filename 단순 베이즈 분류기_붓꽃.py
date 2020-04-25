import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
df.target = df.target.map({0:'setosa', 1:'versicolor', 2:'virginia'})

print(df.head())



X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
model = GaussianNB()
model.fit(X_train, y_train)

predicted_y = model.predict(X_test)
performance = metrics.classification_report(y_test, predicted_y)
print('performance = ', performance)
accuracy = accuracy_score(y_test, predicted_y)
print('accuracy = ', accuracy)


def showDist(r, c, idx, category, attr):
    data_df = df[df.target == category]
    plt.subplot(r, c, idx)
    ax = data_df[attr].plot(kind='hist')
    data_df[attr].plot(kind='kde', ax=ax, secondary_y = True, title = category+' '+attr, figsize = (8, 4))

showDist(2,2,1, 'setosa', 'sepal length (cm)')
showDist(2,2,2, 'setosa', 'sepal width (cm)')
showDist(2,2,3, 'setosa', 'petal length (cm)')
showDist(2,2,4, 'setosa', 'petal width (cm)')
plt.show()

showDist(2,2,1, 'vericolor', 'sepal length (cm)')
showDist(2,2,2, 'vericolor', 'sepal width (cm)')
showDist(2,2,3, 'vericolor', 'petal length (cm)')
showDist(2,2,4, 'vericolor', 'petal width (cm)')
plt.show()

showDist(2,2,1, 'virginia', 'sepal length (cm)')
showDist(2,2,2, 'virginia', 'sepal width (cm)')
showDist(2,2,3, 'virginia', 'petal length (cm)')
showDist(2,2,4, 'virginia', 'petal width (cm)')
plt.show()

