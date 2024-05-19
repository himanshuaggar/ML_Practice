#AND GATE
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = np.array([[1, 0], [0, 1], [0, 0], [1, 1], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 0, 0, 1, 1, 0, 0])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=12)

DTree = DecisionTreeClassifier()
DTree.fit(train_x, train_y)

pred_y = DTree.predict(test_x)

print("Test Value: ", test_y)
print("Predicted Value: ", pred_y)

print("Accuracy Score: {:.2f}%".format(accuracy_score(test_y, pred_y) * 100))


#OR GATE
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = np.array([[1, 0], [0, 1], [0, 0], [1, 1], [1, 1], [1, 0], [0, 1]])
y = np.array([1, 1, 0, 1, 1, 1, 1])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=12)

DTree = DecisionTreeClassifier()
DTree.fit(train_x, train_y)

pred_y = DTree.predict(test_x)

print("Test Value: ", test_y)
print("Predicted Value: ", pred_y)

print("Accuracy Score: {:.2f}%".format(accuracy_score(test_y, pred_y) * 100))


#XOR GATE
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 0], [0, 1], [1, 1], [0, 0]])
y = np.array([0, 1, 1, 0, 1, 1, 0, 0])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=12)

DTree = DecisionTreeClassifier()
DTree.fit(train_x, train_y)

pred_y = DTree.predict(test_x)

print("Test Value: ", test_y)
print("Predicted Value: ", pred_y)

print("Accuracy Score: {:.2f}%".format(accuracy_score(test_y, pred_y) * 100))