import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data_dict = pickle.load(open('data.pickle', 'rb'))

# print(data_dict.keys())
# print(data_dict)

data = np.asarray(data_dict['data'])
lables = np.asarray(data_dict['lables'])

x_train, x_test, y_train, y_test = train_test_split(data, lables, test_size=0.2, shuffle=True, stratify=lables)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)

print(score)

f = open('model.pickle', 'wb')
pickle.dump({'model': model}, f)
f.close()