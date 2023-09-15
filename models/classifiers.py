import time
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def decision_tree(train_data, testing_data, train_labels, **params_dict):

	start_time = time.time()
	
	#training
	clf = tree.DecisionTreeClassifier(**params_dict)
	clf = clf.fit(train_data, train_labels)

	end_time = time.time()
	train_time = end_time-start_time

	start_time = time.time()
	prediction = list(clf.predict(testing_data))
	end_time = time.time()
	infer_time = end_time-start_time

	return prediction, train_time, infer_time

def KNN(train_data, testing_data, train_labels, **params_dict):

	def drop_duplicate(data, labels):
		data_eli = []
		label_eli = []
		for idx, x in enumerate(data):
			if x not in data_eli:
				data_eli.append(x)
				label_eli.append(labels[idx])
		return data_eli, label_eli	
	start_time = time.time()
	train_data, train_labels_new = drop_duplicate(train_data, train_labels)

	#training
	clf = KNeighborsClassifier(**params_dict)
	clf.fit(train_data, train_labels_new)

	end_time = time.time()
	train_time = end_time-start_time

	start_time = time.time()
	prediction = []
	pre_dict = {}
	for idx, x in enumerate(testing_data):
		if str(x) not in pre_dict.keys():
			temp_prediction = list(clf.predict([x]))[0]
			pre_dict[str(x)] = temp_prediction
			prediction.append(temp_prediction)
		else:
			prediction.append(pre_dict[str(x)])
	end_time = time.time()
	infer_time = end_time-start_time	
	return prediction, train_time, infer_time

def MLP(train_data, testing_data, train_labels, **params_dict):
	start_time = time.time()
	
	#training
	clf = MLPClassifier(**params_dict)
	clf.fit(train_data, train_labels)

	end_time = time.time()
	train_time = end_time-start_time

	start_time = time.time()
	prediction = clf.predict(testing_data)
	end_time = time.time()
	infer_time = end_time-start_time
	return prediction, train_time, infer_time
