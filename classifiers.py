from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

def classifier1():
# Results
# 165.5 Seconds to train SVC..
# Error on predict_proba
	svc = LinearSVC()
	clf = tree.DecisionTreeClassifier()

	eclf1 = VotingClassifier(estimators=[('ls', svc), ('dt', clf)], voting='soft', weights=[2,1])

	# Check the training time for the SVC
	t=time.time()
	eclf1.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')

	print('Test Accuracy of SVC = ', round(eclf1.score(X_test, y_test), 4))
	
def classifier2():
# Results
# 579.12 Seconds to train SVC...
# Test Accuracy of SVC =  0.9873
	svc = SVC(kernel="linear")
	svc.probability = True
	clf = tree.DecisionTreeClassifier()

	eclf1 = VotingClassifier(estimators=[('ls', svc), ('dt', clf)], voting='soft', weights=[2,1])

	# Check the training time for the SVC
	t=time.time()
	eclf1.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	pred = eclf1.predict(X_test)
	acc = accuracy_score(y_test, pred)
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(acc, 4))
	# Check the prediction time for a single sample
	t=time.time()
	
def classifier3():
# Results
# 221.45 Seconds to train SVC...
# Test Accuracy of SVC =  0.9738
	svc = SVC(kernel="linear")
	gnb = GaussianNB()

	eclf1 = VotingClassifier(estimators=[('ls', svc), ('gnb', clf)], voting='hard')

	# Check the training time for the SVC
	t=time.time()
	eclf1.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	pred = eclf1.predict(X_test)
	acc = accuracy_score(y_test, pred)
	print('Test Accuracy of SVC = ', round(accuracy_score(y_test, pred), 4))
	# Check the prediction time for a single sample
	t=time.time()
	
def classifier4()
	svc = SVC(kernel="linear")
	gnb = GaussianNB()

	eclf1 = VotingClassifier(estimators=[('ls', svc), ('gnb', clf)], voting='soft', weights=[2,1])

	# Check the training time for the SVC
	t=time.time()
	eclf1.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	pred = eclf1.predict(X_test)
	acc = accuracy_score(y_test, pred)
	print('Test Accuracy of SVC = ', round(accuracy_score(y_test, pred), 4))
	# Check the prediction time for a single sample
	t=time.time()