from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def train_simple_model_and_print_results(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=0, stratify=target)
    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
    pipe.fit(x_train, y_train)

    y_test_predicted = pipe.predict(x_test)

    print('Test accuracy:', accuracy_score(y_test, y_test_predicted))
    print('Test f1_score:', f1_score(y_test, y_test_predicted))
    print('Test confusion matrix:\n', confusion_matrix(y_test, y_test_predicted))

    return None
