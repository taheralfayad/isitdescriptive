import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import joblib

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred,  pos_label=1)
    f1 = metrics.f1_score(y_test, y_pred)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    labels = ['Descriptive (0)', 'Non-descriptive (1)']
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                  index=[f'Actual {label}' for label in labels], 
                                  columns=[f'Predicted {label}' for label in labels])

try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    X_resampled, y_resampled = joblib.load('resampled_data.pkl')
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
except FileNotFoundError:
    data = pd.read_csv('results.csv')
    data = data.dropna()
    X = data['line']
    y = data['label']

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    joblib.dump(vectorizer, 'vectorizer.pkl')

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
    joblib.dump((X_resampled, y_resampled), 'resampled_data.pkl')

    class_weights = {0: 1, 1: 3}
    model = SGDClassifier(loss='log_loss', random_state=42, class_weight=class_weights)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
    model.partial_fit(X_train, y_train, classes=np.unique(y_resampled))
    joblib.dump(model, 'model.pkl')

    y_pred = model.predict(X_test)
    evaluate_model(model, X_test, y_test)

def predict_and_update(text, actual_label):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    prediction_result = "Non-descriptive" if prediction[0] == 1 else "Descriptive"

    if actual_label is not None and actual_label == 0 or actual_label == 1:
        model.partial_fit(text_tfidf, [actual_label])
        joblib.dump(model, 'model.pkl')
        evaluate_model(model, X_test, y_test)
        actual_label = "Non-descriptive" if actual_label == 1 else "Descriptive"
        feedback = f"Model updated based on feedback. Prediction was: {prediction_result}, actual label was: {actual_label}"
    
    return prediction[0]


if __name__ == "__main__":
    prediction = sys.argv[1]
    actual_label = sys.argv[2]

    result = predict_and_update(prediction, int(actual_label))

    # Returns the prediction to the PHP wrapper
    print(int(result))
