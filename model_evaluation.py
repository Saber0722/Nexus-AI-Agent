import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Generate and print classification report
    cls_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(cls_report)
    
    return accuracy, cm, cls_report

# Assuming X_test and y_test are already defined
accuracy, cm, cls_report = evaluate_model(your_trained_model, X_test, y_test)

print(f"Model Accuracy: {accuracy}")