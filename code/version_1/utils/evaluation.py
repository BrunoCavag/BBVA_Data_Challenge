import copy as cp
from typing import Tuple

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, KFold, cross_val_score
from sklearn.metrics import confusion_matrix


def cross_validation(model, _X, _y, _cv=5):
  _scoring = ['accuracy', 'precision', 'recall', 'f1']
  results = cross_validate(estimator=model,
                            X=_X,
                            y=_y,
                            cv=_cv,
                            scoring=_scoring,
                            return_train_score=True)

  return {"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          }

def plot_validation_result(x_label, y_label, plot_title, train_data, val_data):
  # Set size of plot
  plt.figure(figsize=(12,6))
  labels = ["1st Fold", "2nd Fold", "3rd Fold"]
  X_axis = np.arange(len(labels))
  ax = plt.gca()
  plt.ylim(0.40000, 1)
  plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
  plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
  plt.title(plot_title, fontsize=30)
  plt.xticks(X_axis, labels)
  plt.xlabel(x_label, fontsize=14)
  plt.ylabel(y_label, fontsize=14)
  plt.legend()
  plt.grid(True)
  plt.show()

def cross_val_predict(model, kfold : KFold, X : np.array, y : np.array) \
                        -> Tuple[np.array, np.array, np.array]:
  model_ = cp.deepcopy(model)
  no_classes = len(np.unique(y))
  actual_classes = np.empty([0], dtype=int)
  predicted_classes = np.empty([0], dtype=int)
  predicted_proba = np.empty([0, no_classes]) 
  for train_ndx, test_ndx in kfold.split(X):
      train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
      actual_classes = np.append(actual_classes, test_y)
      model_.fit(train_X, train_y)
      predicted_classes = np.append(predicted_classes, model_.predict(test_X))
      try:
          predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
      except:
          predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)
  return actual_classes, predicted_classes, predicted_proba




def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array):
  #matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
  matrix = confusion_matrix(actual_classes, predicted_classes)
  plt.figure(figsize=(12.8,6))
  #sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
  sns.heatmap(matrix, annot=True, cmap="Blues", fmt="g")
  plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

  plt.show()