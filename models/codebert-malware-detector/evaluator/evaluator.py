# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[idx] = int(label)
    return predictions

# def calculate_scores(answers,predictions):
#     Acc=[]
#     for key in answers:
#         if key not in predictions:
#             logging.error("Missing prediction for index {}.".format(key))
#             sys.exit()
#         Acc.append(answers[key]==predictions[key])

#     scores={}
#     scores['Acc']=np.mean(Acc)
#     return scores

# def calculate_scores(answers, predictions):
#     y_true = []
#     y_pred = []
#     for key in answers:
#         if key not in predictions:
#             logging.error(f"Missing prediction for index {key}.")
#             sys.exit()
#         y_true.append(answers[key])
#         y_pred.append(predictions[key])

#     scores = {
#         'Accuracy': accuracy_score(y_true, y_pred),
#         'Precision': precision_score(y_true, y_pred),
#         'Recall': recall_score(y_true, y_pred),
#         'F1-score': f1_score(y_true, y_pred)
#     }
#     return scores

def calculate_scores(answers, predictions):
    y_true = []
    y_pred = []
    for key in answers:
        if key not in predictions:
            logging.error(f"Missing prediction for index {key}.")
            sys.exit()
        y_true.append(answers[key])
        y_pred.append(predictions[key])

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()  # assumes binary classification

    scores = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'ConfusionMatrix': {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }
    }
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions)
    print(scores)

if __name__ == '__main__':
    main()
