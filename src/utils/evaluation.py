from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt

def one_vs_all_evaluation(labels, preds):
    classes = []
    for k in range(len(preds[0])):
        class_labels, class_preds = single_class(labels, preds, k)
        classes.append(BinaryEval(class_labels, class_preds))
        
    thresholds = np.array([act.op_point()[0] for act in classes])
    
    hits_dist, score = [], 0
    for pred, lab in zip(preds, labels):
        hits = np.array(pred) > thresholds
        hits_dist.append(sum(hits))
        score += (lab in hits)/sum(hits)
    acc = score/len(preds)
    
    return classes

def single_class(labels, preds, k):
    class_labels, class_preds = [], []
    for pred, lab in zip(preds, labels):
        class_preds.append(pred[k])
        class_labels.append(int(lab == k))
    return class_labels, class_preds

class BinaryEval:
    def __init__(self, labels, preds):
        self.labels = labels
        self.preds = preds
    
    def plot_PR(self):
        precision, recall, thresholds = self.PR()
        ax = iso_F1()
        ax.plot(recall, precision)
        ax.set_xlabel('Recall'), ax.set_ylabel('Precision')
        plt.show()
        
    def plot_calib(self):
        prob_true, prob_pred = self.calibration()
        fig, ax = plt.subplots()
        x = np.linspace(0,1,50)
        ax.plot(x, x)
        ax.plot(prob_pred, prob_true, 'x')
        ax.set_xlim(0,1), ax.set_ylim(0,1.02)
        ax.set_xlabel('Recall'), ax.set_ylabel('Precision')

    def op_point(self):
        precision, recall, thresholds = self.PR()
        F1 = [calc_F1(P, R) for P, R in zip(precision, recall)]
        operating_point = max(zip(thresholds, F1), key=lambda x: x[1])
        return operating_point
        
    def PR(self):
        precision, recall, thresholds = precision_recall_curve(self.labels, self.preds)
        return (precision, recall, thresholds)

    def calibration(self):
        prob_true, prob_pred = calibration_curve(self.labels, self.preds, n_bins=10, strategy='quantile')
        return (prob_true, prob_pred)

def iso_F1():
    fig, ax = plt.subplots()
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
        ax.set_xlim(0,1), ax.set_ylim(0,1.02)
    return ax

def calc_F1(P, R):
    return (2*P*R)/(P+R)
