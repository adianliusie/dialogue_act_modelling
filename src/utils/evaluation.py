from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import Counter
import numpy as np
import time

class MutliClassEval:
    def __init__(self, labels, preds):
        classes = self.one_vs_all_eval(labels, preds)
        self.thresholds = np.array([act.op_point[0] for act in classes])
        self.op_point_eval(labels, preds, self.thresholds)
        self.performance_curves(classes)
        
    def one_vs_all_eval(self, labels, preds):
        classes = []
        for k in range(len(preds[0])):
            class_labels, class_preds = single_class(labels, preds, k)
            classes.append(BinaryEval(class_labels, class_preds))
        return classes

    def op_point_eval(self, labels, preds, thresholds):
        hits_dist, score_base, score_thresh = [], 0, 0
        for pred, lab in zip(preds, labels):
            score_base += (max(pred) == pred[lab])
            hits = np.array(pred) > thresholds
            hits_dist.append(sum(hits))
            if sum(hits) > 0:
                score_thresh += hits[lab]/sum(hits)
        
        print(f'accuracy using max prediction: {score_base/len(preds):.3f}')
        print(f'accuracy using 1 vs all operating points: {score_thresh/len(preds):.3f}')
        sns.histplot(hits_dist, binwidth=1, binrange=[-0.5, 4])
        
        print(Counter(hits_dist))
        plt.show()

    def performance_curves(self, classes):
        sns.set_theme()
        precisions = np.array([act.op_point[2] for act in classes])
        recalls = np.array([act.op_point[3] for act in classes])
        frequencies = np.array([sum(act.labels) for act in classes])
        colors = cm.rainbow(np.linspace(0, 1, len(classes)))

        ax = iso_F1()
        ax.scatter(precisions, recalls, color=colors, marker='x')
        plt.show()
    
        fig, ax = plt.subplots(figsize=(10,0.5))
        ax.scatter(frequencies, np.zeros(len(classes)), color=colors, marker='x')
        ax.get_yaxis().set_ticks([])
        ax.set_xlabel('frequency of training points')
        ax.set_xscale("log")
        ax.set_xlim([0.5,2*max(frequencies)])
        plt.show()

    def __getitem__(self):
        pass
        
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

    def PR(self):
        precision, recall, thresholds = precision_recall_curve(self.labels, self.preds)
        return (precision, recall, thresholds)

    def calibration(self):
        prob_true, prob_pred = calibration_curve(self.labels, self.preds, n_bins=10, strategy='quantile')
        return (prob_true, prob_pred)
    
    @property
    def ECE(self):
        prob_true, prob_pred = self.calibration()
        """
        x = np.linspace(0,1,50)
        plt.plot(x, x)
        plt.plot(prob_pred, prob_true, 'x')
        plt.show()
        plt.xlim(0,1), plt.ylim(0,1.02)
        """
        return sum([abs(i-j) for i, j in zip(prob_true, prob_pred)])

    
    @property
    def op_point(self):
        precision, recall, thresholds = self.PR()
        F1 = [calc_F1(P, R) for P, R in zip(precision, recall)]
        operating_point = max(zip(thresholds, F1, precision, recall), key=lambda x: x[1])
        return operating_point

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
    ax.set_xlabel('Recall'), ax.set_ylabel('Precision') 
    return ax

def calc_F1(P, R):
    return (2*P*R)/(P+R)