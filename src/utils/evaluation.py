from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import Counter
import numpy as np
import time

class MutliClassEval:
    def __init__(self, preds, labels):
        self.preds, self.labels = preds, labels
        self.num_classes = len(preds[0])
        self.class_perfs = self.one_vs_all_eval()
        self.thresholds = np.array([act.op_point[0] for act in self.class_perfs])

    def one_vs_all_eval(self):
        classes = []
        for k in range(self.num_classes):
            class_preds, class_labels = single_class(self.preds, self.labels, k)
            classes.append(BinaryEval(class_preds, class_labels))
        return classes

    def max_prob_eval(self):
        class_array = np.zeros([2, self.num_classes])
        for pred, lab in zip(self.preds, self.labels):
            class_array[0, lab] += (max(pred) == pred[lab])
            class_array[1, lab] += 1
        acc = np.sum(class_array[0])/np.sum(class_array[1])
        class_acc = class_array[0]/class_array[1]
        avg_acc = np.nanmean(class_acc)
        print(f'accuracy using max prob: {acc:.3f}')
        
    def op_point_eval(self):
        hits_dist, score_thresh = [], 0
        for pred, lab in zip(self.preds, self.labels):
            hits = np.array(pred) > self.thresholds
            hits_dist.append(sum(hits))
            if sum(hits) > 0:
                score_thresh += hits[lab]/sum(hits)
        
        print(f'accuracy using 1 vs all: {score_thresh/len(self.preds):.3f}')
        print(Counter(hits_dist))
        sns.histplot(hits_dist, binwidth=1, binrange=[-0.5, 4])
        plt.show()

    def plot_class_curves(self):
        sns.set_theme()
        precisions = np.array([act.op_point[2] for act in self.class_perfs])
        recalls = np.array([act.op_point[3] for act in self.class_perfs])
        frequencies = np.array([sum(act.labels) for act in self.class_perfs])
        colors = cm.rainbow(np.linspace(0, 1, len(self.class_perfs)))

        ax = iso_F1()
        ax.scatter(precisions, recalls, color=colors, marker='x')
        plt.show()
    
        fig, ax = plt.subplots(figsize=(10,0.5))
        ax.scatter(frequencies, np.zeros(len(self.class_perfs)), color=colors, marker='x')
        ax.get_yaxis().set_ticks([])
        ax.set_xlabel('frequency of training points')
        ax.set_xscale("log")
        ax.set_xlim([0.5,2*max(frequencies)])
        plt.show()

    def __getitem__(self, k):
        return self.class_perfs[k]
        
def single_class(preds, labels, k):
    class_preds, class_labels = [], []
    for pred, lab in zip(preds, labels):
        class_preds.append(pred[k])
        class_labels.append(int(lab == k))
    return class_preds, class_labels

class BinaryEval:
    def __init__(self, preds, labels):
        self.preds = preds
        self.labels = labels

    def plot_PR(self):
        precision, recall, thresholds = self.PR()
        ax = iso_F1()
        ax.plot(recall, precision)

        threshold_markings = []
        for p,r,t in zip(precision, recall, thresholds):
            if all((p-p_2)**2 + (r-r_2)**2 > 0.04  for (p_2, r_2) in threshold_markings):
                threshold_markings.append((p,r))
                ax.text(r+0.02, p+0.005, f'{t:.3f}', color='r')
                ax.plot(r, p, 'rx')

    def plot_calib(self):
        prob_pred, prob_true = self.calibration()
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
        return (prob_pred, prob_true)
    
    @property
    def ECE(self):
        prob_pred, prob_true = self.calibration()
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

    def __repr__(self):
        T, F1, P, R = self.op_point
        output = f'T:{T:.3f}   F1:{F1:.3f}   P:{P:.3f}   R:{R:.3f}'
        return output
        
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
