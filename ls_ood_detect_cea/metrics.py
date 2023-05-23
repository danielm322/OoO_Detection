import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import torchmetrics.functional as tmf
import seaborn as sns
from icecream import ic


def get_hz_detector_results(detect_exp_name: str,
                            ind_samples_scores: np.ndarray,
                            ood_samples_scores: np.ndarray) -> pd.DataFrame:

    labels_ind_test = np.ones((ind_samples_scores.shape[0], 1)) # postive class
    labels_ood_test = np.zeros((ood_samples_scores.shape[0], 1)) # negative class
    
    ind_samples_scores = np.expand_dims(ind_samples_scores, 1)
    ood_samples_scores = np.expand_dims(ood_samples_scores, 1)
    
    scores = np.vstack((ind_samples_scores, ood_samples_scores))
    labels = np.vstack((labels_ind_test, labels_ood_test))
    
    results_table = pd.DataFrame(columns=['experiment', 'fpr', 'tpr', 'auc', 'acc', 'mcc', 'f1', 'fpr@95'])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    roc_auc = tmf.auroc(torch.from_numpy(scores),
                        torch.from_numpy(labels))
    
    fpr_95 = fpr[torch.where(tpr >= 0.95)[0][0]]
    
    acc = tmf.accuracy(torch.from_numpy(scores),
                       torch.from_numpy(labels))
    
    mcc = tmf.matthews_corrcoef(torch.from_numpy(scores),
                                torch.from_numpy(labels), num_classes=2)

    f1 = tmf.f1_score(torch.from_numpy(scores),
                      torch.from_numpy(labels),
                      task='binary')
    
    results_table = results_table.append({'experiment': detect_exp_name,
                                          'fpr': fpr,
                                          'tpr': tpr,
                                          'acc': acc,
                                          'mcc': mcc,
                                          'auc': roc_auc,
                                          'fpr@95': fpr_95,
                                          'f1': f1}, ignore_index=True)
    
    results_table.set_index('experiment', inplace=True)
    
    print("AUROC: {:0.4f}".format(results_table['auc'][0].item()))
    print("FPR95: {:0.4f}".format(results_table['fpr@95'][0].item()))
    print("F1: {:0.4f}".format(results_table['f1'][0].item()))

    return results_table


def get_ood_detector_results(classifier_name: str, classifier_ood, samples_test_ds, labels_test_ds) -> pd.DataFrame:
    """

    :rtype: pd.DataFrame
    """
    kde_class_models = {
        classifier_name: classifier_ood
    }
    datasets = {
        classifier_name: samples_test_ds
    }
    labels = {
        classifier_name: labels_test_ds
    }
    preds = {
        classifier_name: [None, None]
    }

    for cls in kde_class_models:
        # print(cls)
        preds[cls][0] = kde_class_models[cls].pred_prob(datasets[cls])[:, 1]  # pred probs
        preds[cls][1] = kde_class_models[cls].predict(datasets[cls])  # pred class

    results_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc', 'acc', 'mcc', 'f1', 'fpr@95'])

    for cls_type in preds:
        print(cls_type)
        # sklearn
        fpr, tpr, thresholds = roc_curve(labels[cls_type], preds[cls_type][0])
        auc = roc_auc_score(labels[cls_type], preds[cls_type][0])
        # torch-metrics
        roc_auc = tmf.auroc(torch.from_numpy(preds[cls_type][0]),
                            torch.from_numpy(labels[cls_type]), thresholds=20)

        fpr, tpr, thresholds = tmf.roc(torch.from_numpy(preds[cls_type][0]),
                                       torch.from_numpy(labels[cls_type]))

        fpr_95 = fpr[torch.where(tpr >= 0.95)[0][0]]

        acc = tmf.accuracy(torch.from_numpy(preds[cls_type][0]),
                           torch.from_numpy(labels[cls_type]))

        mcc = tmf.matthews_corrcoef(torch.from_numpy(preds[cls_type][0]),
                                    torch.from_numpy(labels[cls_type]), 2)

        f1 = tmf.f1_score(torch.from_numpy(preds[cls_type][0]),
                          torch.from_numpy(labels[cls_type]))
        # results table
        results_table = results_table.append({'classifiers': cls_type,
                                              'fpr': fpr,
                                              'tpr': tpr,
                                              'acc': acc,
                                              'mcc': mcc,
                                              'auc': roc_auc,
                                              'fpr@95': fpr_95,
                                              'f1': f1}, ignore_index=True)

    # Set name of the classifiers as index labels
    results_table.set_index('classifiers', inplace=True)
    return results_table


def plot_roc_ood_detector(results_table, legend_title: str = "Legend Title", plot_title: str = "Plot Title"):
    fig = plt.figure(figsize=(8, 6))
    # legend_title = "Woodscape vs Woodscape-Anomalies"
    for i in results_table.index:
        print(i)
        plt.plot(results_table.loc[i]['fpr'],
                 results_table.loc[i]['tpr'],
                 label=legend_title + ", AUROC={:.4f}".format(results_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(plot_title, fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 12}, loc='lower right')
    plt.show()
