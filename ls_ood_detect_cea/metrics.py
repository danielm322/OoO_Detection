from typing import Union, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import torchmetrics.functional as tmf
import seaborn as sns


def get_hz_detector_results(
    detect_exp_name: str,
    ind_samples_scores: np.ndarray,
    ood_samples_scores: np.ndarray,
    return_results_for_mlflow: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    labels_ind_test = np.ones((ind_samples_scores.shape[0], 1))  # postive class
    labels_ood_test = np.zeros((ood_samples_scores.shape[0], 1))  # negative class

    ind_samples_scores = np.expand_dims(ind_samples_scores, 1)
    ood_samples_scores = np.expand_dims(ood_samples_scores, 1)

    scores = np.vstack((ind_samples_scores, ood_samples_scores))
    labels = np.vstack((labels_ind_test, labels_ood_test))
    labels = labels.astype("int32")

    results_table = pd.DataFrame(
        columns=[
            "experiment",
            "auroc",
            "fpr@95",
            "aupr",
            "fpr",
            "tpr",
            "roc_thresholds",
            "precision",
            "recall",
            "pr_thresholds",
        ]
    )

    roc_auc = tmf.auroc(torch.from_numpy(scores), torch.from_numpy(labels))

    fpr, tpr, roc_thresholds = tmf.roc(torch.from_numpy(scores), torch.from_numpy(labels))

    fpr_95 = fpr[torch.where(tpr >= 0.95)[0][0]]

    precision, recall, pr_thresholds = tmf.precision_recall_curve(torch.from_numpy(scores), torch.from_numpy(labels))
    aupr = auc(recall.numpy(), precision.numpy())

    results_table = results_table.append(
        {
            "experiment": detect_exp_name,
            "auroc": roc_auc.item(),
            "fpr@95": fpr_95.item(),
            "aupr": aupr,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_thresholds": roc_thresholds.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "pr_thresholds": pr_thresholds.tolist(),
        },
        ignore_index=True,
    )

    results_table.set_index("experiment", inplace=True)

    # print("AUROC: {:0.4f}".format(results_table['auroc'][0].item()))
    # print("FPR95: {:0.4f}".format(results_table['fpr@95'][0].item()))
    # print("AUPR: {:0.4f}".format(results_table['aupr'][0].item()))
    if not return_results_for_mlflow:
        return results_table
    else:
        results_for_mlflow = results_table.loc[detect_exp_name, ["auroc", "fpr@95", "aupr"]].to_dict()
        # MLFlow doesn't accept the character '@'
        results_for_mlflow["fpr_95"] = results_for_mlflow.pop("fpr@95")
        return results_table, results_for_mlflow


def get_ood_detector_results(classifier_name: str, classifier_ood, samples_test_ds, labels_test_ds) -> pd.DataFrame:
    """
    Calculates metrics for OoD detection
    :rtype: pd.DataFrame
    """
    kde_class_models = {classifier_name: classifier_ood}
    datasets = {classifier_name: samples_test_ds}
    labels = {classifier_name: labels_test_ds}
    preds = {classifier_name: [None, None]}

    for cls in kde_class_models:
        # print(cls)
        preds[cls][0] = kde_class_models[cls].pred_prob(datasets[cls])[:, 1]  # pred probs
        preds[cls][1] = kde_class_models[cls].predict(datasets[cls])  # pred class

    results_table = pd.DataFrame(columns=["classifiers", "fpr", "tpr", "auroc", "acc", "mcc", "f1", "fpr@95"])

    for cls_type in preds:
        print(cls_type)
        # sklearn
        fpr, tpr, thresholds = roc_curve(labels[cls_type], preds[cls_type][0])
        auc = roc_auc_score(labels[cls_type], preds[cls_type][0])
        # torch-metrics
        roc_auc = tmf.auroc(
            torch.from_numpy(preds[cls_type][0]),
            torch.from_numpy(labels[cls_type]),
            thresholds=20,
        )

        fpr, tpr, thresholds = tmf.roc(torch.from_numpy(preds[cls_type][0]), torch.from_numpy(labels[cls_type]))

        fpr_95 = fpr[torch.where(tpr >= 0.95)[0][0]]

        acc = tmf.accuracy(torch.from_numpy(preds[cls_type][0]), torch.from_numpy(labels[cls_type]))

        mcc = tmf.matthews_corrcoef(torch.from_numpy(preds[cls_type][0]), torch.from_numpy(labels[cls_type]), 2)

        f1 = tmf.f1_score(torch.from_numpy(preds[cls_type][0]), torch.from_numpy(labels[cls_type]))
        # results table
        results_table = results_table.append(
            {
                "classifiers": cls_type,
                "fpr": fpr,
                "tpr": tpr,
                "acc": acc,
                "mcc": mcc,
                "auroc": roc_auc,
                "fpr@95": fpr_95,
                "f1": f1,
            },
            ignore_index=True,
        )

    # Set name of the classifiers as index labels
    results_table.set_index("classifiers", inplace=True)
    return results_table


def plot_roc_ood_detector(results_table, legend_title: str = "Legend Title", plot_title: str = "Plot Title"):
    fig = plt.figure(figsize=(8, 6))
    for i in results_table.index:
        # print(i)
        plt.plot(
            results_table.loc[i]["fpr"],
            results_table.loc[i]["tpr"],
            label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
        )

    plt.plot([0, 1], [0, 1], color="orange", linestyle="--")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(plot_title, fontweight="bold", fontsize=15)
    plt.legend(prop={"size": 12}, loc="lower right")
    plt.show()


def save_roc_ood_detector(results_table: pd.DataFrame, plot_title: str = "Plot Title") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in results_table.index:
        ax.plot(
            results_table.loc[i]["fpr"],
            results_table.loc[i]["tpr"],
            label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
        )

    ax.plot([0, 1], [0, 1], color="orange", linestyle="--")
    ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_title(plot_title, fontweight="bold", fontsize=15)
    ax.legend(prop={"size": 12}, loc="lower right")
    return fig


def plot_auprc_ood_detector(
    results_table: pd.DataFrame,
    legend_title: str = "Legend Title",
    plot_title: str = "Plot Title",
):
    fig = plt.figure(figsize=(8, 6))
    for i in results_table.index:
        print(i)
        plt.plot(
            results_table.loc[i]["recall"],
            results_table.loc[i]["precision"],
            label=legend_title + ", AUPRC={:.4f}".format(results_table.loc[i]["aupr"]),
        )

    plt.plot([0, 1], [0, 1], color="orange", linestyle="--")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision", fontsize=15)
    plt.title(plot_title, fontweight="bold", fontsize=15)
    plt.legend(prop={"size": 12}, loc="lower right")
    plt.show()


def save_scores_plots(scores_gtsrb, scores_gtsrb_anomal, scores_stl10, scores_cifar10):
    df_scores_gtsrb = pd.DataFrame(scores_gtsrb, columns=["Entropy score"])
    df_scores_gtsrb_anomal = pd.DataFrame(scores_gtsrb_anomal, columns=["Entropy score"])
    df_scores_stl10 = pd.DataFrame(scores_stl10, columns=["Entropy score"])
    df_scores_cifar10 = pd.DataFrame(scores_cifar10, columns=["Entropy score"])

    df_scores_gtsrb.insert(0, "Dataset", "")
    df_scores_gtsrb.loc[:, "Dataset"] = "gtsrb"

    df_scores_gtsrb_anomal.insert(0, "Dataset", "")
    df_scores_gtsrb_anomal.loc[:, "Dataset"] = "gtsrb-anomal"

    df_scores_stl10.insert(0, "Dataset", "")
    df_scores_stl10.loc[:, "Dataset"] = "stl10"

    df_scores_cifar10.insert(0, "Dataset", "")
    df_scores_cifar10.loc[:, "Dataset"] = "cifar10"

    df_h_z_valid_scores = pd.concat([df_scores_gtsrb, df_scores_stl10, df_scores_cifar10]).reset_index(drop=True)
    gsc = sns.displot(df_h_z_valid_scores, x="Entropy score", hue="Dataset", kind="hist", fill=True)

    df_h_z_valid_scores = pd.concat([df_scores_gtsrb, df_scores_gtsrb_anomal]).reset_index(drop=True)
    gga = sns.displot(df_h_z_valid_scores, x="Entropy score", hue="Dataset", kind="hist", fill=True)

    df_h_z_valid_scores = pd.concat([df_scores_gtsrb, df_scores_cifar10]).reset_index(drop=True)
    gc = sns.displot(df_h_z_valid_scores, x="Entropy score", hue="Dataset", kind="hist", fill=True)

    df_h_z_valid_scores = pd.concat([df_scores_gtsrb, df_scores_stl10]).reset_index(drop=True)
    gs = sns.displot(df_h_z_valid_scores, x="Entropy score", hue="Dataset", kind="hist", fill=True)
    return gsc, gga, gc, gs


def get_pred_scores_plots_gtsrb(
    ind_gtsrb_pred_score, gtsrb_anomal_pred_score, stl10_pred_score, cifar10_pred_score, title: str, x_axis_name: str
):
    df_pred_h_scores_gtsrb = pd.DataFrame(ind_gtsrb_pred_score, columns=[x_axis_name])
    df_pred_h_scores_gtsrb_anomal = pd.DataFrame(gtsrb_anomal_pred_score, columns=[x_axis_name])
    df_pred_h_scores_stl10 = pd.DataFrame(stl10_pred_score, columns=[x_axis_name])
    df_pred_h_scores_cifar10 = pd.DataFrame(cifar10_pred_score, columns=[x_axis_name])

    df_pred_h_scores_gtsrb.insert(0, "Dataset", "")
    df_pred_h_scores_gtsrb.loc[:, "Dataset"] = "gtsrb"

    df_pred_h_scores_gtsrb_anomal.insert(0, "Dataset", "")
    df_pred_h_scores_gtsrb_anomal.loc[:, "Dataset"] = "gtsrb-anomal"

    df_pred_h_scores_stl10.insert(0, "Dataset", "")
    df_pred_h_scores_stl10.loc[:, "Dataset"] = "stl10"

    df_pred_h_scores_cifar10.insert(0, "Dataset", "")
    df_pred_h_scores_cifar10.loc[:, "Dataset"] = "cifar10"

    df_pred_h_scores = pd.concat(
        [df_pred_h_scores_gtsrb, df_pred_h_scores_gtsrb_anomal, df_pred_h_scores_stl10, df_pred_h_scores_cifar10]
    ).reset_index(drop=True)

    ax = sns.displot(df_pred_h_scores, x=x_axis_name, hue="Dataset", kind="hist", fill=True).set(title=title)
    return ax
