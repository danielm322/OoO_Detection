# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez
#    Daniel Montoya

from typing import Union, Tuple
import numpy as np
import torch
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc
import torchmetrics.functional as tmf
import seaborn as sns

from .uncertainty_estimation import LaREMPostprocessor, LaREDPostprocessor


def get_hz_detector_results(
    detect_exp_name: str,
    ind_samples_scores: np.ndarray,
    ood_samples_scores: np.ndarray,
    return_results_for_mlflow: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Calculates the metrics relevant OoD detection: AUROC, FPR, AUPR, TPR, precision, recall,
    and classification thresholds. Can optionally format results for mlflow logging (no @ allowed).
    Automatically inverts labels if AUROC<0.5.

    Args:
        detect_exp_name: Name of the current experiment. This will be of the name of the row
         of the returned pandas df
        ind_samples_scores: Array of InD scores
        ood_samples_scores: Array of OoD scores
        return_results_for_mlflow: Optionally return AUROC, FPR and AUPR formatted for mlflow
         logging

    Returns:
        (pd.Dataframe): Results in a pandas dataframe format and optionally a dictionary with
            results for mlflow
    """
    assert isinstance(detect_exp_name, str), "detect_exp_name must be a string"
    assert isinstance(ind_samples_scores, np.ndarray), "ind_samples_scores must be a numpy array"
    assert isinstance(ood_samples_scores, np.ndarray), "ood_samples_scores must be a numpy array"
    assert isinstance(
        return_results_for_mlflow, bool
    ), "return_results_for_mlflow must be a boolean"
    labels_ind_test = np.ones((ind_samples_scores.shape[0], 1))  # positive class
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
    # Make sure of always getting a positive ROC curve, by inverting the labels
    if roc_auc < 0.5:
        labels_ind_test = np.zeros((ind_samples_scores.shape[0], 1))  # positive class
        labels_ood_test = np.ones((ood_samples_scores.shape[0], 1))  # negative class
        labels = np.vstack((labels_ind_test, labels_ood_test))
        labels = labels.astype("int32")
        roc_auc = tmf.auroc(torch.from_numpy(scores), torch.from_numpy(labels))

    fpr, tpr, roc_thresholds = tmf.roc(torch.from_numpy(scores), torch.from_numpy(labels))

    fpr_95 = fpr[torch.where(tpr >= 0.95)[0][0]]

    precision, recall, pr_thresholds = tmf.precision_recall_curve(
        torch.from_numpy(scores), torch.from_numpy(labels)
    )
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

    if not return_results_for_mlflow:
        return results_table
    else:
        results_for_mlflow = results_table.loc[
            detect_exp_name, ["auroc", "fpr@95", "aupr"]
        ].to_dict()
        # MLFlow doesn't accept the character '@'
        results_for_mlflow["fpr_95"] = results_for_mlflow.pop("fpr@95")
        return results_table, results_for_mlflow


def plot_roc_ood_detector(results_table, plot_title: str = "Plot Title"):
    """
    Plot ROC curve from the results table from the function get_hz_detector_results.

    Args:
        results_table: Pandas table obtained with the get_hz_detector_results function
        plot_title: Title of the plot

    """
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


def save_roc_ood_detector(
    results_table: pd.DataFrame, plot_title: str = "Plot Title"
) -> plt.Figure:
    """
    Returns a ROC plot figure that can be saved or logged with mlflow. Does not display any
    figure to screen

    Args:
        results_table (pd.Dataframe): Dataframe with results as rows and experiments names as
            indexes
        plot_title (str): Title of the plot

    Returns:
        (plt.Figure): A figure to be saved or logged with mlflow
    """
    assert isinstance(results_table, pd.DataFrame), "results_table must be a pandas dataframe"
    assert isinstance(plot_title, str), "plot_title must be a string"
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in results_table.index:
        if "LaRED" in i or "LaREM" in i:
            ax.plot(
                results_table.loc[i]["fpr"],
                results_table.loc[i]["tpr"],
                label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
                linestyle="solid",
                linewidth=3.0,
            )
        else:
            ax.plot(
                results_table.loc[i]["fpr"],
                results_table.loc[i]["tpr"],
                label=i + ", AUROC={:.4f}".format(results_table.loc[i]["auroc"]),
                linestyle="dashed",
                linewidth=1.7,
            )

    ax.plot([0, 1], [0, 1], color="orange", linestyle="--")
    ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_title(plot_title, fontweight="bold", fontsize=15)
    ax.legend(prop={"size": 12}, loc="lower right")
    return fig


def save_scores_plots(
    scores_ind: np.ndarray,
    ood_lared_scores_dict: dict,
    ood_datasets_list: list,
    ind_dataset_name: str,
) -> dict:
    """
    InD and OoD agnostic function that takes as input the InD numpy ndarray with the LaRED scores,
    a dictionary of OoD LaRED scores, a list of the names of the OoD dataset, and the name of the
    InD dataset, and returns a histogram of pairwise comparisons, that can be saved to a
    file, logged with mlflow, or shown in screen

    Args:
        scores_ind: InD LaRED scores as numpy ndarray
        ood_lared_scores_dict: Dictionary keys as ood datasets names and values as ndarrays of
            LaRED scores per each
        ood_datasets_list: List of OoD datasets names
        ind_dataset_name: String with the name of the InD dataset

    Returns:
        Dictionary of plots where the keys are the plot names and the values are the figures
    """
    assert isinstance(scores_ind, np.ndarray), "scores_ind must be a numpy array"
    assert isinstance(ood_lared_scores_dict, dict), "ood_lared_scores_dict must be a dictionary"
    assert hasattr(ood_datasets_list, "__iter__"), "ood_datasets_list must be an iterable"
    assert all(isinstance(item, str) for item in ood_datasets_list), (
        "ood_datasets_list items must" " be strings"
    )
    assert isinstance(ind_dataset_name, str), "ind_dataset_name must be a string"
    df_scores_ind = pd.DataFrame(scores_ind, columns=["Entropy score"])
    df_scores_ind.insert(0, "Dataset", "")
    df_scores_ind.loc[:, "Dataset"] = ind_dataset_name
    ood_df_dict = {}
    for ood_dataset_name in ood_datasets_list:
        ood_df_dict[ood_dataset_name] = pd.DataFrame(
            ood_lared_scores_dict[ood_dataset_name], columns=["Entropy score"]
        )
        ood_df_dict[ood_dataset_name].insert(0, "Dataset", "")
        ood_df_dict[ood_dataset_name].loc[:, "Dataset"] = ood_dataset_name

    plots_dict = {}
    for ood_dataset_name in ood_datasets_list:
        df_h_z_scores = pd.concat([df_scores_ind, ood_df_dict[ood_dataset_name]]).reset_index(
            drop=True
        )
        plots_dict[ood_dataset_name + "_lared_scores"] = sns.displot(
            df_h_z_scores, x="Entropy score", hue="Dataset", kind="hist", fill=True
        )

    return plots_dict


def get_pred_scores_plots(
    experiment: dict, ood_datasets_list: list, title: str, ind_dataset_name: str
):
    """
    Function that takes as input an experiment dictionary (one classification technique), a list
    of ood datasets, a plot title, and the InD dataset name and returns a plot of the predictive
    score density

    Args:
        experiment: Dictionary with keys 'InD':ndarray, 'x_axis':str, and 'plot_name':str and other
            keys as ood dataset names with values as ndarray
        ood_datasets_list: List with OoD datasets names
        title: Title of the plot
        ind_dataset_name: String with the name of the InD dataset

    Returns:
        Figure with the density scores of the InD and the OoD datasets
    """
    assert isinstance(experiment, dict)
    assert hasattr(ood_datasets_list, "__iter__"), "ood_datasets_list must be an iterable"
    assert all(isinstance(item, str) for item in ood_datasets_list), (
        "ood_datasets_list items must" " be strings"
    )
    assert isinstance(title, str)
    assert isinstance(ind_dataset_name, str)
    df_pred_h_scores_ind = pd.DataFrame(experiment["InD"], columns=[experiment["x_axis"]])
    df_pred_h_scores_ind.insert(0, "Dataset", "")
    df_pred_h_scores_ind.loc[:, "Dataset"] = ind_dataset_name
    ood_df_dict = {}
    for ood_dataset_name in ood_datasets_list:
        ood_df_dict[ood_dataset_name] = pd.DataFrame(
            experiment[ood_dataset_name], columns=[experiment["x_axis"]]
        )
        ood_df_dict[ood_dataset_name].insert(0, "Dataset", "")
        ood_df_dict[ood_dataset_name].loc[:, "Dataset"] = ood_dataset_name

    all_dfs = [df_pred_h_scores_ind]
    all_dfs.extend(list(ood_df_dict.values()))
    df_pred_h_scores = pd.concat(all_dfs).reset_index(drop=True)

    ax = sns.displot(
        df_pred_h_scores, x=experiment["x_axis"], hue="Dataset", kind="hist", fill=True
    ).set(title=title)
    return ax


def log_evaluate_lared_larem(
    ind_train_h_z: np.ndarray,
    ind_test_h_z: np.ndarray,
    ood_h_z_dict: dict,
    experiment_name_extension: str = "",
    return_density_scores: bool = False,
    log_step: Union[int, None] = None,
    mlflow_logging: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray, dict]]:
    """
    Function that takes as input InD numpy arrays of entropies and one dictionary for all OoD
    datasets and returns LaRED and LaREM results in the form of a pandas dataframe.
    Optionally logs to a running mlflow experiment.

    Args:
        ind_train_h_z: InD training samples for both LaRED and LaREM as numpy ndarray
        ind_test_h_z: InD test samples as numpy ndarray
        ood_h_z_dict: OoD dictionary where keys are the OoD datasets and the values are the
            ndarrays of entropies
        experiment_name_extension: Extra string to add to the default experiment name, useful for
            PCA experiments
        return_density_scores: return LaRED density scores for further analysis
        log_step: optional step useful for PCA experiments. None if not performing PCA with
            several components
        mlflow_logging: Optionally log to an existing mlflow run

    Returns:
        Pandas dataframe with results, optionally LaRED density score
    """
    assert isinstance(ind_train_h_z, np.ndarray)
    assert isinstance(ind_test_h_z, np.ndarray)
    assert isinstance(ood_h_z_dict, dict)
    assert isinstance(experiment_name_extension, str)
    assert isinstance(return_density_scores, bool)
    if log_step is not None:
        assert isinstance(log_step, int), "log_step is either None or an integer"
    assert isinstance(mlflow_logging, bool)

    # Initialize df to store all the results
    overall_metrics_df = pd.DataFrame(
        columns=[
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
    ######################################################
    # Evaluate OoD detection method LaRED
    ######################################################
    lared_ds_shift_detector = LaREDPostprocessor()
    lared_ds_shift_detector.setup(ind_train_h_z)
    ind_lared_score = lared_ds_shift_detector.postprocess(ind_test_h_z)
    ood_lared_scores_dict = {}
    for dataset_name, ood_dataset in ood_h_z_dict.items():
        ood_lared_scores_dict[dataset_name] = lared_ds_shift_detector.postprocess(ood_dataset)

    ######################################################
    # Evaluate OoD detection method LaREM
    ######################################################
    larem_ds_shift_detector = LaREMPostprocessor()
    larem_ds_shift_detector.setup(ind_train_h_z)
    ind_larem_score = larem_ds_shift_detector.postprocess(ind_test_h_z)
    ood_larem_scores_dict = {}
    for dataset_name, ood_dataset in ood_h_z_dict.items():
        ood_larem_scores_dict[dataset_name] = larem_ds_shift_detector.postprocess(ood_dataset)

    #########################
    # Prepare logging of results
    lared_larem_experiments = {}
    for dataset_name, ood_dataset in ood_h_z_dict.items():
        lared_larem_experiments[f"{dataset_name} LaRED"] = {
            "InD": ind_lared_score,
            "OoD": ood_lared_scores_dict[dataset_name],
        }
        lared_larem_experiments[f"{dataset_name} LaREM"] = {
            "InD": ind_larem_score,
            "OoD": ood_larem_scores_dict[dataset_name],
        }

    # Log Results
    for experiment_name, experiment in lared_larem_experiments.items():
        experiment_name = experiment_name + experiment_name_extension
        r_df, r_mlflow = get_hz_detector_results(
            detect_exp_name=experiment_name,
            ind_samples_scores=experiment["InD"],
            ood_samples_scores=experiment["OoD"],
            return_results_for_mlflow=True,
        )
        # Add OoD dataset to metrics name
        if "PCA" in experiment_name:
            r_mlflow = dict(
                [(f"{' '.join(experiment_name.split()[:-1])}_{k}", v) for k, v in r_mlflow.items()]
            )
        else:
            r_mlflow = dict([(f"{experiment_name}_{k}", v) for k, v in r_mlflow.items()])
        if mlflow_logging:
            mlflow.log_metrics(r_mlflow, step=log_step)
        overall_metrics_df = overall_metrics_df.append(r_df)

    if return_density_scores:
        return overall_metrics_df, ind_lared_score, ood_lared_scores_dict
    else:
        return overall_metrics_df


def select_and_log_best_lared_larem(
    overall_metrics_df: pd.DataFrame,
    n_pca_components_list: list,
    technique: str,
    log_mlflow: bool = False,
) -> Tuple[float, float, float, int]:
    """
    Takes as input a Dataframe with the columns 'auroc', 'aupr' and 'fpr@95', a list of PCA number
    of components, and the name of the technique: either 'LaRED' or 'LaREM', and logs to and
    existing mlflow run the best metrics

    Args:
        overall_metrics_df: Pandas DataFrame with the LaRED or LaREM experiments results
        n_pca_components_list: List with the numbers of PCA components
        technique: Either 'LaRED' or 'LaREM'
        log_mlflow: Log to mlflow boolean flag

    Returns:
        Tuple with the best auroc, aupr, fpr and the N components.
    """
    assert isinstance(overall_metrics_df, pd.DataFrame)
    assert hasattr(n_pca_components_list, "__iter__")
    assert isinstance(log_mlflow, bool)
    assert technique in ("LaRED", "LaREM"), f"Got {technique}"
    means_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
    stds_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
    temp_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
    # Calculate mean of no PCA run
    for row_name in overall_metrics_df.index:
        if technique in row_name and "anomalies" not in row_name and "PCA" not in row_name:
            temp_df = temp_df.append(overall_metrics_df.loc[row_name, ["auroc", "fpr@95", "aupr"]])
    means_temp_df = temp_df.mean()
    stds_temp_df = temp_df.std()
    means_df = means_df.append(pd.DataFrame(dict(means_temp_df), index=[technique]))
    stds_df = stds_df.append(pd.DataFrame(dict(stds_temp_df), index=[technique]))
    # Calculate means of PCA runs
    for n_components in n_pca_components_list:
        temp_df = pd.DataFrame(columns=["auroc", "fpr@95", "aupr"])
        for row_name in overall_metrics_df.index:
            if (
                technique in row_name
                and "anomalies" not in row_name
                and f"PCA {n_components}" in row_name
            ):
                temp_df = temp_df.append(
                    overall_metrics_df.loc[row_name, ["auroc", "fpr@95", "aupr"]]
                )
        means_temp_df = temp_df.mean()
        stds_temp_df = temp_df.std()
        means_df = means_df.append(
            pd.DataFrame(dict(means_temp_df), index=[f"{technique} PCA {n_components}"])
        )
        stds_df = stds_df.append(
            pd.DataFrame(dict(stds_temp_df), index=[f"{technique} PCA {n_components}"])
        )

    best_index = means_df[means_df.auroc == means_df.auroc.max()].index[0]
    # Here we assume the convention that 0 PCA components would mean the no PCA case
    if "PCA" in best_index:
        best_n_comps = int(best_index.split()[-1])
    else:
        best_n_comps = 0

    if log_mlflow:
        mlflow.log_metric(f"{technique}_auroc_mean", means_df.loc[best_index, "auroc"])
        mlflow.log_metric(f"{technique}_auroc_std", stds_df.loc[best_index, "auroc"])
        mlflow.log_metric(f"{technique}_aupr_mean", means_df.loc[best_index, "aupr"])
        mlflow.log_metric(f"{technique}_aupr_std", stds_df.loc[best_index, "aupr"])
        mlflow.log_metric(f"{technique}_fpr95_mean", means_df.loc[best_index, "fpr@95"])
        mlflow.log_metric(f"{technique}_fpr95_std", stds_df.loc[best_index, "fpr@95"])
        mlflow.log_metric(f"Best {technique}", best_n_comps)
    return (
        means_df.loc[best_index, "auroc"],
        means_df.loc[best_index, "aupr"],
        means_df.loc[best_index, "fpr@95"],
        best_n_comps,
    )


baseline_name_dict = {
    "pred_h": {
        "plot_title": "Predictive H distribution",
        "x_axis": "Predictive H score",
        "plot_name": "pred_h",
    },
    "mi": {
        "plot_title": "Predictive MI distribution",
        "x_axis": "Predictive MI score",
        "plot_name": "pred_mi",
    },
    "msp": {
        "plot_title": "Predictive MSP distribution",
        "x_axis": "Predictive MSP score",
        "plot_name": "pred_msp",
    },
    "energy": {
        "plot_title": "Predictive energy score distribution",
        "x_axis": "Predictive energy score",
        "plot_name": "pred_energy",
    },
    "mdist": {
        "plot_title": "Mahalanobis Distance distribution",
        "x_axis": "Mahalanobis Distance score",
        "plot_name": "pred_mdist",
    },
    "knn": {
        "plot_title": "kNN distance distribution",
        "x_axis": "kNN Distance score",
        "plot_name": "pred_knn",
    },
    "ash": {
        "plot_title": "ASH score distribution",
        "x_axis": "ASH score",
        "plot_name": "ash_score",
    },
    "dice": {
        "plot_title": "DICE score distribution",
        "x_axis": "DICE score",
        "plot_name": "dice_score",
    },
    "react": {
        "plot_title": "ReAct score distribution",
        "x_axis": "ReAct score",
        "plot_name": "react_score",
    },
    "dice_react": {
        "plot_title": "DICE + ReAct score distribution",
        "x_axis": "DICE + ReAct score",
        "plot_name": "dice_react_score",
    },
    "filtered_energy": {
        "plot_title": "Predictive filtered energy score distribution",
        "x_axis": "Predictive filtered energy score",
        "plot_name": "pred_filtered_energy",
    },
    "filtered_ash": {
        "plot_title": "ASH filtered score distribution",
        "x_axis": "ASH filtered score",
        "plot_name": "filtered_ash_score",
    },
    "filtered_react": {
        "plot_title": "ReAct filtered score distribution",
        "x_axis": "Filtered ReAct score",
        "plot_name": "filtered_react_score",
    },
    "filtered_dice": {
        "plot_title": "DICE filtered score distribution",
        "x_axis": "DICE filtered score",
        "plot_name": "filtered_dice_score",
    },
    "filtered_dice_react": {
        "plot_title": "DICE + ReAct filtered score distribution",
        "x_axis": "DICE + ReAct filtered score",
        "plot_name": "filtered_dice_react_score",
    },
    "raw_energy": {
        "plot_title": "Predictive raw energy score distribution",
        "x_axis": "Predictive raw energy score",
        "plot_name": "pred_raw_energy",
    },
    "raw_ash": {
        "plot_title": "ASH raw score distribution",
        "x_axis": "ASH raw score",
        "plot_name": "raw_ash_score",
    },
    "raw_react": {
        "plot_title": "ReAct raw score distribution",
        "x_axis": "raw ReAct score",
        "plot_name": "raw_react_score",
    },
    "raw_dice": {
        "plot_title": "DICE raw score distribution",
        "x_axis": "DICE raw score",
        "plot_name": "raw_dice_score",
    },
    "raw_dice_react": {
        "plot_title": "DICE + ReAct raw score distribution",
        "x_axis": "DICE + ReAct raw score",
        "plot_name": "raw_dice_react_score",
    },
}
