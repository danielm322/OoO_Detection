# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad


def get_msp_score(inputs, model, forward_func, method_args, logits=None):
    """
    taken from https://github.com/deeplearning-wisc/knn-ood/blob/master/util/score.py
    :param inputs:
    :type inputs:
    :param model:
    :type model:
    :param forward_func:
    :type forward_func:
    :param method_args:
    :type method_args:
    :param logits:
    :type logits:
    :return:
    :rtype:
    """
    if logits is None:
        with torch.no_grad():
            logits = forward_func(inputs, model)
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_godin_score(inputs, model, forward_func, method_args):
    """
    taken from https://github.com/deeplearning-wisc/knn-ood/blob/master/util/score.py
    :param inputs:
    :type inputs:
    :param model:
    :type model:
    :param forward_func:
    :type forward_func:
    :param method_args:
    :type method_args:
    :return:
    :rtype:
    """
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)
    # outputs = model(inputs)
    outputs, _, _ = forward_func(inputs, model)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    # outputs = model(Variable(tempInputs))
    with torch.no_grad():
        _, hx, _ = forward_func(tempInputs, model)
    # Calculating the confidence after adding perturbations
    nnOutputs = hx.data.cpu()
    nnOutputs = nnOutputs.numpy()
    # nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    # nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores
