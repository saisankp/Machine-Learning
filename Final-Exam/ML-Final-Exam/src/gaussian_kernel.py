import numpy as np


def gaussian_kernel1(distances):
    weights = np.exp((1 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel5(distances):
    weights = np.exp((5 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel10(distances):
    weights = np.exp((10 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel30(distances):
    weights = np.exp((-30 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel50(distances):
    weights = np.exp((-50 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel100(distances):
    weights = np.exp((-100 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel150(distances):
    weights = np.exp((-150 * (distances ** 2)))
    return weights / np.sum(weights)