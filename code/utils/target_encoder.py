# coding: utf-8

import numpy as np
import pandas as pd


class TargetEncoder(object):
    """
    Target encoding as in the paper by Daniele Micci-Barreca available at:
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    """

    def __init__(self, smoothing=1, min_samples=1, noise_level=0):
        """
        min_samples (int): minimum samples to use category average
        smoothing (int): smoothing effect to balance category average vs prior
        noise_level (int): add jitter to encoded values
        """
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.noise_level = noise_level

    def _add_noise(self, series):
        return series * (1 + self.noise_level * np.random.randn(len(series)))

    def _get_averages(self, series, target):
        temp = pd.concat([series, target], axis=1)
        # Compute target mean
        averages = (temp.groupby(by=series.name)[target.name].
                    agg(["mean", "count"]))
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples) /
                         self.smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = (prior * (1 - smoothing) +
                                 averages["mean"] * smoothing)
        averages.drop(["mean", "count"], axis=1, inplace=True)
        averages.reset_index().rename(columns={'index': target.name,
                                               target.name: 'average'})
        return averages

    def _apply_averages(self, series, averages, fill_val):
        tmp = (pd.merge(series.to_frame(series.name), averages,
                        on=series.name, how='left')['average'].
               rename(series.name + '_mean').fillna(fill_val))
        tmp.index = series.index
        return tmp

    def encode(self, train_series=None, test_series=None, target=None):
        """
        train_series: training categorical feature as a pd.Series
        test_series: test categorical feature as a pd.Series
        target: target values as a pd.Series
        """
        assert len(train_series) == len(target)
        assert train_series.name == test_series.name
        averages = self._get_averages(train_series, target)
        # Apply averages to train and test series
        prior = target.mean()
        ft_train_series = self._apply_averages(train_series, averages, prior)
        ft_test_series = self._apply_averages(test_series, averages, prior)
        return {'train': self._add_noise(ft_train_series),
                'test': self._add_noise(ft_test_series)}
