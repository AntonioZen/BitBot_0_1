import numpy as np
import math
import pandas as pd
import pymysql
#import mysql.connector
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from colorama import Fore


#---------------------

def exponential_smoothing(series, alpha):

    smooth = [series[0]]
    for n in range(1, len(series)):
        smooth.append(alpha * series[n] + (1 - alpha) * smooth[n - 1])
    smooth_series = pd.Series(smooth, index=series.index)

    return smooth_series


def add_change_points(dataframe, z=3):

    for i in range(z):
        dataframe['CUSUM+' + str(i)] = dataframe['CUSUM'].shift(i - 1)
    dataframe = dataframe.fillna(0)

    dataframe.loc[(abs(dataframe['CUSUM+1']) > abs(dataframe['CUSUM+0'])) &
             (abs(dataframe['CUSUM+1']) > abs(dataframe['CUSUM+2'])), 'change_point'] = 1
    dataframe = dataframe.fillna(0)

    return dataframe


def find_mean(data):

    S = 0
    n = len(data)
    data['mean_in_range'] = np.nan
    data['change_point_weight'] = 1

    for i, row in data.iterrows():
        if row['change_point'] == 1:
            print(data[S:i]['metric'].mean())
            Xmean = data[S:i]['metric'].mean()
            data.ix[S:i, 'mean_in_range'] = Xmean
            data.ix[S:i + 1, 'change_point_weight'] = i - S
            #W = i - S + 1
            #print(W)
            S = i

        if i == n - 1:
            Xmean = data[S:n]['metric'].mean()
            data.ix[S:n, 'mean_in_range'] = Xmean

    return data


def cusum(data):

    rS = 0
    CUSUM = []
    n = len(data)
    xmean = data.mean()
    for i in range(n):
        rS = rS + (data[i] - xmean)
        CUSUM.append(rS)

    CUSUM = pd.Series(CUSUM, index=data.index)

    return CUSUM


def confidence_interval(data, sampling=100):

    CUSUM_0 = cusum(data)
    CUSUMdiff0 = CUSUM_0.max() - CUSUM_0.min()
    Nbreaks = 0

    for i in range(sampling - 1):

        data_new = data.sample(frac=1).reset_index(drop=True)
        CUSUM_simulation = cusum(data_new)
        CUSUMdiff = CUSUM_simulation.max() - CUSUM_simulation.min()

        if CUSUMdiff < CUSUMdiff0:
            Nbreaks = Nbreaks + 1

    confidence_interval = Nbreaks / sampling
    return confidence_interval


def series_engine(Series, alpha_metric = 0.8, alpha_cusum = 0.9):
    conf_level = 0
    series_view = pd.DataFrame()
    series_view['metric'] = Series
    series_view['series_view_index_value'] = Series.index
    data = pd.Series(exponential_smoothing(Series, alpha_metric))
    series_view['CUSUM'] = exponential_smoothing(cusum(data), alpha_cusum)
    series_view = add_change_points(series_view)
    series_view = series_view.reset_index()
    series_view = find_mean(series_view)

    conf_level = confidence_interval(Series)

    return series_view, conf_level


def read_series_view(Dataframe):

    for i, row in Dataframe.iterrows():

        ChangePoint.create()


def read_segment_dataframe(Dataframe):

    for column in Dataframe.columns:
        series_view = series_engine(Dataframe[column])
        read_series_view(series_view)

def scan_view(data, conf_level = 0):

    result = pd.DataFrame()


    segment_lenght = len(data)
    mean = data['metric'].mean()
    observation_size = len(data)
    for i, row in data.iterrows():
        if row['change_point'] == 1:
            #segment = str(data)
            change_point_index = i
            print(Fore.LIGHTBLUE_EX)
            #print(change_point_index)
            #date_change_point = row['group']
            value_from = data.ix[i-1, 'mean_in_range']
            value_to = data.ix[i, 'mean_in_range']
            signal_0 = data.ix[i, 'metric']
            signal_1 = data.ix[i+1, 'metric']
            observation_mean = mean
            confidence_interval = conf_level
            change_point_weight = row['change_point_weight']
            series_view_index_value = i


            print('change_point_index: '+str(change_point_index)
                  , 'series_view_index_value: ' + str(series_view_index_value)
                  , 'observation_size: ' + str(observation_size)
                  , 'observation_mean: ' + str(observation_mean)
                  , 'signal_0: '+str(signal_0)
                  , 'signal_1: ' + str(signal_1)
                  , 'value_from: ' + str(value_from)
                  , 'value_to: ' + str(value_to)
                  , 'confidence_interval: ' + str(confidence_interval)
                  , 'segment_lenght: '+str(segment_lenght)
                  , 'change_point_weight: '+str(change_point_weight))
            result = result.append({'signal':signal_1,'series_view_index_value':series_view_index_value,'segment_lenght':segment_lenght,'value_from': value_from, 'value_to': value_to, 'confidence_interval': confidence_interval, 'change_point_weight': change_point_weight}, ignore_index=True)
    return result

class SeriesView:
    def __init__(self, data, alpha_metric=0.8, alpha_cusum=0.9):
        self.data = pd.DataFrame()
        self.data['metric'] = data
        tmpdata = pd.Series(self.exponential_smoothing(data, alpha_metric))
        self.data['metric_alpha_1'] = tmpdata
        self.data['CUSUM'] = self.exponential_smoothing(
            self.cusum(tmpdata), alpha_cusum)
        self.data = self.add_change_points(self.data)
        self.data = self.data.reset_index()
        self.data = self.find_mean(self.data)

        self.data['mean_delta'] = self.data['metric'] - self.data['mean_in_range']
        self.data['mean_delta_percent'] = self.data['mean_delta'] / self.data['mean_in_range']

        self.data['mean_in_range_running'] = pd.Series(self.exponential_smoothing(self.data['mean_in_range'], 0.1))

    def __iter__(self):
        yield self.data.iterrows().__iter__()

    def getData(self):
        return self.data

    def getConfidenceInterval(self):
        return self.confidence_interval(self.getData())

    def confidence_interval(self, columnData, sampling=100):

        CUSUM_0 = self.cusum(columnData)
        CUSUMdiff0 = CUSUM_0.max() - CUSUM_0.min()
        Nbreaks = 0

        for i in range(sampling - 1):

            data_new = columnData.sample(frac=1).reset_index(drop=True)
            CUSUM_simulation = self.cusum(data_new)
            CUSUMdiff = CUSUM_simulation.max() - CUSUM_simulation.min()

            if CUSUMdiff < CUSUMdiff0:
                Nbreaks = Nbreaks + 1

        confidence_interval = Nbreaks / sampling
        return confidence_interval

    def exponential_smoothing(self, data, alpha):

        smooth = [data[0]]
        for n in range(1, len(data)):
            smooth.append(alpha * data[n] + (1 - alpha) * smooth[n - 1])
        smooth_series = pd.Series(smooth, index=data.index)

        return smooth_series

    def add_change_points(self, data, z=3):

        for i in range(z):
            data['CUSUM+' + str(i)] = data['CUSUM'].shift(i - 1)
            data = data.fillna(0)

        l_index = (abs(data['CUSUM+1']) > abs(data['CUSUM+0'])) & (
            abs(data['CUSUM+1']) > abs(data['CUSUM+2']))

        data.loc[l_index, 'change_point'] = 1
        data = data.fillna(0)

        return data

    def find_mean(self, data):
        S = 0
        n = len(data)
        data['mean_in_range'] = np.nan
        data['change_point_weight'] = 1

        for i, row in data.iterrows():
            if row['change_point'] == 1:
                #print(data[S:i]['metric'].mean())
                Xmean = data[S:i]['metric'].mean()
                data.ix[S:i, 'mean_in_range'] = Xmean
                data.ix[S:i, 'change_point_weight'] = i - S

                S = i

            if i == n - 1:
                Xmean = data[S:n]['metric'].mean()
                data.ix[S:n, 'mean_in_range'] = Xmean

        return data

    def cusum(self, data):

        rS = 0
        CUSUM = []
        n = len(data)
        xmean = data.mean()
        for i in range(n):
            rS = rS + (data[i] - xmean)
            CUSUM.append(rS)

        CUSUM = pd.Series(CUSUM, index=data.index)

        return CUSUM
