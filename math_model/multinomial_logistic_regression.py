#!/usr/bin/env python
# multinomial_logistic_regression.py
# Author : Saimadhu Polamuri
# Date: 05-May-2017
# About: Multinomial logistic regression model implementation

import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

import plotly.graph_objs as go
# import plotly.plotly as py
from plotly.graph_objs import *
# py.sign_in('saimadhu7', 'w22er5en40')

# Dataset Path
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "./data1.xlsx"


def scatter_with_color_dimension_graph(feature, target, layout_labels):
    """
    Scatter with color dimension graph to visualize the density of the
    Given feature with target
    :param feature:
    :param target:
    :param layout_labels:
    :return:
    """
    trace1 = go.Scatter(
        y=feature,
        mode='markers',
        marker=dict(
            size='16',
            color=target,
            colorscale='Viridis',
            showscale=True
        )
    )
    layout = go.Layout(
        title=layout_labels[2],
        xaxis=dict(title=layout_labels[0]), yaxis=dict(title=layout_labels[1]))
    data = [trace1]
    fig = Figure(data=data, layout=layout)
    # plot_url = py.plot(fig)
    # py.image.save_as(fig, filename=layout_labels[1] + '_Density.png')


def create_density_graph(dataset, features_header, target_header):
    """
    Create density graph for each feature with target
    :param dataset:
    :param features_header:
    :param target_header:
    :return:
    """
    for feature_header in features_header:
        print("Creating density graph for feature:: {} ".format(feature_header))
        layout_headers = ["Number of Observation", feature_header + " & " + target_header,
                          feature_header + " & " + target_header + " Density Graph"]
        scatter_with_color_dimension_graph(dataset[feature_header], dataset[target_header], layout_headers)


def main():
    glass_data_headers = ["总销售额(不含税)","总票数","废票数","废票率","销售利润率","成本费用利润率","需求不稳定性","供给不稳定性","销售额增长率","信誉评级"]
    glass_data = pd.read_excel(DATASET_PATH,index_col=0)
    train_X = pd.read_excel("xtrain_v2.xlsx", index_col=0)
    train_Y = pd.read_excel("ytrain_v2.xlsx", index_col=0)
    glass_data = glass_data[glass_data_headers]
    X_scaler = StandardScaler()
    glass_data = pd.DataFrame(X_scaler.fit_transform(glass_data[glass_data_headers[:-1]]))

    # print("Number of observations :: ", len(glass_data.index))
    # print("Number of columns :: ", len(glass_data.columns))
    # print("Headers :: ", glass_data.columns.values)
    # print("Target :: ", glass_data[glass_data_headers[-1]])
    # Train , Test data split

    # print("glass_data_RI :: ", list(glass_data["RI"][:10]))
    # print("glass_data_target :: ", np.array([1, 1, 1, 2, 2, 3, 4, 5, 6, 7]))
    # graph_labels = ["Number of Observations", "RI & Glass Type", "Sample RI - Glass Type Density Graph"]
    # scatter_with_color_dimension_graph(list(glass_data["RI"][:10]),
    #                                    np.array([1, 1, 1, 2, 2, 3, 4, 5, 6, 7]), graph_labels)

    # print "glass_data_headers[:-1] :: ", glass_data_headers[:-1]
    # print "glass_data_headers[-1] :: ", glass_data_headers[-1]
    # create_density_graph(glass_data, glass_data_headers[1:-1], glass_data_headers[-1])

    train_x, test_x, train_y, test_y = train_test_split(train_X,train_Y, train_size=0.8)
    # Train multi-classification model with logistic regression
    lr = linear_model.LogisticRegression()
    lr.fit(train_x, train_y)

    # Train multinomial logistic regression model
    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)

    print("Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, lr.predict(train_x)))
    print("Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, lr.predict(test_x)))

    print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
    print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))

    # 需要调优的参数
    # 请尝试将L1正则和L2正则分开，并配合合适的优化求解算法（slover）
    # tuned_parameters = {'penalty':['l1','l2'],
    #                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #                   }
    penaltys = ['l1', 'l2']
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    tuned_parameters = dict(penalty=penaltys, C=Cs)

    grid = GridSearchCV(mul_lr, tuned_parameters, cv=5, scoring='neg_log_loss')
    grid.fit(train_x, train_y)

    grid.cv_results_

    print(-grid.best_score_)
    print(grid.best_params_)

    # 绘制plot CV误差曲线
    test_means = grid.cv_results_['mean_test_score']
    test_stds = grid.cv_results_['std_test_score']
    train_means = grid.cv_results_['mean_train_score']
    train_stds = grid.cv_results_['std_train_score']

    # plot results
    n_Cs = len(Cs)
    number_penaltys = len(penaltys)
    test_scores = np.array(test_means).reshape(n_Cs, number_penaltys)
    train_scores = np.array(train_means).reshape(n_Cs, number_penaltys)
    test_stds = np.array(test_stds).reshape(n_Cs, number_penaltys)
    train_stds = np.array(train_stds).reshape(n_Cs, number_penaltys)

    x_axis = np.log10(Cs)
    for i, value in enumerate(penaltys):
        # pyplot.plot(log(Cs), test_scores[i], label= 'penalty:'   + str(value))
        pyplot.errorbar(x_axis, test_scores[:, i], yerr=test_stds[:, i], label=penaltys[i] + ' Test')
        pyplot.errorbar(x_axis, train_scores[:, i], yerr=train_stds[:, i], label=penaltys[i] + ' Train')

    pyplot.legend()
    pyplot.xlabel('log(C)')
    pyplot.ylabel('neg-logloss')
    pyplot.savefig('LogisticGridSearchCV_C.png')

    pyplot.show()

if __name__ == "__main__":
    main()