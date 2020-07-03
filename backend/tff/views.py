from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.core import serializers
import numpy as np
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import k_means
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow_federated as tff
from scipy.spatial.distance import pdist
from scipy.special import softmax
import json
import random
from . import api
from time import time
import os

cache_path = os.path.join(os.getcwd(), 'cache')
# dataset_path = os.path.join(os.getcwd(), 'MNIST_data/')
# emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
# mnist = input_data.read_data_sets(dataset_path, one_hot=True)
# clientNum = len(emnist_train.client_ids)
batch_size = 500#mnist.validation.num_examples

def index(request):
    return HttpResponse("hello", content_type="text/plain")

def findServer(request):
    server = api.find_server()
    iternum = len(server)
    num = [0] * iternum
    acc = [0] * iternum
    loss = [0] * iternum
    iter = 0
    for state in server:
        itertmp = state.iter - 1
        # print(itertmp)
        num[itertmp] = state.num
        acc[itertmp] = state.acc
        loss[itertmp] = state.loss
        iter = max(itertmp + 1, iter)
    result = {}
    result["num"] = num
    result["acc"] = acc
    result["loss"] = loss
    if iter != iternum:
        print("server count data wrong!")
    result["iternum"] = iter
    return JsonResponse(result, safe=False)


def findClientStas(request, miniter, maxiter):
    clients = api.find_client_stas(miniter, maxiter)
    iternum = maxiter - miniter
    losssum = []
    accsum = []
    indexsum = []
    for i in range(iternum + 1):
        losssum.append([])
        accsum.append([])
        indexsum.append([])
    for client in clients:
        losssum[client.iter - miniter].append(client.loss)
        accsum[client.iter - miniter].append(client.acc)
        indexsum[client.iter - miniter].append(client.index)
    finalresult = {}
    for i in range(iternum + 1):
        result = {}
        result["loss"] = losssum[i]
        result["acc"] = accsum[i]
        result["index"] = indexsum[i]
        finalresult[i + miniter] = result
    return JsonResponse(finalresult, safe=False)


def findServerParaByIter(request, iter):
    para = api.find_server_para(iter)[0]
    result = [item for sublist in para["w1"] for item in sublist]
    result.extend(para["b1"])
    return JsonResponse(result, safe=False)


def findClientByIndex(request, index):
    result = api.find_client_by_index(index)
    return JsonResponse(result.to_json(), safe=False)


def findClientByIter(request, iter):
    result = api.find_client_by_iter(iter)
    return JsonResponse(result.to_json(), safe=False)


def findClientParaByIter(request, iter):
    # for i in range(334):
    #     iter = i+1
    # cache_file = os.path.join(cache_path,'project_iter_'+str(iter)+'.json')
    # if os.path.exists(cache_file):
    #     with open(cache_file, 'r', encoding = 'utf-8') as fr:
    #         result = json.load(fr)
    #         return JsonResponse(json.dumps(result), safe=False)
    print("开始转换第" + str(iter) + "次投影数据。")
    start = time()
    result = {}
    result['idList'] = ['server']
    server_data = api.find_server_para(iter)
    client_data = api.find_client_para_by_iter(iter)
    server = server_data[0]
    client_num = len(client_data)
    class_num = len(client_data[0]["w1"][0])
    weight_num = len(client_data[0]["w1"])
    bias_num = len(client_data[0]["b1"])
    X = np.zeros((client_num + 1, weight_num * class_num + bias_num))
    client_index = 1
    server_data = [item for w in server["w1"] for item in w]
    server_data.extend(server["b1"])
    X[0] = server_data
    for client in client_data:
        result['idList'].append(client["index"])
        row = [item for w in client["w1"] for item in w]
        row.extend(client["b1"])
        X[client_index] = row
        client_index += 1
    print(str(time() - start) + "s")
    print("完成转换第" + str(iter) + "次投影数据。")

    """PROJECT"""
    distance_matrix = pairwise_distances(X, X, metric='cosine') #* np.linalg.norm(X) * np.linalg.norm(X)
    # X_project = manifold.MDS(n_components=2,random_state=300, dissimilarity="precomputed").fit_transform(distance_matrix)
    X_project = manifold.MDS(n_components=2, random_state=300).fit_transform(X)

    # X_project = PCA(n_components=2).fit_transform(X)

    # X_project = manifold.TSNE(n_components=2)

    '''嵌入空间可视化'''
    x_min, x_max = X_project.min(0), X_project.max(0)

    X_norm = 0.5 * (X_project - x_min) / (x_max - x_min)  # 归一化
    x_mid = np.array([0.5, 0.5])
    server_coord = X_norm[0]

    transform_x = abs(x_mid[0] - server_coord[0])
    transform_y = abs(x_mid[1] - server_coord[1])
    scale = 0.8 * (0.5 / (0.5 - min(transform_x, transform_y, abs(0.5 - transform_x), abs(0.5 - transform_y))))

    X_tmp = X_norm + (x_mid - server_coord)  # 平移
    X_result = (X_tmp - x_mid) * scale + x_mid  # 按比例放大至尽可能充满
    result["pos"] = X_result.tolist()

    """Outlier detect"""

    # _,y_pred,_ = k_means(X, n_clusters=2)

    # y_pred = LocalOutlierFactor(n_neighbors=30, metric='precomputed', contamination=0.1).fit_predict(distance_matrix)

    # y_pred = DBSCAN(eps=3, min_samples=2).fit(X)
    # y_pred = y_pred.labels_

    # clf = OneClassSVM(gamma='auto').fit(X)
    # y_pred = clf.predict(X)

    # cov = EllipticEnvelope(random_state=0).fit(X)
    # y_pred = cov.predict(X)

    clf = IsolationForest(n_estimators=10, warm_start=True).fit(X)
    sklearn_y_pred = clf.predict(X)

    sklearn_y_pred = sklearn_y_pred.tolist()

    normList = [0.]
    server_np = np.array(X[0])
    for client in X[1:]:
        client = np.array(client)
        server = server_np

        dist = np.linalg.norm(client-server)

        # dist = np.dot(client, server) #/(np.linalg.norm(client)*np.linalg.norm(server))

        #X = np.vstack([client, server])
        #XT = X.T
        #dist = pdist(XT, 'mahalanobis')

        normList.append(dist)
    sortedList = sorted(normList)
    if len(sortedList)%2==1:
        index = int((len(sortedList) - 1) / 2)
        M = sortedList[index]
        sortedList = [abs(d - M) for d in sortedList]
        sortedList.sort()
        MAD = 0.8676 * sortedList[index]
    else:
        index = int(len(sortedList) / 2)
        M = 1/2 * (sortedList[index-1] + sortedList[index])
        sortedList = [abs(d - M) for d in sortedList]
        sortedList.sort()
        MAD = 0.8676 * 1/2 * (sortedList[index-1] + sortedList[index])

    y_pred = []
    for dist in normList:
        if (dist > M + 2.5 * MAD):# or (dist < M - 2.5 * MAD):
            y_pred.append(-1)
        else:
            y_pred.append(1)

    # isNormal = []
    # for i in range(len(y_pred)):
    #     if y_pred[i] == -1 or sklearn_y_pred[i] == -1:
    #         isNormal.append(-1)
    #     else:
    #         isNormal.append(1)

    result['isNormal'] = y_pred

    # with open(cache_file, 'w', encoding = 'utf-8') as fw:
    #     json.dump(result, fw)

    return JsonResponse(json.dumps(result), safe=False)

def findClientParaByIterIndex(request, iter, index):
    res = api.find_client_para_by_iter_index(iter, index)[0].to_mongo()
    result = [item for sublist in res["w1"] for item in sublist]
    result.extend(res["b1"])
    return JsonResponse(result, safe=False)


def findClientParaByIterIndexarr(request, iter, indexarr):
    indexarr = indexarr.split(",")
    para = api.find_client_para_by_iter_indexarr(iter, indexarr)
    result = []
    for para_i in para:
        temp = [item for sublist in para_i["w1"] for item in sublist]
        temp.extend(para_i["b1"])
        result.append(temp)
    return JsonResponse(result, safe=False)

def findConfusionMatrixByiIerClientIndex(request, iter, index):
    # clientList = random.sample(range(1,clientNum),10)
    # para = api.find_client_para_by_iter_index(iter, index)[0].to_mongo()
    # W = np.array(para['w1']).astype(np.float32)
    # b = np.array(para['b1']).astype(np.float32)
    # predictions = []
    # labels = []
    # with tf.Session() as sess:
    #     for sampleClient in clientList:
    #         client_dataset = emnist_test.create_tf_dataset_for_client(
    #             emnist_test.client_ids[sampleClient])
    #         iterator = client_dataset.make_one_shot_iterator()
    #         next_item = iterator.get_next()
    #         for i in range(batch_size):
    #             try:
    #                 item = sess.run(next_item)
    #                 x = tf.reshape(item['pixels'], [-1])
    #                 y = tf.matmul([x], W) + b
    #                 labels.append(item['label'])
    #                 predictions.append(sess.run(tf.argmax(y[0])))
    #             except tf.errors.OutOfRangeError:
    #                 break
    #     confusion_matrix = tf.contrib.metrics.confusion_matrix(labels, predictions, num_classes=10)
    #     result = sess.run(confusion_matrix)
    #     result = result.tolist()

    client = api.find_client_by_iter_index(iter, index)[0].to_mongo()
    result = np.array(client['matrix']).astype(np.float32)
    return JsonResponse(result.tolist(), safe=False)
