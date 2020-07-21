import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re

# what and where
inceptionV3_dir = 'v4'
images_dir = 'images'

# inception-v


def create_graph():
    
    with tf.io.gfile.GFile(os.path.join(inceptionV3_dir, 'inception_v4.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    
    nb_features0 = 41
    nb_features1 = 41
    nb_features2 = 1536
    features = np.zeros((len(list_images), nb_features1, nb_features0, nb_features2)).astype(object)
    labels = []

    create_graph()

    # pool_3:0: next-to-last layer containing 2048 float description of the image.
    # DecodeJpeg/contents:0:JPEG encoding of the image.

    with tf.compat.v1.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('InceptionV4/Logits/AvgPool_1a/AvgPool:0')

        for ind, image in enumerate(list_images):
            imlabel = image.split('/')[1]

            # rough indication of progress
            if ind % 100 == 0:
                print('Processing', image, imlabel)
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            decoded_image = tf.image.decode_image(image_data)
            #squeezed_image = tf.squeeze(decoded_image,0)
            resized_image = tf.expand_dims(decoded_image,0)
            image_float = tf.image.convert_image_dtype(resized_image, tf.float32)
            decoded_image_float = sess.run(image_float)
            print (type(decoded_image_float))
            #decoded_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
            predictions = sess.run(next_to_last_tensor, {'InputImage:0': decoded_image_float})
            features[ind, :] = np.squeeze(predictions)
            labels.append(imlabel)

    return features, labels


# Graphics


def plot_features(feature_labels, t_sne_features):
    
    plt.figure(figsize=(9, 9), dpi=100)

    uniques = {x: labels.count(x) for x in feature_labels}
    od = collections.OrderedDict(sorted(uniques.items()))

    colors = itertools.cycle(["r", "b", "g", "c", "m", "y",
                              "slategray", "plum", "cornflowerblue",
                              "hotpink", "darkorange", "forestgreen",
                              "tan", "firebrick", "sandybrown"])
    n = 0
    for label in od:
        count = od[label]
        m = n + count
        plt.scatter(t_sne_features[n:m, 0], t_sne_features[n:m, 1], c=next(colors), s=10, edgecolors='none')
        c = (m + n) // 2
        plt.annotate(label, (t_sne_features[c, 0], t_sne_features[c, 1]))
        n = m

    plt.show()




def plot_confusion_matrix(y_true, y_pred, matrix_title):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    plt.title(matrix_title, fontsize=12)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.show()
    
    


#classifier function to run classifier and output results


def run_classifier(clfr, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str):
    start_time = time.time()
    clfr.fit(x_train_data, y_train_data)
    y_pred = clfr.predict(x_test_data)
    print("%f seconds" % (time.time() - start_time))

    # confusion matrix 
    print(acc_str.format(accuracy_score(y_test_data, y_pred) * 100))
    plot_confusion_matrix(y_test_data, y_pred, matrix_header_str)


# get the images and the labels 
dir_list = [x[0] for x in os.walk(images_dir)]
dir_list = dir_list[1:]
list_images = []
for image_sub_dir in dir_list:
	sub_dir_images = [image_sub_dir + '/' + f for f in os.listdir(image_sub_dir) if re.search('jpg|JPG', f)]
	list_images.extend(sub_dir_images)

# extract features
features, labels = extract_features(list_images)




# Classification

tsne_features = TSNE().fit_transform(features)


plot_features(labels, tsne_features)

# training and test datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42, stratify = labels)


# Support Vector Machine
print('Support Vector Machine starting ...')
cl = LinearSVC()
run_classifier(cl, X_train, y_train, X_test, y_test, "CNN-SVM Accuracy: {0:0.1f}%", "SVM Confusion matrix")

#Extra Trees
print('Extra Trees Classifier starting ...')
cl = ExtraTreesClassifier(n_jobs=1,  n_estimators=10, criterion='gini', min_samples_split=2,
                           max_features=50, max_depth=None, min_samples_leaf=1)
run_classifier(cl, X_train, y_train, X_test, y_test, "CNN-ET Accuracy: {0:0.1f}%", "Extra Trees Confusion matrix")

# Random Forest
print('Random Forest Classifier starting ...')
cl = RandomForestClassifier(n_jobs=1, criterion='entropy', n_estimators=10, min_samples_split=2)
run_classifier(cl, X_train, y_train, X_test, y_test, "CNN-RF Accuracy: {0:0.1f}%", "Random Forest Confusion matrix")

#knn
print('K-Nearest Neighbours Classifier starting ...')
cl = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
run_classifier(cl, X_train, y_train, X_test, y_test, "CNN-KNN Accuracy: {0:0.1f}%",
               "K-Nearest Neighbor Confusion matrix")

#MyLittlePony
print('Multi-layer Perceptron Classifier starting ...')
clf = MLPClassifier()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-MLP Accuracy: {0:0.1f}%",
               "Multi-layer Perceptron Confusion matrix")


#Gaussian Naive Bayes Classifier
print('Gaussian Naive Bayes Classifier starting ...')
clf = GaussianNB()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-GNB Accuracy: {0:0.1f}%",
               "Gaussian Naive Bayes Confusion matrix")

#LDA
print('Linear Discriminant Analysis Classifier starting ...')
clf = LinearDiscriminantAnalysis()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-LDA Accuracy: {0:0.1f}%",
               "Linear Discriminant Analysis Confusion matrix")

#QDA
print('Quadratic Discriminant Analysis Classifier starting ...')
clf = QuadraticDiscriminantAnalysis()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-QDA Accuracy: {0:0.1f}%",
               "Quadratic Discriminant Analysis Confusion matrix")

#import/InceptionV4/Logits/Logits/biases
#import/InceptionV4/Logits/Logits/biases/read
#import/InceptionV4/Logits/Logits/BiasAdd
#import/InceptionV4/Logits/Predictions



