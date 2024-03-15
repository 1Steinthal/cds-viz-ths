#Import of packages
import os
import sys
sys.path.append("..")

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util
import cv2
import pandas as pd
from utils.imutils import jimshow
from utils.imutils import jimshow_channel

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

#Dataset
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Visualisation
import matplotlib.pyplot as plt


def main():
    #X_train_p = X_preproces(X_train, 32, "train")
    #X_test_p = X_preproces(X_test, 32, "test")

    inpath = os.path.join("..", "data")
    X_train_p = X_load(os.path.join(inpath, "X_train_preprocessed.npy"))
    X_test_p = X_load(os.path.join(inpath, "X_test_preprocessed.npy"))

    log_mdl = create_models(X_train_p, y_train, 0) #The logistic reg
    #MLP_mdl = create_models(X_train_p, y_train, 1) #The MLP-clas

    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(save_cls_reports_and_plots(log_mdl, X_test_p, y_test, labels, "log", 0))
    #print(save_cls_reports_and_plots(MLP_mdl, X_test_p, y_test, labels, "MLP", 1))


def X_preproces(data, dim, name):
    output_array = np.empty((1, dim*dim))

    for i, img in enumerate(data):
        img_g = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
        img_gr = img_g.reshape(-1, dim*dim)
        img_grn = img_gr/255.0
        output_array = np.append(output_array, img_grn, axis=0) 

    outpath = os.path.join("..", "data", "X_" + name + "_preprocessed.npy")
    np.save(outpath, output_array)
    return output_array



def X_load(filename):
    X_file = np.load(filename)
    X_file = np.delete(X_file, 0, 0) #For some reason my np.array is lengthened by one. I have identified the first element to be the issue

    return X_file



def create_models(X_train_p, y_train, model):
    if model == 0:
        clf = LogisticRegression(tol=0.1, 
                        solver='saga',
                        multi_class='multinomial').fit(X_train_p, y_train)
    
    elif model == 1:
        clf = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (100,),
                           max_iter=1000,
                           verbose=True,
                           random_state = 69).fit(X_train_p, y_train)
        
    return clf
    

def save_cls_reports_and_plots(model, X_test, y_test, lbl_names, path_spec, buf):
    y_pred = model.predict(X_test)

    cr = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict=True)).transpose()
    outpath = os.path.join("..", "out", "classification_report_" + path_spec + ".csv")
    cr.to_csv(outpath, index= True)

    if buf == 1: 
        plt.plot(model.loss_curve_)
        plt.title("Loss curve during training", fontsize=14)
        plt.xlabel('Iterations')
        plt.ylabel('Loss score')
        plt.show()
        plt.plot(model.loss_curve_)
        outpath = os.path.join("..", "out", "loss_curve" + path_spec + ".png")
        plt.savefig(outpath)

    return "Finished"

if __name__ == "__main__":
    main()