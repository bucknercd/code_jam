#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Adapted on Mon Sep 05 14:58:57 2016 from
http://blog.yhat.com/posts/image-classification-in-Python.html
@author: Gregory B. Meyer
"""


from bs4 import BeautifulSoup
from skimage import data
from PIL import Image
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from IPython.display import HTML
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import requests
import re
import urllib2
import os
import time


def get_soup(url):
    return BeautifulSoup(requests.get(url).text)
    
def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]
    
def make_plot(pd):
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label":np.where(y==1, query, query1)})
    colors = ["red", "yellow"]
    for label, color in zip(df['label'].unique(), colors):
        mask = df['label']==label
        pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
    pl.legend()

def pull_images(queryObject, queryColor):       
    if queryColor == "BW":  
        url = "http://www.bing.com/images/search?q=" + queryObject  + "&qft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR7"
    else:
        url = "http://www.bing.com/images/search?q=" + queryObject + "&qft=+filterui:imagesize-large&FORM=R5IR7"
    print "URL " + url
    soup = get_soup(url)
    images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]    
    for img in images:
        print 'before getting raw image'
        raw_img = urllib2.urlopen(img).read()
        print 'after raw image'
        cntr = len([i for i in os.listdir("images") if queryObject in i]) + 1
        f = open("images/" + queryObject + "_"+ str(cntr) +".jpeg", 'wb')
        f.write(raw_img)
        f.close()        
    print "Finished pulling training images for " + queryObject

    
continueString = "Yes"   
while continueString in ["Yes","yes", "y", "Y"]:
    # input for 2 types of objects to train the ML
    query = str(raw_input("What is your image first subject query?  (Examples: bird, dog, cat) : "))
    query1 = str(raw_input("What is your image second subject query?  (Examples: turtle, fish, ball) : "))    
    bwSelection="x"
    # add input control for Black and White or Color
    while bwSelection not in {'BW','Color'}:
        bwSelection = str(raw_input("Use black and white images or color? enter 'BW' or 'Color' : "))
    try:
        pull_images(query,bwSelection)
        pull_images(query1,bwSelection)
    except:
        print 'exception caught'    

    time.sleep(3)
    
    standardImageDefinition = 0 
    while standardImageDefinition <50 or standardImageDefinition > 140:   
        standardImageDefinition = int(raw_input("How many pixels do you want to sample across images? Choose a number between 50 and 140. "))
    
    #setup a standard image size; this will distort some images but will get everything into the same shape
    STANDARD_SIZE = (standardImageDefinition, standardImageDefinition)
    
    img_dir = "images/"
    images = [img_dir+ f for f in os.listdir(img_dir)]
    labels = [query if query in f.split('/')[-1] else query1 for f in images]
    
    data = []
    print "Flattening the images and converting them to numerical data"
    for image in images:
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)
        
    data = np.array(data)
    is_train = np.random.uniform(0, 1, len(data)) <= 0.7
    y = np.where(np.array(labels)==query, 1, 0)
    
    train_x, train_y = data[is_train], y[is_train]
    test_x, test_y = data[is_train==False], y[is_train==False]
    
    print "You have " + str(len(train_x)) + " images to train your machine"
    print "You have " + str(len(test_x)) + " images to test your machine's learning"
    
    uniqueImageComponents = 0
    #add input to specify number of components to determine
    while uniqueImageComponents < 2 or uniqueImageComponents >6:
        uniqueImageComponents = int(raw_input("How many unique features are needed to distinguish between your image types? Choose a number between 2 and 6. "))
    
    pca = RandomizedPCA(n_components=uniqueImageComponents)
    X = pca.fit_transform(data)
    
    
    make_plot(pd)
    pl.show()
    
    time.sleep(5)
    
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    print "Training and Test sets are created."
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')
    print "Running your machine learning model on the test set."
    knn.fit(train_x, train_y)
    result = knn.predict(test_x)
    rawAccuracy = accuracy_score(test_y, result) 
    percentageAccuracy = round(rawAccuracy*100, 2)
    print "Your machine model was " + str(percentageAccuracy) + "% accurate!"
    print test_y
    print result
    os.chdir("images")        
    dir = os.getcwd()
    print "directory is " + str(dir)
    
    replyString = str(raw_input("Do you want to delete your images? (Yes or No)"))
    if replyString in ['Yes', 'yes', 'Si', 'si']:
        files = os.listdir(dir)
        for file in files:
            if file.endswith(".jpeg"):
                #print "Removing " + str(file)
                os.remove(os.path.join(dir,file))
    os.chdir("..")
    print "directory is " + str(dir)
    continueString = str(raw_input("Do you want to run the Machine Learning program, Spot the Image? (Yes or No)") )
        
pl.close()
