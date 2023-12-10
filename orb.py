# Uses orb method

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image

def preprocess_fingers(orb, fingers):
    descriptors = {}
    for finger in fingers:
        des = get_finger_features(orb, finger)
        descriptors[finger] = des
    return descriptors

def get_finger_features(orb, finger):
    image = cv2.imread(finger)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    features, des = orb.detectAndCompute(image, None)
    return des

def compare_fingers(orb, train_fingers, test_fingers, train_descriptors, test_descriptors):

    positives = 0
    negatives = 0
    falsePositives = 0
    falseNegatives = 0
    for train_finger in train_fingers:
        match = False

        for test_finger in test_fingers:

            des1 = train_descriptors[train_finger]
            des2 = test_descriptors[test_finger]
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            count = len(matches)
    
            if count > 200:
                match = True
                positives += 1
                print("Positive: ", positives)
                print("Negatives: ", negatives)

        if match == False:
            negatives += 1
                
        
    return positives, negatives, falsePositives, falseNegatives

# function for getting all the images
def get_fingers():
    f_train_fingers = []
    f_test_fingers = []
    s_train_fingers = []
    s_test_fingers = []

    train = "train"
    test = "test"

    for file in os.listdir(train):
        if file.endswith('.png'):
            if file[0] == "s":
                fingerPath = os.path.join(train, file)
                s_train_fingers.append(fingerPath)
            else:
                fingerPath = os.path.join(train, file)
                f_train_fingers.append(fingerPath) 
    
    for file in os.listdir(test):
        if file.endswith('.png'):
            if file[0] == "s":
                fingerPath = os.path.join(test, file)
                s_test_fingers.append(fingerPath)
            else:
                fingerPath = os.path.join(test, file)
                f_test_fingers.append(fingerPath)

    return f_train_fingers, f_test_fingers, s_train_fingers, s_test_fingers


def main():
    print("Starting training")

    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2)
    f_train_fingers, f_test_fingers, s_train_fingers, s_test_fingers = get_fingers()
    f_train_descriptors = preprocess_fingers(orb, f_train_fingers)
    #f_test_descriptors = preprocess_fingers(orb, f_test_fingers)
    s_train_descriptors = preprocess_fingers(orb, s_train_fingers)
    #s_test_descriptors = preprocess_fingers(orb, s_test_fingers)

    print("Start comparison")
    # Start comparison
    positives, negatives, falsePositives, falseNegatives = compare_fingers(orb, f_train_fingers, s_train_fingers, f_train_descriptors, s_train_descriptors)
    #positives, negatives, falsePositives, falseNegatives = compare_fingers(orb, s_train_fingers, s_test_fingers, s_train_fingers_class, s_test_fingers_class)

    print("Test Results:")
    print(positives, negatives)

if __name__ == "__main__":
    main()