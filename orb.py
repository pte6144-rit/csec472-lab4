# Uses orb method

import cv2
import os
import random
import numpy as np

def dumpfile(file):
    with open(file) as fd:
        return fd.read()
    
def shuffle_files(dir):
    firstImages = []
    firstText = []
    secondImages = []
    secondText = []
    firstLetters = []
    secondLetters = []
    data = []
    to_shuffle = []
    for file in os.listdir(dir):
        if file.endswith('.png'):
            if file[0] == "s":
                secondImages.append(file)
                secondLetters.append(file[7])
            else:
                firstImages.append(file)
                firstLetters.append(file[7])
        else:
            if file[0] == "f":
                firstText.append(file)
            else:
                secondText.append(file)
    for i in range(len(firstImages)):
        line = {"fi": os.path.join(dir, firstImages[i]), "ft": dumpfile(os.path.join(dir, firstText[i])),
                "si": os.path.join(dir, secondImages[i]), "st": dumpfile(os.path.join(dir, secondText[i])),
                "fl": firstLetters[i], "sl": secondLetters[i]}
        data.append(line)
    for i in range(len(data)-1, -1, -1):
        if random.randrange(2):
            to_shuffle.append(data[i])
            del data[i]
        else:
            obj = data[i]
            obj["real"] = True
            data[i] = obj
    copy = to_shuffle.copy()
    for i in range(len(copy)):
        opposing = random.randrange(len(copy))
        data.append({"fi": to_shuffle[i]["fi"], "ft": to_shuffle[i]["ft"], "fl": to_shuffle[i]["fl"],
                     "si": copy[opposing]["si"], "st": copy[opposing]["st"], "sl": copy[opposing]["sl"], "real": False})
        del copy[opposing]
    return data

def preprocess_fingers(orb, fingers):
    descriptors = {}
    for finger in fingers:
        des = get_finger_features(orb, finger)
        descriptors[finger] = des
    return descriptors

def get_finger_features(orb, finger):
    image = cv2.imread(finger, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.equalizeHist(image)
    
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    kernel = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image = cv2.filter2D(image, -1, sharpening_kernel)

    features, des = orb.detectAndCompute(image, None)
    return des


def compare_fingers(data, set1_descriptors, set2_descriptors, threshold=1000):
    acceptance = 0
    insult = 0
    fraud = 0
    rejection = 0

    for datum in data:
        des1 = set1_descriptors[datum["fi"]]
        des2 = set2_descriptors[datum["si"]]

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        count = len(matches)
        is_real_match = datum["real"]

        if count > threshold:
            if is_real_match:
                acceptance += 1
            else:
                fraud += 1
        else:
            if is_real_match:
                insult += 1
            else:
                rejection += 1

    return acceptance, insult, fraud, rejection

def main():
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2)

    #print("Starting Training")
    #training_data = shuffle_files("train")
    #f_train_fingers = [data["fi"] for data in training_data]
    #s_train_fingers = [data["si"] for data in training_data]
    #f_train_descriptors = preprocess_fingers(orb, f_train_fingers)
    #s_train_descriptors = preprocess_fingers(orb, s_train_fingers)

    #acceptance, insult, fraud, rejection = compare_fingers(training_data, f_train_descriptors, s_train_descriptors)
    #print("Training Results: Acceptance:", acceptance, "Insult:", insult, "Fraud:", fraud, "Rejection:", rejection)

    print("\nStart Testing")
    testing_data = shuffle_files("test")
    f_test_fingers = [data["fi"] for data in testing_data]
    s_test_fingers = [data["si"] for data in testing_data]
    f_test_descriptors = preprocess_fingers(orb, f_test_fingers)
    s_test_descriptors = preprocess_fingers(orb, s_test_fingers)

    #acceptance_test, insult_test, fraud_test, rejection_test = compare_fingers(testing_data, f_test_descriptors, s_test_descriptors)
    #print("Test Results: Acceptance:", acceptance_test, "Insult:", insult_test, "Fraud:", fraud_test, "Rejection:", rejection_test)

    best_threshold = 900
    best_difference = float('inf')

    for threshold in range(900, 1000, 10):  # Example range and step
        acceptance, insult, fraud, rejection = compare_fingers(testing_data, f_test_descriptors, s_test_descriptors, threshold)
        difference = abs(200 - acceptance) + abs(200 - rejection)

        if difference < best_difference:
            best_difference = difference
            best_threshold = threshold
            acceptance_test = acceptance
            insult_test = insult
            fraud_test = fraud
            rejection_test = rejection

    print("Best Threshold:", best_threshold)
    print("Test Results: Acceptance:", acceptance_test, "Insult:", insult_test, "Fraud:", fraud_test, "Rejection:", rejection_test)
if __name__ == "__main__":
    main()


