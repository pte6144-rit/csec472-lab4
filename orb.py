# Uses orb method

import cv2
import os
import random
import numpy as np

THRESHOLD = 15
BAD_DISTANCE = 100

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
    image = cv2.equalizeHist(image)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features, des = orb.detectAndCompute(image, None)
    return des

def compare_fingers(data, set1_descriptors, set2_descriptors):
    global THRESHOLD
    global BAD_DISTANCE

    acceptance = 0
    insult = 0
    fraud = 0
    rejection = 0
    for datum in data:
        des1 = set1_descriptors[datum["fi"]]
        des2 = set2_descriptors[datum["si"]]

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        sum_distance = 0
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                sum_distance += m.distance

        count = len(good_matches)
        avg_distance = sum_distance/count
        current_threshold = THRESHOLD

        is_real_match = datum["real"]
        if count > current_threshold:
            if not is_real_match:
                BAD_DISTANCE += avg_distance
                BAD_DISTANCE = BAD_DISTANCE/2
            if is_real_match:
                acceptance += 1
            elif avg_distance < BAD_DISTANCE*.95:
                fraud += 1
            else:
                rejection += 1
        else:
            if is_real_match and avg_distance > BAD_DISTANCE*1.05:
                insult += 1
            else:
                rejection += 1

    return acceptance, insult, fraud, rejection

def main():
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.1, nlevels=15, edgeThreshold=40, fastThreshold=18, WTA_K=3, scoreType=cv2.ORB_HARRIS_SCORE)

    print("Starting Training")
    training_data = shuffle_files("train")
    f_train_fingers = [data["fi"] for data in training_data]
    s_train_fingers = [data["si"] for data in training_data]
    f_train_descriptors = preprocess_fingers(orb, f_train_fingers)
    s_train_descriptors = preprocess_fingers(orb, s_train_fingers)

    acceptance, insult, fraud, rejection = compare_fingers(training_data, f_train_descriptors, s_train_descriptors)
    print("Finished Training")

    print("\nStarting Testing")
    testing_data = shuffle_files("test")
    f_test_fingers = [data["fi"] for data in testing_data]
    s_test_fingers = [data["si"] for data in testing_data]
    f_test_descriptors = preprocess_fingers(orb, f_test_fingers)
    s_test_descriptors = preprocess_fingers(orb, s_test_fingers)

    acceptance_test, insult_test, fraud_test, rejection_test = compare_fingers(testing_data, f_test_descriptors, s_test_descriptors)
    total = (acceptance_test + insult_test + fraud_test + rejection_test)/100
    print("Test Results: \nAcceptance:", acceptance_test/total, "% \nInsult:", insult_test/total, "% \nFraud:", fraud_test/total, "% \nRejection:", rejection_test/total, "%")
if __name__ == "__main__":
    main()


