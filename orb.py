# Uses orb method

import cv2
import os
import random

THRESHOLD = 13

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
        current_threshold = THRESHOLD

        is_real_match = datum["real"]
        if count > current_threshold and datum["fl"] == datum["sl"]:
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
    global THRESHOLD
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.1, nlevels=15, edgeThreshold=40, fastThreshold=18, WTA_K=3, scoreType=cv2.ORB_HARRIS_SCORE)

    print("Starting Training")
    training_data = shuffle_files("train")
    f_train_fingers = [data["fi"] for data in training_data]
    s_train_fingers = [data["si"] for data in training_data]
    f_train_descriptors = preprocess_fingers(orb, f_train_fingers)
    s_train_descriptors = preprocess_fingers(orb, s_train_fingers)

    acceptance, insult, fraud, rejection = compare_fingers(training_data, f_train_descriptors, s_train_descriptors)
    if (insult/fraud) > 1.25:
        THRESHOLD -= 1
    elif (insult/fraud) < .75:
        THRESHOLD += 1
    print("Finished Training")

    insult_min = 0
    insult_max = 0
    insult_avg = 0
    fraud_min = 0
    fraud_max = 0
    fraud_avg = 0
    err = []
    for i in range (15):
        print("\nStarting Testing")
        testing_data = shuffle_files("test")
        f_test_fingers = [data["fi"] for data in testing_data]
        s_test_fingers = [data["si"] for data in testing_data]
        f_test_descriptors = preprocess_fingers(orb, f_test_fingers)
        s_test_descriptors = preprocess_fingers(orb, s_test_fingers)

        acceptance_test, insult_test, fraud_test, rejection_test = compare_fingers(testing_data, f_test_descriptors, s_test_descriptors)
        total = (acceptance_test + insult_test + fraud_test + rejection_test)/100
        print("Test Results: \nAcceptance:", acceptance_test/total, "%\nRejection:", rejection_test/total, "% \nInsult:", insult_test/total, "% \nFraud:", fraud_test/total, "%")

        if i == 0:
            insult_min = insult_test/total
            insult_max = insult_test/total
            insult_avg = insult_test/total
            fraud_min = fraud_test/total
            fraud_max = fraud_test/total
            fraud_avg = fraud_test/total
        else:
            if insult_test < insult_min:
                insult_min = insult_test/total
            if insult_test > insult_max:
                insult_max = insult_test/total
            if fraud_test < fraud_min:
                fraud_min = fraud_test/total
            if fraud_test > fraud_max:
                fraud_max = fraud_test/total

            insult_avg = ((insult_avg) + (insult_test/total))/2
            fraud_avg = ((fraud_avg) + (fraud_test/total))/2
        if insult_test/total == fraud_test/total:
            err.append(insult_test/total)

    print("FRR Min: ", insult_min)
    print("FRR Avg: ", insult_avg)
    print("FRR Max: ", insult_max)

    print("FAR Min: ", fraud_min)
    print("FAR Avg: ", fraud_avg)
    print("FAR Max: ", fraud_max)
    print("Err: ", err)

    

if __name__ == "__main__":
    main()


