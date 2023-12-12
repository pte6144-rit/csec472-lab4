import cv2
from skimage.metrics import structural_similarity
import os
import random
import numpy as np
from PIL import Image

# Define similarity threshold for prediction
similarity_threshold = 0.09


# Define functions for image processing and similarity score calculation
def preprocess_image(img):
    img1_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img1_processed

def calculate_similarity(img1, img2):
    img1_processed = preprocess_image(img1)
    img2_processed = preprocess_image(img2)

    (score, diff) = structural_similarity(img1_processed, img2_processed, full=True)
    
    
    return score


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
        if file[-1] == "g":
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
        line = {"fi": Image.open(dir + "/" + firstImages[i]), "ft": dumpfile(dir + "/" + firstText[i]),
                "si": Image.open(dir + "/" + secondImages[i]), "st": dumpfile(dir + "/" + secondText[i]),
                "fl": firstLetters[i], "sl": secondLetters[i], "fn": dir + "/" + firstImages[i],
                "sn": dir + "/" + secondImages[i]}
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
                     "si": copy[opposing]["si"], "st": copy[opposing]["st"], "sl": copy[opposing]["sl"],
                     "fn": to_shuffle[i]["fn"], "sn": copy[opposing]["sn"], "real": False})
        del copy[opposing]
    return data







def comparing_fingers(datum):
    # Test a new pair of images
    new_img1 = cv2.imread(datum["fn"])
    new_img2 = cv2.imread(datum["sn"])

    new_similarity = calculate_similarity(new_img1, new_img2)

    return new_similarity


def main():
    print("Starting Training")
    training_data = shuffle_files("train")
    accept = 0
    insult = 0
    fraud = 0
    reject = 0

    #for datum in training_data:
    #    new_similarity = comparing_fingers(datum)
    #    if new_similarity > similarity_threshold and datum["fl"] == datum["sl"]:
    #        if datum["real"]:
    #            accept += 1
    #        else:
    #            fraud += 1
    #    else:
    #        if datum["real"]:
    #            insult += 1
    #        else:
    #            reject += 1

    print("Finished Training")

    print("\nStarting Testing")
    testing_data = shuffle_files("test")
    test_accept = 0
    test_insult = 0
    test_fraud = 0
    test_reject = 0

    for datum in testing_data:
        new_similarity = comparing_fingers(datum)
        fc = datum["ft"].split("\n")[1][7]
        sc = datum["st"].split("\n")[1][7]
        if new_similarity > similarity_threshold and datum["fl"] == datum["sl"] and fc == sc:
            if datum["real"]:
                test_accept += 1
            else:
                test_fraud += 1
        else:
            if datum["real"]:
                test_insult += 1
            else:
                test_reject += 1
    
    print("Acceptance:", "{:.2%}".format(test_accept/500))
    print("Rejection:", "{:.2%}".format(test_reject/500))
    print("Insult:", "{:.2%}".format(test_insult/500))
    print("Fraud:", "{:.2%}".format(test_fraud/500))



   


if __name__ == "__main__":
    main()
