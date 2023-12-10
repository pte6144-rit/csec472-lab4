import SimpleCV
import os
import random
from sklearn.linear_model import LogisticRegression
# Define similarity threshold for prediction
similarity_threshold = 0.8


# Define functions for image processing and similarity score calculation
def preprocess_image(img):
    return img.grayscale().equalize()

def calculate_similarity(img1, img2):
    img1_processed = preprocess_image(img1)
    img2_processed = preprocess_image(img2)
    correlation = img1_processed.matchTemplate(img2_processed)
    return correlation.maxVal()


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







def comparing_fingers(data):
    accept = 0
    insult = 0
    fraud = 0
    reject = 0


    for datum in data:
        # Test a new pair of images
        new_img1 = SimpleCV.Image(datum["fi"])
        new_img2 = SimpleCV.Image(datum["si"])

        new_similarity = calculate_similarity(new_img1, new_img2)

    

        if new_similarity > similarity_threshold:
            if datum["real"]:
                accept += 1
            else:
                fraud += 1
        else:
            if datum["real"]:
                insult += 1
            else:
                reject += 1

    return accept, insult, fraud, reject

def main():
    print("Starting Training")
    training_data = shuffle_files("train")
    accept, insult, fraud, reject = comparing_fingers(training_data)


    print("Finished Training")

    print("\nStarting Testing")
    testing_data = shuffle_files("test")
    test_accept, test_insult, test_fraud, test_reject = comparing_fingers(testing_data)
    print("Acceptance:", "{:.2%}".format(accept/500))
    print("Rejection:", "{:.2%}".format(reject/500))
    print("Insult:", "{:.2%}".format(insult/500))
    print("Fraud:", "{:.2%}".format(fraud/500))



   


if __name__ == "__main__":
    main()
