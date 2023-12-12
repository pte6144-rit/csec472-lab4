import imagediff, orb, SimpleCV
import cv2


def majority(datum, f_test_descriptors, s_test_descriptors):
    results = []

    # get result from imagediff, using metric 11, average result from training
    results.append(imagediff.decide(datum, 11))

    # get result from orb, including the required preprocessing
    results.append(orb.compare_fingers(datum, f_test_descriptors, s_test_descriptors))

    # get result from SimpleCV
    results.append(SimpleCV.stringy_decide(datum))

    accept = 0
    reject = 0

    for result in results:
        if result == "accept" or result == "fraud":
            accept += 1
        else:
            reject += 1

    if max(accept, reject) == accept:
        return "accept"
    else:
        return "reject"


def main():
    # setup for orb
    print("Prepping test")
    orbO = cv2.ORB_create(nfeatures=3000, scaleFactor=1.1, nlevels=15, edgeThreshold=40, fastThreshold=18, WTA_K=3,
                          scoreType=cv2.ORB_HARRIS_SCORE)
    testData = imagediff.shuffle_files("test")

    f_test_fingers = [data["fn"] for data in testData]
    s_test_fingers = [data["sn"] for data in testData]
    f_test_descriptors = orb.preprocess_fingers(orbO, f_test_fingers)
    s_test_descriptors = orb.preprocess_fingers(orbO, s_test_fingers)

    accept = 0;
    reject = 0;
    insult = 0;
    fraud = 0;

    print("Testing...")
    for datum in testData:
        result = majority(datum, f_test_descriptors, s_test_descriptors)
        if result == "accept":
            if datum["real"]:
                accept += 1
            else:
                fraud += 1
        else:
            if datum["real"]:
                insult += 1
            else:
                reject += 1

    total = (accept + reject + fraud + insult) / 100
    print("Test Results: \nAcceptance:", accept / total, "%\nRejection:", reject / total, "% \nInsult:", insult / total,
          "% \nFraud:", fraud / total, "%")


if __name__ == "__main__":
    main()