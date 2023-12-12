from PIL import Image
import os
import random
import sys

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

def get_ridges(line):
    first_previous = 0
    second_previous = 0
    third_previous = 0
    previous = 0
    ridges = 0
    for value in line:
        new = 1 if value > previous else -1
        previous = value
        if third_previous == -1 and second_previous == -1 and first_previous == 1 and new == 1:
            ridges += 1
        third_previous, second_previous, first_previous = second_previous, first_previous, new
    return ridges


def decide(datum, metric):
    line = list(datum["fi"].getdata())
    line = line[int(512 * 127.3):int(512 * 127.7)] + line[int(512 * 255.3):int(512 * 255.7)] + line[
                                                                                               int(512 * 383.3):int(
                                                                                                   512 * 383.7)]
    firstLines = get_ridges(line)
    line = list(datum["si"].getdata())
    line = line[int(512 * 127.3):int(512 * 127.7)] + line[int(512 * 255.3):int(512 * 255.7)] + line[
                                                                                               int(512 * 383.3):int(
                                                                                                   512 * 383.7)]
    secondLines = get_ridges(line)
    if abs(firstLines - secondLines) <= metric and datum["fl"] == datum["sl"]:
        if datum["real"]:
            return "accept"
        else:
            return "fraud"
    else:
        if datum["real"]:
            return "insult"
        else:
            return "reject"

def main():
    print("Beginning Training")
    print("Please be Patient")
    dictionary = shuffle_files("train")
    real_differences = []
    fake_differences = []
    for datum in dictionary:
        line = list(datum["fi"].getdata())
        line = line[int(512*127.3):int(512*127.7)] + line[int(512*255.3):int(512*255.7)] + line[int(512*383.3):int(512*383.7)]
        firstLines = get_ridges(line)
        line = list(datum["si"].getdata())
        line = line[int(512*127.3):int(512*127.7)] + line[int(512*255.3):int(512*255.7)] + line[int(512*383.3):int(512*383.7)]
        secondLines = get_ridges(line)
        if (datum["real"]):
            real_differences.append({"f": firstLines, "s": secondLines, "t": datum["fl"] == datum["sl"]})
        else:
            fake_differences.append({"f": firstLines, "s": secondLines, "t": datum["fl"] == datum["sl"]})
    eer_sum = 0
    insult = sum(map(lambda a: 1 if eer_sum < abs(a["f"] - a["s"]) and a["t"] else 0, real_differences)) / len(real_differences)
    fraud = sum(map(lambda a: 1 if eer_sum > abs(a["f"] - a["s"]) and a["t"] else 0, fake_differences)) / len(fake_differences)
    while (insult > fraud):
        eer_sum += 1
        insult = sum(map(lambda a: 1 if eer_sum < abs(a["f"] - a["s"]) and a["t"] else 0, real_differences)) / len(real_differences)
        fraud = sum(map(lambda a: 1 if eer_sum > abs(a["f"] - a["s"]) and a["t"] else 0, fake_differences)) / len(fake_differences)

    print("Beginning Testing")
    test = shuffle_files("test")
    accept = 0
    insult = 0
    fraud = 0
    reject = 0
    for datum in test:
        result = decide(datum, eer_sum)
        if result == "accept":
            accept += 1
        elif result == "reject":
            reject += 1
        elif result == "fraud":
            fraud += 1
        else:
            insult += 1

    print("Test Results:")
    print("Acceptance:", "{:.2%}".format(accept/500))
    print("Rejection:", "{:.2%}".format(reject/500))
    print("Insult:", "{:.2%}".format(insult/500))
    print("Fraud:", "{:.2%}".format(fraud/500))


if __name__ == "__main__":
    main()