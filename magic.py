import pandas
import features
import classify
import cluster
import itertools

INPUT_FILE = "input.log"
FEATURES_FILE = "features.csv"
LABEL_COLUMN_NAME = "label"
LINE_NUMBER_COLUMN_NAME = "line_number"

features.extractFeatures(inputFile=INPUT_FILE, outputFile=FEATURES_FILE)

unlabelledSet = pandas.read_csv(FEATURES_FILE, skipinitialspace=True, header=0)

unlabelledSet[LABEL_COLUMN_NAME] = pandas.DataFrame(classify.predict(unlabelledSet), columns=[LABEL_COLUMN_NAME])

#unlabelledSet[LABEL_COLUMN_NAME] = pandas.read_csv("labels.csv", skipinitialspace=True, header=0)

labelledSet = unlabelledSet

#classify.train(labelledSet, 1000)

#classify.evaluate(labelledSet, 1)

type10 = cluster.kMeans(10, labelledSet, columnPrefix="type")
type100 = cluster.kMeans(100, labelledSet, columnPrefix="type")
type1000 = cluster.kMeans(1000, labelledSet, columnPrefix="type")

offsetValue = 10

derivedFeatureLists10 = features.deriveFeatures(type10, offsetValue)
derivedFeatureLists100 = features.deriveFeatures(type100, offsetValue)
derivedFeatureLists1000 = features.deriveFeatures(type1000, offsetValue)

patternSet = pandas.concat([labelledSet, derivedFeatureLists10, derivedFeatureLists100, derivedFeatureLists1000], axis=1)

pattern100 = cluster.kMeans(100, patternSet, columnPrefix="pattern")

resultSet = pandas.concat([pattern100, patternSet], axis=1)

resultSet[LINE_NUMBER_COLUMN_NAME] = resultSet.index

interestingIndexes = []

for k, l in itertools.groupby(resultSet.iterrows(), key=lambda x: x[1]["pattern_100"]):
    consecutiveLines = [(t[1][LINE_NUMBER_COLUMN_NAME], t[1][LABEL_COLUMN_NAME]) for t in l]
    if len(consecutiveLines) > (offsetValue / 3):
        if consecutiveLines[0][1] > 0:
            interestingIndexes.append(consecutiveLines[0])

resultSet.to_csv("clusterResult.csv", sep=",", encoding="utf-8", index=True, index_label="index")

def findSegmentTerminals(lineNumber, lineType):
    startIndex = lineNumber
    stopIndex = lineNumber
    while (resultSet.loc[resultSet[LINE_NUMBER_COLUMN_NAME] == startIndex - 1, LABEL_COLUMN_NAME].values[0] == lineType):
        startIndex -= 1
    while (resultSet.loc[resultSet[LINE_NUMBER_COLUMN_NAME] == stopIndex + 1, LABEL_COLUMN_NAME].values[0] == lineType):
        stopIndex += 1
    return (startIndex, stopIndex)

def getLinesFromFile(lineRange):
    lineList = []
    with open(INPUT_FILE, "r") as logFile:
        for line in itertools.islice(logFile, lineRange[0], lineRange[1] + 1):
            lineList.append(line)
    return "".join(lineList)

f = open("detectedSections.log", "a")
for i, j in interestingIndexes:
    lineRange = findSegmentTerminals(i, j)
    if j == 1:
        f.write("NV PAIR : " + str(lineRange[0]) + " - " + str(lineRange[1]) + "\n")
        f.write("=======\n")
        f.write(getLinesFromFile(lineRange))
        f.write("\n")
    elif j == 2:
        f.write("ALIGNED BASIC : " + str(lineRange[0]) + " - " + str(lineRange[1]) + "\n")
        f.write("=============\n")
        f.write(getLinesFromFile(lineRange))
        f.write("\n")
f.close()
