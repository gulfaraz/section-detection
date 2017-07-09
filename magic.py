import pandas
import features
import classify
import cluster
from itertools import groupby

features.extractFeatures(inputFile="input.log", outputFile="features.csv")

unlabelledSet = pandas.read_csv("features.csv", skipinitialspace=True, header=0)

unlabelledSet["label"] = pandas.DataFrame(classify.predict(unlabelledSet), columns=["label"])

#unlabelledSet["label"] = pandas.read_csv("labels.csv", skipinitialspace=True, header=0)

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

resultSet["line_number"] = resultSet.index

print "Sections are probably at:"
for k, l in groupby(resultSet.iterrows(), key=lambda x: x[1]["pattern_100"]):
    consecutiveLines = [(t[1]["line_number"], t[1]["label"]) for t in l]
    if len(consecutiveLines) > (offsetValue / 3):
        if consecutiveLines[0][1] > 0:
            print consecutiveLines[0]

resultSet.to_csv("clusterResult.csv", sep=",", encoding="utf-8", index=True, index_label="index")
