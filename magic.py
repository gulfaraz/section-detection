import pandas
import features
import classify
import cluster
from itertools import groupby

features.extractFeatures(inputFile="input.log", outputFile="features.csv")

unlabelledSet = pandas.read_csv("features.csv", skipinitialspace=True, header=0)

labels = pandas.read_csv("labels.csv", skipinitialspace=True, header=0)

unlabelledSet["Label"] = labels

labelledSet = unlabelledSet

offsetValue = 10

type10 = cluster.kMeans(10, labelledSet, columnPrefix="Type")

type100 = cluster.kMeans(100, labelledSet, columnPrefix="Type")

type1000 = cluster.kMeans(1000, labelledSet, columnPrefix="Type")

derivedFeatureLists10 = features.deriveFeatures(type10, offsetValue)
derivedFeatureLists100 = features.deriveFeatures(type100, offsetValue)
derivedFeatureLists1000 = features.deriveFeatures(type1000, offsetValue)

patternSet = pandas.concat([labelledSet, derivedFeatureLists10, derivedFeatureLists100, derivedFeatureLists1000], axis=1)

pattern100 = cluster.kMeans(100, patternSet, columnPrefix="Pattern")

resultSet = pandas.concat([pattern100, patternSet], axis=1)

resultSet["Line Number"] = resultSet.index

print "Sections are probably at:"
for k, l in groupby(resultSet.iterrows(), key=lambda x: x[1]["Pattern100"]):
    consecutiveLines = [(t[1]["Line Number"], t[1]["Label"]) for t in l]
    if len(consecutiveLines) > (offsetValue / 3):
        if consecutiveLines[0][1] > 0:
            print consecutiveLines[0]

resultSet.to_csv("clusterResult.csv", sep=",", encoding="utf-8", index=True, index_label="Index")
