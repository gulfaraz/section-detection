import numpy as np
import features
import cluster

features.extractFeatures()
offsetValue = 30
derivedFeatureLists = features.deriveFeatures(cluster.cluster(10), offsetLimit = offsetValue) + features.deriveFeatures(cluster.cluster(100), offsetLimit = offsetValue) + features.deriveFeatures(cluster.cluster(1000), offsetLimit = offsetValue)

features.mergeFeatures(derivedFeatureLists)

k = 100

result = cluster.cluster(k, featuresFile = "featuresDerived.log")

features.mergeFeatures([result], featuresFile = "featuresDerived.log", outputFile = "result.log", appendLineNumber = False)
