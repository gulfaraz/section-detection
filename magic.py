import numpy as np
import features
import cluster

features.extractFeatures()

derivedFeatureLists = features.deriveFeatures(cluster.cluster(10), offsetLimit = 30) + features.deriveFeatures(cluster.cluster(100), offsetLimit = 30) + features.deriveFeatures(cluster.cluster(1000), offsetLimit = 30)

features.mergeFeatures(derivedFeatureLists)

k = 100

result = cluster.cluster(k, featuresFile = "featuresDerived.log")

print(list(result))
