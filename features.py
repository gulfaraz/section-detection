import numpy as np
import re

def getFrequency(string, regex):
    return len(re.findall(regex, string));

def extractFeatures(inputFile = "input.log", outputFile = "features.log"):
    with open(inputFile, "r") as fileContent:
        content = fileContent.read()
        lines = content.splitlines()
        features = []
        for line in lines:
            lineFeatures = []
            #number of characters
            lineFeatures.append(getFrequency(line, r"."))
            #number of alphanumerics
            lineFeatures.append(getFrequency(line, r"[a-zA-Z0-9]"))
            #number of special characters
            lineFeatures.append(getFrequency(line, r"[@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]"))
            #number of alphabets
            lineFeatures.append(getFrequency(line, r"[a-zA-Z]"))
            #number of capital letters
            lineFeatures.append(getFrequency(line, r"[A-Z]"))
            #number of vowels
            lineFeatures.append(getFrequency(line, r"[aeiouAEIOU]"))
            #number of numeric characters
            lineFeatures.append(getFrequency(line, r"[0-9]"))
            #number of words with special characters
            lineFeatures.append(getFrequency(line, r"[a-zA-Z0-9@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]+"))
            #number of words
            lineFeatures.append(getFrequency(line, r"[a-zA-Z0-9]+"))
            #number of commas
            lineFeatures.append(getFrequency(line, r"[,]"))
            #number of colons
            lineFeatures.append(getFrequency(line, r"[:]"))
            #number of semi colons
            lineFeatures.append(getFrequency(line, r"[;]"))
            #number of hyphens
            lineFeatures.append(getFrequency(line, r"[-]"))
            #number of underscores
            lineFeatures.append(getFrequency(line, r"[_]"))
            #number of pipes
            lineFeatures.append(getFrequency(line, r"[|]"))
            #number of brackets and paranthesis
            lineFeatures.append(getFrequency(line, r"[()\[\]]"))
            #number of brackets
            lineFeatures.append(getFrequency(line, r"[\[\]]"))
            #number of left brackets
            lineFeatures.append(getFrequency(line, r"[\[]"))
            #number of right brackets
            lineFeatures.append(getFrequency(line, r"[\]]"))
            #number of paranthesis
            lineFeatures.append(getFrequency(line, r"[()]"))
            #number of left paranthesis
            lineFeatures.append(getFrequency(line, r"[(]"))
            #number of right paranthesis
            lineFeatures.append(getFrequency(line, r"[)]"))
            features.append(", ".join(str(lineFeature) for lineFeature in lineFeatures))
        f = open(outputFile, "w")
        f.write("\n".join(features))
        f.close()

def mergeFeatures(derivedFeatureLists, featuresFile = "features.log", outputFile = "featuresDerived.log", appendLineNumber = True):
    lines = np.genfromtxt(featuresFile, delimiter=",")
    features = []
    for lineIndex, lineFeatures in enumerate(lines):
        newFeatures = []
        for featureIndex, derivedFeatureList in enumerate(derivedFeatureLists):
            newFeatures.append(derivedFeatureLists[featureIndex][lineIndex])
        if appendLineNumber:
            features.append([lineIndex + 1] + newFeatures + lineFeatures.tolist())
        else:
            features.append(newFeatures + lineFeatures.tolist())
    f = open(outputFile, "w")
    featureEntries = []
    for line in np.array(features).tolist():
        featureEntries.append(", ".join(str(lineFeature) for lineFeature in line))
    f.write("\n".join(featureEntries))
    f.close()

def deriveFeatures(feature, offsetLimit = 20):
    derivedFeatures = []
    for offset in range(-offsetLimit, offsetLimit):
        derivedFeatures.append(np.roll(feature, offset))
    return derivedFeatures
