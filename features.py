import numpy as np
import pandas
import re

featureHeaders = [
    "Characters",
    "Alphanumerics",
    "Special Characters",
    "Alphabets",
    "Capital Letters",
    "Vowels",
    "Numeric Characters",
    "Words with Special Characters",
    "Words",
    "Commas",
    "Colons",
    "Semi Colons",
    "Hyphens",
    "Underscores",
    "Pipes",
    "Brackets and Paranthesis and Braces",
    "Brackets",
    "Left Brackets",
    "Right Brackets",
    "Paranthesis",
    "Left Paranthesis",
    "Right Paranthesis",
    "Braces",
    "Left Braces",
    "Right Braces",
    "All Math Symbols",
    "Common Math Operations",
    "Sum Operator",
    "Product Operator",
    "Division Operator",
    "Assignment Operator",
    "Greater Sign",
    "Lesser Sign",
    "Question Mark",
    "Single and Double Quotations",
    "Single Quotations",
    "Double Quotations",
    "Ampersands",
    "At Sign",
    "Hash Symbol",
    "Dollar Sign",
    "Caps Sign",
    "Percentage Sign",
    "Logical Operators",
    "Excalmation"
];

def getFrequency(string, regex):
    return len(re.findall(regex, string));

def extractFeatures(inputFile, outputFile):
    with open(inputFile, "r") as fileContent:
        content = fileContent.read()
        lines = content.splitlines()
        features = [ ", ".join(featureHeaders) ]
        for line in lines:
            lineFeatures = []
            #number of characters
            lineFeatures.append(getFrequency(line, r"."))
            #number of alphanumerics
            lineFeatures.append(getFrequency(line, r"[a-zA-Z0-9]"))
            #number of special characters
            lineFeatures.append(getFrequency(line, r"[\@\#\$\%\^\&\*\(\)\_\+\\\-\=\[\]\{\}\;\'\:\"\\\|\,\.\<\>\/\?\!]"))
            #number of alphabets
            lineFeatures.append(getFrequency(line, r"[a-zA-Z]"))
            #number of capital letters
            lineFeatures.append(getFrequency(line, r"[A-Z]"))
            #number of vowels
            lineFeatures.append(getFrequency(line, r"[aeiouAEIOU]"))
            #number of numeric characters
            lineFeatures.append(getFrequency(line, r"[0-9]"))
            #number of words with special characters
            lineFeatures.append(getFrequency(line, r"[a-zA-Z0-9\@\#\$\%\^\&\*\(\)\_\+\\\-\=\[\]\{\}\;\'\:\"\\\|\,\.\<\>\/\?\!]+"))
            #number of words
            lineFeatures.append(getFrequency(line, r"[a-zA-Z0-9]+"))
            #number of commas
            lineFeatures.append(getFrequency(line, r"[\,]"))
            #number of colons
            lineFeatures.append(getFrequency(line, r"[\:]"))
            #number of semi colons
            lineFeatures.append(getFrequency(line, r"[\;]"))
            #number of hyphens
            lineFeatures.append(getFrequency(line, r"[\-]"))
            #number of underscores
            lineFeatures.append(getFrequency(line, r"[\_]"))
            #number of pipes
            lineFeatures.append(getFrequency(line, r"[\|]"))
            #number of brackets, paranthesis and braces
            lineFeatures.append(getFrequency(line, r"[\(\)\[\]\{\}]"))
            #number of brackets
            lineFeatures.append(getFrequency(line, r"[\[\]]"))
            #number of left brackets
            lineFeatures.append(getFrequency(line, r"[\[]"))
            #number of right brackets
            lineFeatures.append(getFrequency(line, r"[\]]"))
            #number of paranthesis
            lineFeatures.append(getFrequency(line, r"[\(\)]"))
            #number of left paranthesis
            lineFeatures.append(getFrequency(line, r"[\(]"))
            #number of right paranthesis
            lineFeatures.append(getFrequency(line, r"[\)]"))
            #number of braces
            lineFeatures.append(getFrequency(line, r"[\{\}]"))
            #number of left braces
            lineFeatures.append(getFrequency(line, r"[\{]"))
            #number of right braces
            lineFeatures.append(getFrequency(line, r"[\}]"))
            #number of all math symbols
            lineFeatures.append(getFrequency(line, r"[\+\*\\\-\=\%\^\<\>]"))
            #number of common math operations
            lineFeatures.append(getFrequency(line, r"[\+\*\\\-\=]"))
            #number of sum operator
            lineFeatures.append(getFrequency(line, r"[\+]"))
            #number of product operator
            lineFeatures.append(getFrequency(line, r"[\*]"))
            #number of division operator
            lineFeatures.append(getFrequency(line, r"[\\]"))
            #number of assignment operator
            lineFeatures.append(getFrequency(line, r"[\=]"))
            #number of greater sign
            lineFeatures.append(getFrequency(line, r"[\>]"))
            #number of lesser sign
            lineFeatures.append(getFrequency(line, r"[\<]"))
            #number of question mark
            lineFeatures.append(getFrequency(line, r"[\?]"))
            #number of single and double quotations
            lineFeatures.append(getFrequency(line, r"[\'\"]"))
            #number of single quotations
            lineFeatures.append(getFrequency(line, r"[\']"))
            #number of double quotations
            lineFeatures.append(getFrequency(line, r"[\"]"))
            #number of ampersands
            lineFeatures.append(getFrequency(line, r"[\&]"))
            #number of at sign
            lineFeatures.append(getFrequency(line, r"[\@]"))
            #number of hash symbol
            lineFeatures.append(getFrequency(line, r"[\#]"))
            #number of dollar sign
            lineFeatures.append(getFrequency(line, r"[\$]"))
            #number of caps sign
            lineFeatures.append(getFrequency(line, r"[\^]"))
            #number of percentage sign
            lineFeatures.append(getFrequency(line, r"[\%]"))
            #number of logical operators
            lineFeatures.append(getFrequency(line, r"[\&\|\!]"))
            #number of excalmation
            lineFeatures.append(getFrequency(line, r"[\!]"))
            features.append(", ".join(str(lineFeature) for lineFeature in lineFeatures))
        f = open(outputFile, "w")
        f.write("\n".join(features))
        f.close()

def deriveFeatures(clusteredSet, offsetLimit):
    keyList = clusteredSet.keys()
    for key in keyList:
        for offset in range(-offsetLimit, offsetLimit + 1):
            clusteredSet[key + " Offset" + str(offset)] = (np.roll(clusteredSet[key], offset))
    return clusteredSet
