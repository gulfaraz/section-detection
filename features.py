import numpy as np
import pandas
import re

featureHeaders = [
    "characters",
    "alphanumerics",
    "special_characters",
    "alphabets",
    "capital_letters",
    "vowels",
    "numeric_characters",
    "words_with_special_characters",
    "words",
    "commas",
    "colons",
    "semi_colons",
    "hyphens",
    "underscores",
    "pipes",
    "brackets_and_paranthesis_and_braces",
    "brackets",
    "left_brackets",
    "right_brackets",
    "paranthesis",
    "left_paranthesis",
    "right_paranthesis",
    "braces",
    "left_braces",
    "right_braces",
    "all_math_symbols",
    "common_math_operations",
    "sum_operator",
    "product_operator",
    "division_operator",
    "assignment_operator",
    "greater_sign",
    "lesser_sign",
    "question_mark",
    "single_and_double_quotations",
    "single_quotations",
    "double_quotations",
    "ampersands",
    "at_sign",
    "hash_symbol",
    "dollar_sign",
    "caps_sign",
    "percentage_sign",
    "logical_operators",
    "excalmation"
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
            clusteredSet[key + "_offset" + str(offset)] = (np.roll(clusteredSet[key], offset))
    return clusteredSet
