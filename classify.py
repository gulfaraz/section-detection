import pandas
import tensorflow as tf

CONTINUOUS_COLUMNS = [
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
]

characters = tf.contrib.layers.real_valued_column("characters")
alphanumerics = tf.contrib.layers.real_valued_column("alphanumerics")
special_characters = tf.contrib.layers.real_valued_column("special_characters")
alphabets = tf.contrib.layers.real_valued_column("alphabets")
capital_letters = tf.contrib.layers.real_valued_column("capital_letters")
vowels = tf.contrib.layers.real_valued_column("vowels")
numeric_characters = tf.contrib.layers.real_valued_column("numeric_characters")
words_with_special_characters = tf.contrib.layers.real_valued_column("words_with_special_characters")
words = tf.contrib.layers.real_valued_column("words")
commas = tf.contrib.layers.real_valued_column("commas")
colons = tf.contrib.layers.real_valued_column("colons")
semi_colons = tf.contrib.layers.real_valued_column("semi_colons")
hyphens = tf.contrib.layers.real_valued_column("hyphens")
underscores = tf.contrib.layers.real_valued_column("underscores")
pipes = tf.contrib.layers.real_valued_column("pipes")
brackets_and_paranthesis_and_braces = tf.contrib.layers.real_valued_column("brackets_and_paranthesis_and_braces")
brackets = tf.contrib.layers.real_valued_column("brackets")
left_brackets = tf.contrib.layers.real_valued_column("left_brackets")
right_brackets = tf.contrib.layers.real_valued_column("right_brackets")
paranthesis = tf.contrib.layers.real_valued_column("paranthesis")
left_paranthesis = tf.contrib.layers.real_valued_column("left_paranthesis")
right_paranthesis = tf.contrib.layers.real_valued_column("right_paranthesis")
braces = tf.contrib.layers.real_valued_column("braces")
left_braces = tf.contrib.layers.real_valued_column("left_braces")
right_braces = tf.contrib.layers.real_valued_column("right_braces")
all_math_symbols = tf.contrib.layers.real_valued_column("all_math_symbols")
common_math_operations = tf.contrib.layers.real_valued_column("common_math_operations")
sum_operator = tf.contrib.layers.real_valued_column("sum_operator")
product_operator = tf.contrib.layers.real_valued_column("product_operator")
division_operator = tf.contrib.layers.real_valued_column("division_operator")
assignment_operator = tf.contrib.layers.real_valued_column("assignment_operator")
greater_sign = tf.contrib.layers.real_valued_column("greater_sign")
lesser_sign = tf.contrib.layers.real_valued_column("lesser_sign")
question_mark = tf.contrib.layers.real_valued_column("question_mark")
single_and_double_quotations = tf.contrib.layers.real_valued_column("single_and_double_quotations")
single_quotations = tf.contrib.layers.real_valued_column("single_quotations")
double_quotations = tf.contrib.layers.real_valued_column("double_quotations")
ampersands = tf.contrib.layers.real_valued_column("ampersands")
at_sign = tf.contrib.layers.real_valued_column("at_sign")
hash_symbol = tf.contrib.layers.real_valued_column("hash_symbol")
dollar_sign = tf.contrib.layers.real_valued_column("dollar_sign")
caps_sign = tf.contrib.layers.real_valued_column("caps_sign")
percentage_sign = tf.contrib.layers.real_valued_column("percentage_sign")
logical_operators = tf.contrib.layers.real_valued_column("logical_operators")
excalamation = tf.contrib.layers.real_valued_column("excalmation")

characters_buckets = tf.contrib.layers.bucketized_column(characters, boundaries=[10, 20, 30, 40, 50, 60, 70, 80, 90])
alphanumerics_buckets = tf.contrib.layers.bucketized_column(alphanumerics, boundaries=[10, 20, 30, 40, 50, 60, 70, 80, 90])
alphabets_buckets = tf.contrib.layers.bucketized_column(alphabets, boundaries=[10, 20, 30, 40, 50, 60, 70, 80, 90])
words_with_special_characters_buckets = tf.contrib.layers.bucketized_column(words_with_special_characters, boundaries=[5, 10, 15, 20, 25, 30, 35, 40, 45])
words_buckets = tf.contrib.layers.bucketized_column(words, boundaries=[5, 10, 15, 20, 25, 30, 35, 40, 45])
capital_letters_buckets = tf.contrib.layers.bucketized_column(capital_letters, boundaries=[5, 10, 15, 20])
special_characters_buckets = tf.contrib.layers.bucketized_column(special_characters, boundaries=[5, 10, 15, 20])
vowels_buckets = tf.contrib.layers.bucketized_column(vowels, boundaries=[5, 10, 15, 20])

characters_buckets_x_words_with_special_characters_buckets = tf.contrib.layers.crossed_column([characters_buckets, words_with_special_characters_buckets], hash_bucket_size=int(1e6))
alphabets_buckets_x_words_buckets = tf.contrib.layers.crossed_column([alphabets_buckets, words_buckets], hash_bucket_size=int(1e6))
words_buckets_x_capital_letters_buckets_x_special_characters_buckets_x_vowels_buckets = tf.contrib.layers.crossed_column([words_buckets, capital_letters_buckets, special_characters_buckets, vowels_buckets], hash_bucket_size=int(1e6))

modelDirectory = "./models/classifier"

m = tf.contrib.learn.LinearClassifier(feature_columns=[
    characters,
    alphanumerics,
    special_characters,
    alphabets,
    capital_letters,
    vowels,
    numeric_characters,
    words_with_special_characters,
    words,
    commas,
    colons,
    semi_colons,
    hyphens,
    underscores,
    pipes,
    brackets_and_paranthesis_and_braces,
    brackets,
    left_brackets,
    right_brackets,
    paranthesis,
    left_paranthesis,
    right_paranthesis,
    braces,
    left_braces,
    right_braces,
    all_math_symbols,
    common_math_operations,
    sum_operator,
    product_operator,
    division_operator,
    assignment_operator,
    greater_sign,
    lesser_sign,
    question_mark,
    single_and_double_quotations,
    single_quotations,
    double_quotations,
    ampersands,
    at_sign,
    hash_symbol,
    dollar_sign,
    caps_sign,
    percentage_sign,
    logical_operators,
    excalamation,
    characters_buckets,
    alphanumerics_buckets,
    alphabets_buckets,
    words_with_special_characters_buckets,
    words_buckets,
    capital_letters_buckets,
    special_characters_buckets,
    vowels_buckets,
    characters_buckets_x_words_with_special_characters_buckets,
    alphabets_buckets_x_words_buckets,
    words_buckets_x_capital_letters_buckets_x_special_characters_buckets_x_vowels_buckets
],
n_classes=3,
optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
model_dir=modelDirectory)

def inputSet(dataFrame, hasLabel):
    continuous_cols = {k: tf.constant(dataFrame[k].values)
                        for k in CONTINUOUS_COLUMNS}
    feature_cols = dict(continuous_cols.items())
    if hasLabel:
        label = tf.constant(dataFrame["label"].values)
        return feature_cols, label
    else:
        return feature_cols

def train(trainingSet, iterations):

    def trainingInputSet():
        return inputSet(trainingSet, True)

    return m.fit(input_fn=trainingInputSet, steps=iterations)

def evaluate(testingSet, iterations):

    def evaluateInputSet():
        return inputSet(testingSet, True)

    return m.evaluate(input_fn=evaluateInputSet, steps=iterations)

def predict(predictionSet):

    def predictionInputSet():
        return inputSet(predictionSet, False)

    return m.predict_classes(input_fn=predictionInputSet)
