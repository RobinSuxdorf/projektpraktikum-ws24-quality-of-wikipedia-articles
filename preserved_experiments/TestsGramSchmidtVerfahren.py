"""Implementing Gram Schmidt procedure for word encoding. Unsuccessfull attemp therefore not fully fledged out

Author: Emmanuelle Steenhof"""

import math
import pandas as pd


def truncate_lists(cat_list):
    """Shortens a list"""
    cat_list_truncated = []
    for i in cat_list:
        temp = len(i)
        padded_word = []
        for word in i.split(' '):
            padded_word.append(i)
        while temp < amount_of_taken_words:
            padded_word.append(0)
            temp = temp+1
        j = padded_word[:amount_of_taken_words]
        cat_list_truncated.append(j)
    return cat_list_truncated
def calculate_scalar_product(v1, v2):
    """calculates the scalar product of the two input vectors"""
    sum_n=0
    for i in range(min(len(v1), len(v2))):
        v3 = v1[i]*v2[i]
        sum_n = sum_n + v3
    return sum_n

def subtract_vectors(v1, scalar, v2):
    """Subtracts vector v2 * scalar from v1"""
    result = []
    for k in range(len(v1)):
        result.append(v1[k]-scalar*v2[k])
    return result

def euclidean_norm(v1):
    """Calculates the euclidian norm of a vector"""
    norm_result = 0.0
    for k in v1:
        norm_result = norm_result + (k*k)
    norm_result = math.sqrt(norm_result)
    return norm_result

def normalize_vector_with_euclidean_norm(v1):
    """normalizes a vector with the euclidean norm"""
    norm_euclid = euclidean_norm(v1)
    result_normed = []
    for l in v1:
        result_normed.append(l/norm_euclid)
    return result_normed



def transform_article_to_vector(dictionary, input_article):
    """transforms an article to a vector by adding the value of each word to the article"""
    output_vector = []
    for i in input_article.split(' '):
        if i == '':
            print(i)
        else:
            output_vector.append(dictionary[i])
    return output_vector


def gram_schmidt(input_vectors):
    """Gram Schmidt algorithm"""
    ortho_vectors = []
    for i in input_vectors:
        w = i
        if len(ortho_vectors) > 0:
            for j in ortho_vectors:
                scalar_curr = calculate_scalar_product(i, j)
                w = subtract_vectors(w, scalar_curr, j)
        w = normalize_vector_with_euclidean_norm(w)
        ortho_vectors.append(w)
    return ortho_vectors


def create_lexicon_for_the_words(list_words):
    """Creates a lexicon to store each word"""
    lexicon = {}
    current_word = 0
    for article in list_words:
        for word in article.split(' '):
            if not word in lexicon.keys():
                lexicon[word] = current_word
                current_word = current_word+1
    return lexicon


def extract_n_arguments_with_label_m(input_articles, label_list, label_value, amount_of_articles):
    """extraxts amount_of_articles articles from a specific label"""
    article_list = []
    label_list_temp = []
    temp = 0
    for i in range(len(input_articles)):
        if (label_list[i] == label_value):
            article_list.append(input_articles[i])
            temp = temp +1
            label_list_temp.append(label_value)
        if (temp == amount_of_articles):
            return article_list, label_list_temp
    return article_list, label_list_temp



def read_in_data():
    df_promo = pd.read_csv("Daten/promotional.csv")
    df_promo["label"] = 0
    df_good = pd.read_csv("Daten/good.csv")
    df_good["label"] = 1
    df_promo2 = df_promo[["text", "label"]]
    df_good2 = df_good[["text", "label"]]
    test = [df_good2, df_promo2]
    df_temp = pd.concat(test)
    return df_temp

def add_list_to_other_list(cat_list, label_list, cat2_list, label_list_2):
    """Merges two lists"""
    temp = 0
    for i in range(len(cat2_list)):
        cat_list.append(cat2_list[i])
        label_list.append(label_list_2[i])
    for i in cat_list:
        temp = temp+1
        article_vectors.append(transform_article_to_vector(test_lexicon, i))
    return cat_list, label_list


"""Here some parameters get defined can be altered as pleased"""
#How many articles per label are taken
amount_of_taken_articles = 10000
#How many words per article are taken
amount_of_taken_words = 1000
#How many base vectors for orthonormalization are being taken per class.
amount_of_base_vectors_per_class = 1000

"""reads in the data and converts it to lists"""
df = read_in_data()
list_words = df["text"].to_list()
list_labels = df["label"].to_list()
"""Creates the lexicon"""
test_lexicon = create_lexicon_for_the_words(list_words)
article_vectors = []
"""extracts the amount of articles that are being taken per label"""
cat_list, label_list = extract_n_arguments_with_label_m(list_words, list_labels, 0, amount_of_taken_articles/2)
cat2_list, label_list_2 = extract_n_arguments_with_label_m(list_words, list_labels, 1, amount_of_taken_articles/2)

cat_list, label_list = add_list_to_other_list(cat_list, label_list, cat2_list, label_list_2)

"""Extracts the vectors used as a base for the Gram Schmidt Verfahren"""
base_vector_cat_1, _ = extract_n_arguments_with_label_m(article_vectors, label_list, 0, amount_of_base_vectors_per_class)
base_vector_cat_2, _ = extract_n_arguments_with_label_m(article_vectors, label_list_2, 1, amount_of_base_vectors_per_class)

"""Merges the base vectors"""
for i in base_vector_cat_2:
    base_vector_cat_1.append(i)

"""Truncates the base vectors to be only amount_of_taken_words long"""
orthogonalized_vectors_base = []
for j in base_vector_cat_1:
    x = []
    for i in range(amount_of_taken_words):
         if i < len(j):
            x.append(j[i])
         else:
             x.append(0)
    orthogonalized_vectors_base.append(x)

"""Orthonomalizes the vectors"""
orthogonalized_vectors_final= gram_schmidt(orthogonalized_vectors_base)

"""Normalizes the vectors"""
normalized_vectors = []
for i in article_vectors:
    normalized_vectors.append(normalize_vector_with_euclidean_norm(i))


"""Truncates the normalized vectors"""
scalar_products_final = []
normalized_vectors_final = []
for j in normalized_vectors:
    x = []
    for i in range(amount_of_taken_words):
        if len(j) > i:
            x.append(j[i])
        else:
            x.append(0)
    normalized_vectors_final.append(x)

"""Calculates the dot product between the orthonormalized vectors and the others"""
for i in normalized_vectors_final:
    coordinates = []
    coordinates.append(calculate_scalar_product(normalized_vectors_final[0],i))
    coordinates.append(calculate_scalar_product(normalized_vectors_final[1],i))
    coordinates.append(calculate_scalar_product(normalized_vectors_final[2],i))
    coordinates.append(calculate_scalar_product(normalized_vectors_final[3],i))
    scalar_products_final.append(coordinates)


from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

"""Calculates the results with the Gram-Schmidt Verfahren as input"""
print("Testing the accuracy of Gram-Schmidt Verfahren")
X_train, X_test, Y_train, Y_test = train_test_split(scalar_products_final, label_list, test_size=0.2, random_state=42)
test_model = svm.SVC(kernel= 'rbf')
test_model.fit(X_train, Y_train)

test_results = test_model.predict(X_test)


from sklearn.metrics import accuracy_score
print("SVM")
print(accuracy_score(Y_test, test_results))


test_model_2 = LogisticRegression()
test_model_2.fit(X_train, Y_train)

test_results_2 = test_model_2.predict(X_test)
print("LR")
print(accuracy_score(Y_test, test_results_2))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
print("Bayes")
gnb.fit(X_train, Y_train)
test_results_3 = gnb.predict(X_test)
print(accuracy_score(Y_test, test_results_3))



cat_list = truncate_lists(cat_list)

"""Calculates the results of the models with normal encoding (meaning word1: 1, word2:2 ...."""
encoded_vector = []
for i in article_vectors:
    j = i[:amount_of_taken_words]
    while len(j)< amount_of_taken_words:
        j.append(0)
    encoded_vector.append(j)

print("Testing the accuracy of normal encoding")
from sklearn.metrics import accuracy_score
X_train, X_test, Y_train, Y_test = train_test_split(encoded_vector, label_list, test_size=0.2, random_state=42)


test_model = svm.SVC(kernel= 'rbf')
test_model.fit(X_train, Y_train)
test_results = test_model.predict(X_test)
print("SVM")
print(accuracy_score(Y_test, test_results))


test_model_2 = LogisticRegression()
test_model_2.fit(X_train, Y_train)

test_results_2 = test_model_2.predict(X_test)
print("LR")
print(accuracy_score(Y_test, test_results_2))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
print("Bayes")
gnb.fit(X_train, Y_train)
test_results_3 = gnb.predict(X_test)
print(accuracy_score(Y_test, test_results_3))