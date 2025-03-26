"""Trying to implement the Gram-Schmidt process for the word encoding. Unsuccessfull attempt therefore not fully fledged out.

Author: Emmanuelle Steenhof"""

import math
import pandas as pd


def sort_the_word_list(list_to_be_sorted_input, size_of_list):
    ##https: // www.geeksforgeeks.org / sorting - algorithms - in -python /
    ##Selection Sort
    sorted_list_test = list_to_be_sorted_input
    for j1 in range(size_of_list):
        min_pos = j1
        for j2 in range(j1 + 1, size_of_list):
            if list_to_be_sorted_input[j2] < list_to_be_sorted_input[min_pos]:
                min_pos = j2
        (sorted_list_test[j1], sorted_list_test[min_pos]) = (
            sorted_list_test[min_pos],
            sorted_list_test[j1],
        )

    return sorted_list_test


def transform_df_to_list(test_dataframe):
    list_temp = []
    for i in range(test_dataframe.__len__()):
        list_temp.append(df.iloc[i])
    z3 = []
    z4 = []
    for i in list_temp:
        z3.append(i[0])
        z4.append(i[1])

    return z3, z4


def create_lexicon(z):
    words = []
    for i in z:
        words.append(i)
    all_words = []
    all_words.append(" ")
    for i in words:
        temp_test = 0
        for j in all_words:
            if i == j:
                temp_test = 1
        if temp_test == 0:
            all_words.append(i)
    return all_words


def calculate_scalar_product(v1, v2):
    sum_n = 0
    for i in range(min(len(v1), len(v2))):
        v3 = v1[i] * v2[i]
        sum_n = sum_n + v3
    return sum_n


def subtract_vectors(v1, scalar, v2):
    result = []
    for k in range(len(v1)):
        result.append(v1[k] - scalar * v2[k])
    return result


def euclidean_norm(v1):
    norm_result = 0.0
    for k in v1:
        norm_result = norm_result + (k * k)
    norm_result = math.sqrt(norm_result)
    return norm_result


def normalize_vector_with_euclidean_norm(v1):
    norm_euclid = euclidean_norm(v1)
    result_normed = []
    for l in v1:
        result_normed.append(l / norm_euclid)
    return result_normed


def calculate_coordinates(v1, v_group):
    x = []
    for i in v_group:
        x.append(calculate_scalar_product(v1, i))
    return x


def transform_article_to_vector(dictionary, input_article):
    output_vector = []
    for i in input_article:
        for j in range(len(dictionary)):
            if i == dictionary[j]:
                output_vector.append(j)
    return output_vector


def gram_schmidt(z1):
    ortho_vectors = []
    for i in z1:
        w = i
        if len(ortho_vectors) > 0:
            for j in ortho_vectors:
                scalar_curr = calculate_scalar_product(i, j)
                w = subtract_vectors(w, scalar_curr, j)
        w = normalize_vector_with_euclidean_norm(w)
        ortho_vectors.append(w)
    return ortho_vectors


def calculate_datapoint(orthogonalized_vectors, datapoint_to_append):
    coordinates_of_point = []
    for i in orthogonalized_vectors:
        coordinates_of_point.append(i, datapoint_to_append)
    return coordinates_of_point


# df = load_data_e_test
def get_lexicon_from_articles(list_words, list_labels):
    list_words_2 = []
    list_labels_2 = []
    #
    for i in range(int(amount_of_taken_articles / 2)):
        for j in list_words[i].split(" "):
            list_words_2.append(j)
        list_labels_2.append(list_labels[i])

    for i in range(
        list_words.__len__() - int(amount_of_taken_articles / 2), list_words.__len__()
    ):
        for j in list_words[i].split(" "):
            list_words_2.append(j)
        list_labels_2.append(list_labels[i])
    lexicon = create_lexicon(list_words_2)
    return lexicon


def extract_list_of_words_articles(article):
    article_as_list = []
    for i in article.split(" "):
        article_as_list.append(i)
    return article_as_list


def extract_n_arguments_with_label_m(
    input_articles, label_list, label_value, amount_of_articles
):
    article_list = []
    label_list_temp = []
    temp = 0
    for i in range(len(input_articles)):
        if label_list[i] == label_value:
            article_list.append(input_articles[i])
            temp = temp + 1
            label_list_temp.append(label_value)
        if temp == amount_of_articles:
            return article_list, label_list_temp
    return article_list, label_list_temp


amount_of_taken_articles = 50
amount_of_taken_words = 100
amount_of_base_vectors_per_class = 1


def read_in_data():
    df_promo = pd.read_csv("Daten/promotional.csv")
    df_promo = df_promo.loc[df_promo["pr"] == 1]
    df_promo = df_promo.loc[df_promo["fanpov"] == 1]
    df_promo = df_promo.loc[df_promo["advert"] == 0]
    df_promo["label"] = 0
    df_good = pd.read_csv("Daten/promotional.csv")
    df_good = df_good.loc[df_good["fanpov"] == 1]
    df_good = df_good.loc[df_good["pr"] == 0]
    df_good = df_good.loc[df_good["advert"] == 0]
    df_good["label"] = 1
    df_promo2 = df_promo[["text", "label"]]
    df_good2 = df_good[["text", "label"]]
    test = [df_good2, df_promo2]
    df_temp = pd.concat(test)
    return df_temp


df = read_in_data()
list_words = df["text"].tolist()
list_labels = df["label"].tolist()
test_lexicon = get_lexicon_from_articles(list_words, list_labels)
article_vectors = []
cat_list, label_list = extract_n_arguments_with_label_m(
    list_words, list_labels, 0, amount_of_taken_articles / 2
)

cat2_list, label_list_2 = extract_n_arguments_with_label_m(
    list_words, list_labels, 1, amount_of_taken_articles / 2
)
temp = 0
for i in range(len(cat2_list)):
    cat_list.append(cat2_list[i])
    label_list.append(label_list_2[i])
for i in cat_list:
    temp = temp + 1
    article_vectors.append(transform_article_to_vector(test_lexicon, i))
base_vector_cat_1, _ = extract_n_arguments_with_label_m(
    article_vectors, label_list, 0, amount_of_base_vectors_per_class
)
base_vector_cat_2, _ = extract_n_arguments_with_label_m(
    article_vectors, label_list_2, 1, amount_of_base_vectors_per_class
)

for i in base_vector_cat_2:
    base_vector_cat_1.append(i)


orthogonalized_vectors_base = []

for j in base_vector_cat_1:
    x = []
    for i in range(amount_of_taken_words):
        x.append(j[i])
    orthogonalized_vectors_base.append(x)

orthogonalized_vectors_final = gram_schmidt(orthogonalized_vectors_base)

normalized_vectors = []
for i in article_vectors:
    normalized_vectors.append(normalize_vector_with_euclidean_norm(i))

scalar_products_final = []

normalized_vectors_final = []
for j in normalized_vectors:
    x = []
    for i in range(amount_of_taken_words):
        x.append(j[i])
    normalized_vectors_final.append(x)


for i in normalized_vectors_final:
    coordinates = []
    coordinates.append(calculate_scalar_product(normalized_vectors_final[0], i))
    coordinates.append(calculate_scalar_product(normalized_vectors_final[1], i))
    coordinates.append(calculate_scalar_product(normalized_vectors_final[2], i))
    coordinates.append(calculate_scalar_product(normalized_vectors_final[3], i))
    scalar_products_final.append(coordinates)


from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

X_train, X_test, Y_train, Y_test = train_test_split(
    scalar_products_final, label_list, test_size=0.4, random_state=42
)
test_model = svm.SVC(kernel="rbf")
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
