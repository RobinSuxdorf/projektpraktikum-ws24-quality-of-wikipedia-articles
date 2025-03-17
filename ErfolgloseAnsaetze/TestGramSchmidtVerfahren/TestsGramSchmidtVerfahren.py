import math
import pandas as pd

def sort_the_word_list(list_to_be_sorted_input, size_of_list):
    ##https: // www.geeksforgeeks.org / sorting - algorithms - in -python /
    ##Selection Sort
    sorted_list_test = list_to_be_sorted_input
    for j1 in range(size_of_list):
        min_pos = j1
        for j2 in range (j1+1, size_of_list):
            if (list_to_be_sorted_input[j2] < list_to_be_sorted_input[min_pos]):
                min_pos = j2
        (sorted_list_test[j1],sorted_list_test[min_pos]) = (sorted_list_test[min_pos],sorted_list_test[j1])

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

    return z3,z4

def create_lexicon(z):
    words = []
    for i in z:
        words.append(i)
    all_words = []
    all_words.append(' ')
    for i in words:
        temp_test = 0
        for j in all_words:
            if i == j:
                temp_test = 1
        if temp_test == 0:
            all_words.append(i)
    all_words_single = []
    """for i in all_words:
        for j in i:
            all_words_single.append(j)
    distinct_words = []
    all_words_sorted = sort_the_word_list(all_words_single, len(all_words_single))
    all_words_sorted2 = []
    print("Start sorting")
    for i in all_words_sorted:
        for j in i:
            all_words_sorted2.append(j)
    for i in range(len(all_words_sorted2)):
        if i >0:
            if (all_words_sorted[i] != all_words_sorted[i-1]):
                distinct_words.append(all_words_sorted[i])
        else:
            distinct_words.append(all_words_sorted[i])"""
    return(all_words)

def calculate_scalar_product(v1, v2):
    sum_n=0
    for i in range(min(len(v1), len(v2))):
        v3 = v1[i]*v2[i]
        sum_n = sum_n + v3
    return sum_n


def subtract_vectors(v1, scalar, v2):
    result = []
    for k in range(len(v1)):
        result.append(v1[k]-scalar*v2[k])
    return result

def euclidean_norm(v1):
    norm_result = 0.0
    for k in v1:
        norm_result = norm_result + (k*k)
    norm_result = math.sqrt(norm_result)
    return norm_result

def normalize_vector_with_euclidean_norm(v1):
    norm_euclid = euclidean_norm(v1)
    result_normed = []
    for l in v1:
        result_normed.append(l/norm_euclid)
    return result_normed



def calculate_coordinates(v1, v_group):
    x = []
    for i in v_group:
        x.append(calculate_scalar_product(v1,i))
    return(x)



def transform_article_to_vector(dictionary, input_article):
    output_vector = []
    for i in input_article:
        for j in range(len(dictionary)):
            if (i == dictionary[j]):
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


#df = load_data_e_test
def get_lexicon_from_articles(list_words, list_labels):

    print(list_words)
    print("test")
##list_words = list_words[:200]
    list_words_2 = []
    list_labels_2 = []
    #
    for i in range(int(amount_of_taken_articles/2)):
        for j in list_words[i].split(' '):
            #print(j)
            list_words_2.append(j)
        list_labels_2.append(list_labels[i])

    for i in range(list_words.__len__()-int(amount_of_taken_articles/2), list_words.__len__()):
        for j in list_words[i].split(' '):
            #print(j)
            list_words_2.append(j)
        list_labels_2.append(list_labels[i])
    print(list_words_2)
    print("Amt of Articles:"+ str(list_labels_2.__len__()))
    lexicon = create_lexicon(list_words_2)
    print("test")
    print(lexicon)
    print("test")
    return lexicon

def extract_list_of_words_articles(article):
    article_as_list = []
    for i in article.split(' '):
        article_as_list.append(i)
    return article_as_list

def extract_n_arguments_with_label_m(input_articles, label_list, label_value, amount_of_articles):
    article_list = []
    label_list_temp = []
    temp = 0
    for i in range(len(input_articles)):
        print(label_list[i])
        if (label_list[i] == label_value):
            article_list.append(input_articles[i])
            temp = temp +1
            label_list_temp.append(label_value)
        if (temp == amount_of_articles):
            print("amount_of_articles")
            print(amount_of_articles)
            return article_list, label_list_temp
    return article_list, label_list_temp


    """def load_data_e_test():
    df_promo = pd.read_csv("Daten/promotional.csv")
    print(df_promo)
    df_good = pd.read_csv("Daten/good.csv")
    df_good_test = df_good["text"]
    df_promotion = df_promo["text"]
    print(df_good_test)
    df_good_test["label"]= 0
    df_promotion["label"]=1
    test = [df_good_test, df_promotion]
    df_test = pd.concat(test, axis=0, ignore_index=True)

    return df_test"""


amount_of_taken_articles = 50
amount_of_taken_words = 100
amount_of_base_vectors_per_class = 1
print("TEST1")
def read_in_data():
    df_promo = pd.read_csv("Daten/promotional.csv")
    df_promo = df_promo.loc[df_promo["pr"] == 1]
    df_promo = df_promo.loc[df_promo["fanpov"] == 1]
    df_promo = df_promo.loc[df_promo["advert"] == 0]
    df_promo["label"] = 0
###df_good = pd.read_csv("Daten/good.csv")
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
list_words, list_labels = transform_df_to_list(df)
test_lexicon = get_lexicon_from_articles(list_words, list_labels)
article_vectors = []
cat_list, label_list = extract_n_arguments_with_label_m(list_words, list_labels, 0, amount_of_taken_articles/2)

cat2_list, label_list_2 = extract_n_arguments_with_label_m(list_words, list_labels, 1, amount_of_taken_articles/2)
temp = 0
for i in range(len(cat2_list)):
    cat_list.append(cat2_list[i])
    label_list.append(label_list_2[i])
for i in cat_list:
    temp = temp+1
    article_vectors.append(transform_article_to_vector(test_lexicon, i))
base_vector_cat_1, _ = extract_n_arguments_with_label_m(article_vectors, label_list, 0, amount_of_base_vectors_per_class)
base_vector_cat_2, _ = extract_n_arguments_with_label_m(article_vectors, label_list_2, 1, amount_of_base_vectors_per_class)

for i in base_vector_cat_2:
    base_vector_cat_1.append(i)


orthogonalized_vectors_base = []

for j in base_vector_cat_1:
    x = []
    for i in range(amount_of_taken_words):
         x.append(j[i])
    orthogonalized_vectors_base.append(x)

orthogonalized_vectors_final= gram_schmidt(orthogonalized_vectors_base)
#print(orthogonalized_vectors_base)

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
    print(len(normalized_vectors_final))
    coordinates = []
    coordinates.append(calculate_scalar_product(normalized_vectors_final[0],i))
    coordinates.append(calculate_scalar_product(normalized_vectors_final[1],i))
    coordinates.append(calculate_scalar_product(normalized_vectors_final[2],i))
    coordinates.append(calculate_scalar_product(normalized_vectors_final[3],i))
    scalar_products_final.append(coordinates)

print(scalar_products_final)



from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X_train, X_test, Y_train, Y_test = train_test_split(scalar_products_final, label_list, test_size=0.4, random_state=42)
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









"""v1 = transform_article_to_vector(lexicon, list_words_2[0])
v2 = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-1])
v3_base = transform_article_to_vector(lexicon, list_words_2[1])
v4_base = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-2])
v5_base = transform_article_to_vector(lexicon, list_words_2[2])
v6_base = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-3])
v7_base = transform_article_to_vector(lexicon, list_words_2[3])
v8_base = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-4])
v9_base = transform_article_to_vector(lexicon, list_words_2[4])
v10_base = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-5])
v11_base = transform_article_to_vector(lexicon, list_words_2[5])
v12_base = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-6])
v13_base = transform_article_to_vector(lexicon, list_words_2[6])
v14_base = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-7])
v15_base = transform_article_to_vector(lexicon, list_words_2[7])
v16_base = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-8])
v17_base = transform_article_to_vector(lexicon, list_words_2[8])
v18_base = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-9])
v19_base = transform_article_to_vector(lexicon, list_words_2[9])
v20_base = transform_article_to_vector(lexicon, list_words_2[amount_of_taken_articles-10])
print(v1)
print("does this work")
v_ges = []
v3 = []
v4 = []
v3_extra = []
v4_extra = []
v5_extra = []
v6_extra = []
v7_extra = []
v8_extra = []
v9_extra = []
v10_extra = []
v11_extra = []
v12_extra = []
v13_extra = []
v14_extra = []
v15_extra = []
v16_extra = []
v17_extra = []
v18_extra = []
v19_extra = []
v20_extra = []
for i in range(amount_of_taken_articles):
    v3.append(v1[i])
    v4.append(v2[i])
    v3_extra.append(v3_base[i])
    v4_extra.append(v4_base[i])
    v5_extra.append(v5_base[i])
    v6_extra.append(v6_base[i])
    v7_extra.append(v7_base[i])
    v8_extra.append(v8_base[i])
    v9_extra.append(v9_base[i])
    v10_extra.append(v10_base[i])
    v11_extra.append(v3_base[i])
    v12_extra.append(v4_base[i])
    v13_extra.append(v5_base[i])
    v14_extra.append(v6_base[i])
    v15_extra.append(v7_base[i])
    v16_extra.append(v8_base[i])
    v17_extra.append(v9_base[i])
    v18_extra.append(v10_base[i])
    v19_extra.append(v9_base[i])
    v20_extra.append(v10_base[i])
v_ges.append(v3)
v_ges.append(v4)
v_ges.append(v3_extra)
v_ges.append(v4_extra)
v_ges.append(v5_extra)
v_ges.append(v6_extra)
v_ges.append(v7_extra)
v_ges.append(v8_extra)
v_ges.append(v9_extra)
v_ges.append(v10_extra)
v_ges.append(v11_extra)
v_ges.append(v12_extra)
v_ges.append(v13_extra)
v_ges.append(v14_extra)
v_ges.append(v15_extra)
v_ges.append(v16_extra)
v_ges.append(v17_extra)
v_ges.append(v18_extra)
v_ges.append(v19_extra)
v_ges.append(v20_extra)
print(v_ges)
ortho_vectors = gram_schmidt(v_ges)"""

"""scalar_products_of_all_vectors_axis_one = []
scalar_products_of_all_vectors_axis_two = []
scalar_products_of_all_vectors_axis_three = []
scalar_products_of_all_vectors_axis_four = []
all_articles_as_vectors = []
for i in list_words_2:
    test_vector = transform_article_to_vector(lexicon, i)
    all_articles_as_vectors.append(normalize_vector_with_euclidean_norm(test_vector))
print(all_articles_as_vectors)
for i in all_articles_as_vectors:
    i1 = []
    for j in range(amount_of_taken_words):
        if len(i) <= j:
            i1.append(0)
        else:
            i1.append(i[j])
    scalar_products_of_all_vectors_axis_one.append(calculate_scalar_product(ortho_vectors[0], i1)
                                                   + calculate_scalar_product(ortho_vectors[2], i1)
                                                   + calculate_scalar_product(ortho_vectors[4], i1)
                                                   + calculate_scalar_product(ortho_vectors[6], i1)
                                                   + calculate_scalar_product(ortho_vectors[8], i1)

                                                   )
    scalar_products_of_all_vectors_axis_two.append(calculate_scalar_product(ortho_vectors[1], i1) +
                                                   calculate_scalar_product(ortho_vectors[3], i1)
                                                  + calculate_scalar_product(ortho_vectors[5], i1)
                                                    +calculate_scalar_product(ortho_vectors[7], i1)
                                                   +calculate_scalar_product(ortho_vectors[9], i1)
                                                   )
    scalar_products_of_all_vectors_axis_three.append(calculate_scalar_product(ortho_vectors[10], i1)
    + calculate_scalar_product(ortho_vectors[12], i1)
    + calculate_scalar_product(ortho_vectors[14], i1) +
                                                   calculate_scalar_product(ortho_vectors[16], i1)
                                                   +
                                                   calculate_scalar_product(ortho_vectors[18], i1))
    scalar_products_of_all_vectors_axis_four.append(calculate_scalar_product(ortho_vectors[11], i1)
                                                   + calculate_scalar_product(ortho_vectors[13], i1)
                                                   + calculate_scalar_product(ortho_vectors[15], i1)
                                                   + calculate_scalar_product(ortho_vectors[17], i1)
                                                   + calculate_scalar_product(ortho_vectors[19], i1))
print(ortho_vectors)

final_scalar_products = []
for i in range(amount_of_taken_articles):
    first_part_scalar =[]
    first_part_scalar.append(scalar_products_of_all_vectors_axis_one[i])
    first_part_scalar.append(scalar_products_of_all_vectors_axis_two[i])
    #first_part_scalar.append(float(list_labels_2[i]))
    final_scalar_products.append(first_part_scalar)

from matplotlib import pyplot as plt
import numpy as np


for i in range(amount_of_taken_articles):
    first_part_scalar =[]
    first_part_scalar.append(scalar_products_of_all_vectors_axis_one[i])
    first_part_scalar.append(scalar_products_of_all_vectors_axis_two[i])
    #first_part_scalar.append(float(list_labels_2[i]))
    final_scalar_products.append(first_part_scalar)

test_array = np.array(final_scalar_products)
print(test_array)
plt.scatter(scalar_products_of_all_vectors_axis_one[0:int(amount_of_taken_articles/2)], scalar_products_of_all_vectors_axis_two[:int(amount_of_taken_articles/2)])
plt.scatter(scalar_products_of_all_vectors_axis_one[int(amount_of_taken_articles/2):amount_of_taken_articles], scalar_products_of_all_vectors_axis_two[int(amount_of_taken_articles/2):amount_of_taken_articles])


data_points = []
data_class = []
for i in range(int(amount_of_taken_articles/2)):
    x = []
    x.append(scalar_products_of_all_vectors_axis_one[i])
    x.append(scalar_products_of_all_vectors_axis_two[i])
    x.append(scalar_products_of_all_vectors_axis_three[i])
    x.append(scalar_products_of_all_vectors_axis_four[i])
    data_points.append(x)
    data_class.append(0)

for i in range(int(amount_of_taken_articles/2), amount_of_taken_articles):
    x = []
    x.append(scalar_products_of_all_vectors_axis_one[i])
    x.append(scalar_products_of_all_vectors_axis_two[i])
    x.append(scalar_products_of_all_vectors_axis_three[i])
    x.append(scalar_products_of_all_vectors_axis_four[i])
    data_points.append(x)
    data_class.append(1)


plt.show()
print(final_scalar_products)
print(euclidean_norm(ortho_vectors[0]))

plt.plot(final_scalar_products)

from sklearn import svm
from sklearn.model_selection import train_test_split
print(data_points)
print(data_class)

from sklearn.linear_model import LogisticRegression
X_train, X_test, Y_train, Y_test = train_test_split(data_points, data_class, test_size=0.3, random_state=42)
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
print(accuracy_score(Y_test, test_results_2))"""
