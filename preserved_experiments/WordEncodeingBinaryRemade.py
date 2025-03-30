"""Trying to classify words as a preparation for word encoding. Unsuccessfull attempt therefore not fully fledged out.


Author: Emmanuelle Steenhof
"""

import pandas as pd


def buildingLexiconLetters():
    """creates a lexicon with letter encodings"""
    alphabet = "abcdefghijklmnopqurstuvwxyz"
    letterLexicon = {}
    for pos in range(len(alphabet)):
        letterLexicon[alphabet[pos]] = pos + 1

    return letterLexicon


def read_in_data_binary(promotional_path, good_path):
    """reads in the data and adds binary labels"""
    df_promo = pd.read_csv(promotional_path)
    df_promo["label"] = 1
    df_promo = df_promo[:10]
    df_good = pd.read_csv(good_path)
    df_good["label"] = 0
    df_good = df_good[:10]
    df_promo = df_promo[["text", "label"]]
    df_good = df_good[["text", "label"]]
    df = pd.concat((df_good, df_promo), axis=0, ignore_index=True)
    df = df.sample(frac=1)
    return df


def read_in_data(promotional_path, good_path):
    """reads in the data and groups the articles based on labels"""
    df_promo = pd.read_csv(promotional_path)
    df_promo_aggregated = df_promo.groupby(["advert", "fanpov", "coi", "pr"]).count()
    df_promo_aggregated["label"] = range(1, len(df_promo_aggregated) + 1)

    df_final = df_promo_aggregated.merge(
        df_promo,
        left_on=["advert", "fanpov", "coi", "pr"],
        right_on=["advert", "fanpov", "coi", "pr"],
        how="inner",
    )

    df_promo_final = df_final[["label", "text_y", "url_y"]]
    df_good = pd.read_csv(good_path)
    df_good = df_good[:10]
    df_good["label"] = 0
    df_promo2 = df_promo_final[["text_y", "label"]]
    df_promo3 = df_promo2.rename(columns={"text_y": "text"})
    df_good2 = df_good[["text", "label"]]
    test = [df_good2, df_promo3]
    df_temp = pd.concat(test)
    return df_temp, len(df_promo_aggregated) + 1


def create_word_label_list(article_test_list, label_test_list):
    """Creates a list with a the words for each articles combined with the label of the articles"""
    word_label_list = []
    for article_no in range(len(article_test_list)):
        for word in article_test_list[article_no].split(" "):
            x = []
            x.append(word)
            x.append(label_test_list[article_no])
            word_label_list.append(x)
    return word_label_list


def create_word_list_with_all_categories(word_label_list):
    """creates a lexicon with the words and all the articles they belong to"""
    lexicon = {}
    for word_label in word_label_list:
        if word_label[0] in lexicon.keys():
            x = lexicon[word_label[0]] + str(word_label[1])
            lexicon[word_label[0]] = x
        else:
            lexicon[word_label[0]] = str(word_label[1])

    return lexicon


def list_words_appears_in_both_categories(lexicon):
    """Creates a list on wheter words appear in only one category or in both"""
    words_in_both = []
    words_in_one_cat = []
    for l in lexicon.keys():
        if "1" in lexicon[l] and "0" in lexicon[l]:
            words_in_both.append(l)
        if not ("1" in lexicon[l] and "0" in lexicon[l]):
            words_in_one_cat.append(l)


def word_in_both_cat(lexicon, word):
    """Checks if a word is in both categories"""
    return "1" in lexicon[word] and "0" in lexicon[word]


def create_final_word_number_mapping(lexicon):
    """Creates the final mapping between words and labels
    meaning of labels
    0: word appears only in the good class
    1 word appears only in the promotional class
    2 word appears in both classes"""
    final_mapping = {}
    for word in lexicon.keys():
        if word_in_both_cat(lexicon, word):
            final_mapping[word] = 2
        else:
            final_mapping[word] = int(lexicon[word][:1])
    return final_mapping


def create_list_for_training_words(lexicon, letter_lexicon):
    """encodes the words based on the alphabet"""
    alphabet_length = len(letter_lexicon)
    all_words_encoded = []
    label_of_words = []
    for word in lexicon.keys():
        word_encoding = []
        for letter in word:
            if letter not in letter_lexicon.keys():
                letter_lexicon[letter] = int(alphabet_length + 1)
                alphabet_length = alphabet_length + 1
            word_encoding.append(letter_lexicon[letter])
        if len(word_encoding) > 32:
            word_encoding = word_encoding[:32]
        else:
            while len(word_encoding) < 32:
                word_encoding.append(0)
        all_words_encoded.append(word_encoding)
        label_of_words.append(lexicon[word])
    return all_words_encoded, label_of_words, letter_lexicon


"""Defining the paths"""
promotional_path = "Daten/good.csv"

good_path = "Daten/promotional.csv"

"""reading in the data"""
df = read_in_data_binary(promotional_path, good_path)


from sklearn import model_selection

"""Splitting the data into testing data and training data"""
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    df["text"], df["label"], shuffle=True, test_size=0.33
)

"""creating the list with the words and all their labels"""
article_test_list = X_train.to_list()
label_test_list = Y_train.to_list()
word_label_list = create_word_label_list(article_test_list, label_test_list)

"""creating the lexicon"""
lexicon = create_word_list_with_all_categories(word_label_list)
"""prints the amount of words in both categories and only one"""
list_words_appears_in_both_categories(lexicon)
final_word_mapping = create_final_word_number_mapping(lexicon)

"""Creates the lexicon with only letters"""
letter_lexicon = buildingLexiconLetters()
all_words_encoded, label_of_words, letter_lexicon = create_list_for_training_words(
    final_word_mapping, letter_lexicon
)

"""Splitting the Words and labels in training and testing data"""
Word_X_train, Word_X_test, Word_Y_train, Word_Y_test = model_selection.train_test_split(
    all_words_encoded, label_of_words, shuffle=True, test_size=0.33
)

from sklearn import svm
from sklearn import metrics

"""Testing the classification of encodings with classic mashine learning algorithms"""

print("training the SVM starts here")
t_model = svm.SVC().fit(Word_X_train, Word_Y_train)
y_predicted = t_model.predict(Word_X_test)
print(metrics.accuracy_score(y_predicted, Word_Y_test))


from sklearn import linear_model

print("training the Logistic Regression starts here")
t_model = linear_model.LogisticRegression().fit(Word_X_train, Word_Y_train)
y_predicted = t_model.predict(Word_X_test)
print(metrics.accuracy_score(y_predicted, Word_Y_test))


from sklearn.naive_bayes import GaussianNB

print("training the GaussianNB starts here")
t_model = GaussianNB().fit(Word_X_train, Word_Y_train)
y_predicted = t_model.predict(Word_X_test)
print(metrics.accuracy_score(y_predicted, Word_Y_test))
