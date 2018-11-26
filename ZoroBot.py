import random
import pickle
import os
import csv
import re
import string
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

GREETINGS = ('hello', 'hi', 'sup', 'greetings', 'sup', "what's up", 'salutations', 'hey', 'konnichiwa', 'how are you')
TRIGGER = ('like', 'dislike', 'love', 'hate')

FAREWELL = ('end', 'bye', 'goodbye', 'cya', 'see ya', 'see you', 'farewell', 'later', 'quit')

FAREWELL_RESPONSES = ["It was nice talking to you!",
                      "Goodbye, friend!",
                      "Please come back soon :("]

GREETING_RESPONSES = ["Hello, I'm Zoro. I'm here to talk to you about sushi.",
                      "Greetings, I'm Zoro! Let's talk about sushi!",
                      "Hi, I'm Zoro. Let's talk about sushi.",
                      "Howdy! I'm Zoro and I like sushi!"]

FALLBACK_RESPONSES = ["I didn't get that. Could you rephrase?",
                      "What was that? I didn't quite catch that.",
                      "I'm sorry, I'm not equipped to handle that."]

SELF_NOUN_RESPONSES = ["Oh, I don't know much about {noun}",
                       "Yes, {noun} seems like an interesting topic",
                       "Do you consider yourself an expert of {noun}?",
                       "I don't want to talk about {noun}, let's talk about sushi instead."]

SELF_ADJ_RESPONSES = ["Oh, {adjective}? Wow.",
                      "What does {adjective} mean?"]

LIKES_DISLIKES = ["what do i like?", "do you remember what I like?", "what do you know about me?", "what do you remember about me?"]

DISLIKE_PHRASES = ["What don't I like?",
                   "What do I hate?",
                   "What do I dislike"]

SUSHI_KEYWORDS = {"sushi", "fish", "ginger", "japan", "tuna", "salmon", "japanese", "chirashi", "inari", "maki", "futomaki", "hosomaki", "nigiri", "sashimi", "wasabi", "uni", "sea", "urchin", "unagi", "sea urchin", "tobiko", "masago", "roe", "tako", "rice", "roll", "rolls", "shoyu", "nori", "fugu", "gari", "abalone", "amaebi", "akagai", "diets", }
java_path = './Java/jre1.8.0_191/bin/java.exe'
os.environ['JAVAHOME'] = java_path


def zoro(sentence, username):
    user_dir = username.replace(" ", "-").lower()
    # print('Zoro parsing sentence: ' + sentence)
    r = response(sentence, user_dir)

    return r


def tf_idf(sentence):
    path1 = './train'
    if not os.path.exists(path1):
        os.makedirs(path1)

    csv_path = "train/train.csv"
    tfidf_vec_path = "train/tfidf_vec.pickle"
    tfidf_matrix_path = "train/tfidf_matrix.pickle"

    i = 0
    sentences = []

    sen_set = (sentence, "")

    sentences.append(" Offset 1.")
    sentences.append(" Offset 2.")

    try:
        f = open(tfidf_vec_path, 'rb')
        tfidf_vectorizer = pickle.load(f)
        f.close()

        f = open(tfidf_matrix_path, 'rb')
        tfidf_matrix = pickle.load(f)
        f.close()
    except:
        with open(csv_path, "r") as file_sentences:
            reader = csv.reader(file_sentences, delimiter=',')

            for row in reader:
                sentences.append(row[0])
                i += 1

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        f = open(tfidf_vec_path, 'wb')
        pickle.dump(tfidf_vectorizer, f)
        f.close()

        f = open(tfidf_matrix_path, 'wb')
        pickle.dump(tfidf_matrix, f)
        f.close()

    tfidf_matrix_user = tfidf_vectorizer.transform(sen_set)
    cosine = cosine_similarity(tfidf_matrix_user, tfidf_matrix)
    cosine = np.delete(cosine, 0)
    max = cosine.max()

    resp_index = 0
    if max > 0.8:
        curr_max = max - 0.01
        list = np.where(cosine > curr_max)
        resp_index = random.choice(list[0])
    else:
        return None, None

    j = 0
    with open(csv_path, "r") as file_sentences:
        reader = csv.reader(file_sentences, delimiter=',')
        for row in reader:
            j += 1
            if j == resp_index:
                return row[1], resp_index
                break


def custom_dialog(sentence, user_dir):

    sen_set = (sentence, "")

    f = open("userdata/{}/likes.txt".format(user_dir), "r")
    likes = []
    for word in f.readlines():
        if word != '\n':
            likes.append(word.replace('\n', ''))

    f = open("userdata/{}/dislikes.txt".format(user_dir), "r")
    dislikes = []
    for word in f.readlines():
        if word != '\n':
            dislikes.append(word.replace('\n', ''))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(LIKES_DISLIKES)

    tfidf_matrix_user = tfidf_vectorizer.transform(sen_set)
    cosine = cosine_similarity(tfidf_matrix_user, tfidf_matrix)
    cosine = np.delete(cosine, 0)
    max = cosine.max()

    if max > 0.7:
        if likes and dislikes:
            if random.choice([True, False]):
                return "Based on previous encounters, I know you like {}!".format(random.choice(likes))
            else:
                return "Based on previous encounters, I know that you dislike {}!".format(random.choice(dislikes))
        elif likes:
            return "Based on previous encounters, I know you like {}!".format(random.choice(likes))
        elif dislikes:
            return "Based on previous encounters, I know that you dislike {}!".format(random.choice(dislikes))


def response(sentence, user_dir):
    cleaned = pronoun_edge(sentence)
    parsed = TextBlob(cleaned)

    pronoun, noun, adjective, verb = get_pos(parsed)

    # resp = self_comment(pronoun, noun, adjective)

    resp = greetings(parsed)

    if not resp:
        resp = custom_dialog(sentence.lower(), user_dir)

    if not resp:
        resp = farewell(parsed)

    if not resp:
        resp, line_id_primary = tf_idf(sentence.lower())

    if not resp:
        resp = self_comment(pronoun, noun, adjective)
    print(resp)

    return resp


def makefile(sentence, username):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sent_nopunct = regex.sub('', sentence)
    blob = TextBlob(sentence)
    tokens = word_tokenize(sent_nopunct)

    like_file = open(path2 + '/likes.txt', 'a+', encoding='utf8')
    dislike_file = open(path2 + '/dislikes.txt', 'a+', encoding='utf8')

    if any(word in sent_nopunct for word in TRIGGER) and 'you' not in sent_nopunct and sent_nopunct.split()[-1] not in TRIGGER:
        pos_tags = nltk.pos_tag(tokens)

        dobj = ''
        for x, y in pos_tags:
            if y == 'NN' or y == 'NNS':
                dobj = x

        # print(pos_tags)

        polarity = blob.sentences[0].sentiment.polarity
        if 'dislike' in blob.sentences[0]:
            polarity -= .5
        if 'like' in blob.sentences[0]:
            polarity += .5
        if 'don\'t' in blob.sentences[0] and 'like' in blob.sentences[0]:
            polarity -= .6
        if 'don\'t' in blob.sentences[0] and 'dislike' in blob.sentences[0]:
            polarity += .6
        if 'don\'t' in blob.sentences[0] and 'love' in blob.sentences[0]:
            polarity -= .6
        if 'don\'t' in blob.sentences[0] and 'hate' in blob.sentences[0]:
            polarity += .6

        # print(polarity)

        if polarity >= 0:
            like_file.write(dobj + '\n')
        else:
            dislike_file.write(dobj + '\n')

    like_file.close()
    dislike_file.close()


def prompt():
    st = StanfordNERTagger('./stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                           './stanford-ner/stanford-ner.jar',
                           encoding='utf-8')

    recognized = False

    while not recognized:
        username_sentence = input('What\'s your name?\n>:')
        tokens = word_tokenize(username_sentence)
        ner_tagged = st.tag(tokens)
        # print(ner_tagged)
        name = ''
        for x,y in ner_tagged:
            if y == 'PERSON':
                name += x + ' '

        if not name:
            recognized = False
            print('Sorry, I didn\'t get that.')
        else:
            recognized = True

    name = name.strip()

    return name


def farewell(sentence):
    for word in sentence.words:
        if word.lower() in FAREWELL:
            return random.choice(FAREWELL_RESPONSES)


def greetings(sentence):
    for word in sentence.words:
        if word.lower() in GREETINGS:
            return random.choice(GREETING_RESPONSES)


def pronoun_edge(sentence):

    cleaned = []
    words = sentence.split(' ')
    for w in words:
        if w == "i":
            w = "I"
        if w == "i'm":
            w = "I'm"
        cleaned.append(w)
    return ' '.join(cleaned)


def get_pos(parsed):
    pronoun = None
    noun = None
    adjective = None
    verb = None

    for sentence in parsed.sentences:
        pronoun = f_pronoun(sentence)
        noun = f_noun(sentence)
        adjective = f_adjective(sentence)
        verb = f_verb(sentence)
    # print(parsed.pos_tags)
    # print("Pronoun: " + str(pronoun) + " Noun: " + str(noun) + " Adjective: " + str(adjective) + " Verb: " + str(verb))
    return pronoun, noun, adjective, verb


def f_pronoun(sentence):
    pronoun = None
    for word, pos in sentence.pos_tags:
        if pos == 'PRP' and word.lower() == 'you':
            pronoun = 'You'
        elif pos == 'PRP' and word == 'I':
            pronoun = 'I'
    return pronoun


def f_verb(sentence):
    verb = None
    pos = None
    for word, pos_tag in sentence.pos_tags:
        if pos_tag.startswith('VB'):
            verb = word
            pos = pos_tag
            break
    return verb, pos


def f_noun(sentence):
    noun = None

    if not noun:
        for word, pos in sentence.pos_tags:
            if (pos == 'NN' or pos == 'NNS' or pos == 'NNP') and word != 'i':
                noun = (word, pos)
                break
    return noun


def f_adjective(sentence):
    adj = None
    for word, pos in sentence.pos_tags:
        if pos == 'JJ':
            adj = word
            break
    return adj


def self_comment(pronoun, noun, adjective):
    re = None
    # print(noun)
    if noun and noun[0] not in SUSHI_KEYWORDS:
        if noun[1] == "NNS":
            re = random.choice(SELF_NOUN_RESPONSES).format(**{'noun': noun[0].lower()})
        else:
            re = "What do you do with a {}? Don't answer that.".format(noun[0].lower())
    elif noun:
        re = "Yes, {} is pretty great!".format(noun[0])
    elif adjective and (not noun or noun[0] not in SUSHI_KEYWORDS):
        re = random.choice(SELF_ADJ_RESPONSES).format(**{'adjective': adjective})
    else:
        re = random.choice(FALLBACK_RESPONSES)
    return re


if __name__ == '__main__':

    path = './userdata'
    if not os.path.exists(path):
        os.makedirs(path)

    username = prompt()
    path2 = path + '/' + username.lower().replace(' ', '-')

    if os.path.exists(path2):
        print('Welcome back, %s!\n' % username)
    else:
        os.makedirs(path2)
        print('Hello, %s!\n' % username)

    while True:
        x = input(">: ")
        zoro(x, username)
        print()
        makefile(x, username)
        if x.lower() in FAREWELL:
            break
