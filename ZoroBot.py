from textblob import TextBlob
import random
import os

GREETINGS = ('hello', 'hi', 'sup', 'greetings', 'sup', "what's up", 'salutations')

GREETING_RESPONSES = ["Hello, I'm Zoro. I'm here to talk to you about sushi.",
                      "Greetings, I'm Zoro. What do you want to talk about? Hopefully sushi...",
                      "Hi, I'm Zoro. Let's talk about sushi.",
                      "Howdy! What questions do you have about sushi?"]

FALLBACK_RESPONSES = ["I didn't get that. Could you rephrase?",
                      "What was that? I didn't quite catch that.",
                      "I'm sorry, I'm not equipped to handle that."]

SELF_NOUN_RESPONSES = ["Oh, I don't know much about {noun}",
                       "Yes, {noun} seems like an interesting topic",
                       "Do you consider yourself an expert of {noun}?",
                       "I don't want to talk about {noun}, let's talk about sushi instead."]

SELF_ADJ_RESPONSES = ["Oh, {adjective}? Wow.",
                      "What does {adjective} mean?"]


def zoro(sentence):
    print('Zoro parsing sentence: ' + sentence)
    r = response(sentence)
    return r


def response(sentence):
    cleaned = pronoun_edge(sentence)
    parsed = TextBlob(cleaned)

    pronoun, noun, adjective, verb = get_pos(parsed)

    resp = self_comment(pronoun, noun, adjective)

    if not resp:
        resp = greetings(parsed)

    if not resp:
        print()
        # main chat bot functionality goes here

    if not resp:
        resp = random.choice(FALLBACK_RESPONSES)
    print(resp)
    return resp


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
    print(parsed.pos_tags)
    print("Pronoun: " + str(pronoun) + " Noun: " + str(noun) + " Adjective: " + str(adjective) + " Verb: " + str(verb))
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
            if pos == 'NN':
                noun = word
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
    if pronoun == 'You' and (noun or adjective):
        if noun:
            re = random.choice(SELF_NOUN_RESPONSES).format(**{'noun': noun})
        else:
            re = random.choice(SELF_ADJ_RESPONSES).format(**{'adjective': adjective})
    return re


if __name__ == '__main__':
    import sys
    # Usage:
    # python broize.py "I am an engineer"

    # if len(sys.argv) > 0:
    #     saying = sys.argv[1]
    # else:
    #     saying = "How are you, brobot?"

    while True:
        x = input()
        zoro(x)
        if x.lower() == 'End':
            break


