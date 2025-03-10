import spacy

from sign_recording import  play_sign_animation

# Load the SpaCy model
nlp = spacy.load("en_core_web_md")

def convert_to_sign_language(text):
    """
    Converts English text into a simplified Sign Language order (VOS or OSV).
    """
    doc = nlp(text)
    sign_sentence = []
    
    question_words = {"WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO"}
    subject = None
    verb = None
    obj = []
    wh_word = None

    for token in doc:
        lemma = token.lemma_.upper()
        
        # Ignore auxiliary verbs, determiners, prepositions, and punctuation
        if token.pos_ in ["AUX", "DET", "ADP", "PUNCT"]:
            continue

        # Capture question words (e.g., WHERE, WHAT)
        if lemma in question_words:
            wh_word = lemma
            continue

        # Identify subject
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subject = lemma
            continue

        # Identify main verb
        if token.dep_ == "ROOT":
            verb = lemma
            continue

        # Identify objects
        if token.dep_ in ["dobj", "attr", "prep", "pobj"]:
            obj.append(lemma)
            continue

        obj.append(lemma)

    # Build the final sentence
    if verb:
        sign_sentence.insert(0, verb)  # Verb first
    if obj:
        sign_sentence.extend(obj)  # Object follows verb
    if subject:
        sign_sentence.append(subject)  # Subject at the end
    if wh_word:
        sign_sentence.append(wh_word)  # WH-Question at the end

    return " ".join(sign_sentence)

# 🔹 Test Cases
test_sentences = [
    # "She is going to the store.",         # Expected: GO STORE SHE
    # "I want to eat an apple.",            # Expected: WANT EAT APPLE I
    # "He is playing football.",            # Expected: PLAY FOOTBALL HE
    # "They are watching a movie.",         # Expected: WATCH MOVIE THEY
    # "We will meet tomorrow.",             # Expected: MEET TOMORROW WE
    # "Can he help me?",                    # Expected: HELP ME HE?
    # "Where is the nearest hospital?",     # Expected: WHERE HOSPITAL NEAREST?
    # "Hello, how are you?",                # Expected: HELLO YOU HOW?
    # "I am learning sign language.",       # Expected: LEARN SIGN LANGUAGE I
    # "She loves reading books."            # Expected: LOVE READ BOOK SHE
]
sentences = [
    # Basic WH- Questions
    "What is your name?",               # NAME YOUR WHAT
    "Where is the nearest hospital?",   # HOSPITAL NEAREST WHERE
    "Who is your best friend?",         # FRIEND BEST WHO
    "When is the meeting?",             # MEETING WHEN
    "Why are you sad?",                 # SAD WHY
    "How do you cook rice?",            # COOK RICE HOW

    # WH- Questions with Context
    "Where did you go yesterday?",      # GO YESTERDAY WHERE
    "When will the train arrive?",      # TRAIN ARRIVE WHEN
    "Who gave you this book?",          # BOOK GIVE WHO
    "Why is the baby crying?",          # BABY CRY WHY
    "How does this machine work?",      # MACHINE WORK HOW

    # Yes/No Questions
    "Do you like coffee?",              # LIKE COFFEE YOU
    "Can she drive a car?",             # DRIVE CAR SHE CAN
    "Is he your brother?",              # BROTHER YOUR HE
    "Are they coming tomorrow?",        # COME TOMORROW THEY
    "Did you finish your homework?",    # FINISH HOMEWORK YOU

    # Questions with Time Reference
    "What time is it now?",             # TIME NOW WHAT
    "Where will you go next week?",     # GO NEXT WEEK WHERE
    "When did you wake up today?",      # WAKE UP TODAY WHEN
    "Why was she absent yesterday?",    # ABSENT YESTERDAY WHY
    "How long does the journey take?",  # JOURNEY TAKE HOW LONG

    # Questions with Location Context
    "Where is your school?",            # SCHOOL YOUR WHERE
    "Where do you work?",               # WORK WHERE YOU
    "Where is my phone?",               # PHONE MY WHERE
    "Where is the police station?",     # POLICE STATION WHERE
    "Where can I find a taxi?",         # TAXI FIND WHERE

    # Questions About People & Relationships
    "Who is your teacher?",             # TEACHER YOUR WHO
    "Who helped you?",                  # HELP YOU WHO
    "Who won the match?",               # MATCH WIN WHO
    "Who is standing near the door?",   # DOOR NEAR STAND WHO
    "Who will come to the party?",      # COME PARTY WHO

    # Complex Questions
    "What do you want to eat?",         # WANT EAT WHAT
    "How do you solve this problem?",   # SOLVE PROBLEM HOW
    "Why do we need to study?",         # STUDY NEED WHY
    "Where should I put this bag?",     # BAG PUT WHERE
    "When will you complete the project?" # COMPLETE PROJECT WHEN
]

sentence="hi hello you good"
# # Run Tests
# for sentence in sentences:
#     print(f"Input: {sentence}")
#     print(f"Sign Language: {convert_to_sign_language(sentence)}\n")

print(f"Sign Language: {convert_to_sign_language(sentence)}\n")

word_list=convert_to_sign_language(sentence).split(" ")
# word_list=['HI', 'HELLO', 'YOU', 'GOOD']
print(word_list)

for word in word_list:
    play_sign_animation(word.lower())

for word in word_list:
    play_sign_animation(word.lower())