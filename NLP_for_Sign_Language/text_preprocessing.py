import spacy
print("before loading")
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_md")



text = "Where is the nearest hospital?"
doc = nlp(text)
print("-------------------------------------")
for token in doc:
    print(token.text, token.pos_, token.dep_)

print("-------------------------------------")