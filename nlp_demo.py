import random
import spacy
from spacy.util import minibatch
from spacy.training.example import Example
from spacy_layout import spaCyLayout

# texts = [
#     'What is the price of 4 Bananas?',
#     '20 Apples cost how much?',
#     'How much does a Phone cost in Rupees?'
# ]

nlp = spacy.load('en_core_web_md')

layout = spaCyLayout(nlp)

doc = layout('./invc_1[1].pdf')

print(doc._.tables)

# ner_labels = nlp.get_pipe('ner').labels
# print(ner_labels)

# docs = [nlp(text) for text in texts]

# for doc in docs:
#     for ent in doc.ents:
#         print(f"{ent.text} | {ent.label_}")






