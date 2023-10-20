# Let's split the reviews into sentences so we can get a fine-grained view of the reviews

import en_core_web_sm

# Load the small English model. You can use the medium or large model for more accuracy but at the cost of speed.
nlp = en_core_web_sm.load()

def split_into_sentences(text):
    # Use the model to split the text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences