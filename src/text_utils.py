# Let's split the reviews into sentences so we can get a fine-grained view of the reviews

import en_core_web_sm

# Load the small English model. You can use the medium or large model for more accuracy but at the cost of speed.
nlp = en_core_web_sm.load()


def split_into_sentences(text, ngram=1):
    # Use the model to split the text into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Handle the case where ngram is 1 (default behavior)
    if ngram == 1:
        return sentences

    # Group sentences into n-grams for ngram > 1
    ngrams = [
        " ".join(sentences[i : i + ngram]) for i in range(len(sentences) - ngram + 1)
    ]
    return ngrams
