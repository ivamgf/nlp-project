# imports
import os
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import re

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Define the list of stopwords for the Portuguese language
stop_words = set(stopwords.words('portuguese'))

# Create the Stemmer
stemmer = PorterStemmer()

# Path
path = "C:\Dataset"

# Get a list of files in the directory
files = os.listdir(path)

# Print the file list
print("Files found in the directory:")
for file in files:
    print(file)

    # Open the file and read the contents
    with open(os.path.join(path, file), 'r', encoding='utf8') as c:
        content = c.read()

        # Remove the text between the < and > symbols
        content = re.sub('<.*?>', ' ', content)

        # Tokenize the content in the sentences
        sentences = sent_tokenize(content)

        # List to store the filtered sentences
        filtered_sentences = []

        # Print sentences without stop words
        print("Sentences in the file " + file + " without stop words:")
        for sentence in sentences:
            # Remove the text between the < and > symbols
            sentence = re.sub('<.*?>', ' ', sentence)

            # Tokenize each sentence into words
            words = word_tokenize(sentence.lower())

            # Remove stop words
            filtered_words = [
                word
                for word in words
                if word not in stop_words and
                   word not in punctuation
            ]

            # Apply stemming to each word
            words_stemming = [
                stemmer.stem(word)
                for word in filtered_words
            ]

            # Put the filtered words together into a sentence again
            filtered_sentence = ' '.join(words_stemming)

            # Add the filtered sentence to the list
            filtered_sentences.append(filtered_sentence)

            # Tokenize the filtered sentences into words
            words = [word_tokenize(sentence) for sentence in filtered_sentences]

# Build vocabulary
model = Word2Vec(words, min_count=1, workers=2)

# Train a Word2Vec model using the filtered sentences
model.train(words, total_examples=model.corpus_count, epochs=model.epochs)

# Print the vocabulary
print("Vocabulary:")
print(list(model.wv.key_to_index.keys()))

# Get the word vector for a specific word
for word in model.wv.key_to_index.keys():
    print("Word vector for the word => " + word + ":")
    print(model.wv.get_vector(word))

# Cluster the word vectors using K-Means
X = model.wv[model.wv.key_to_index.keys()]
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

# Print the clusters
clusters = {}
for i, word in enumerate(model.wv.index_to_key):
    label = kmeans.labels_[i]
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(word)
print("Clusters:")
for label, words in clusters.items():
    print(f"Cluster {label}: {words}")
