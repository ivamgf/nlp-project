# Algorithm 1 - BILSTM-CE Model training
# Dataset cleaning, pre-processing XML and create slots and embeddings
# RNN Bidiretional LSTM Layer with word and char embeddings concatened

# Results in file and browser

# Imports
import os
import xml.etree.ElementTree as ET
import webbrowser
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from keras.layers import Dense

# Downloads
nltk.download('punkt')

# Directory path
path = "C:\\Dataset-TRT"
files = os.listdir(path)

output_dir = "C:\\Outputs"
os.makedirs(output_dir, exist_ok=True)

output_html = ""

output_html += "<h3>Arquivos encontrados no diretório:</h3>"

slot_number = 1

# Functions

# Char Embedding Function
def char_embedding(text):
    embedding = np.zeros((20, 1))
    for i, char in enumerate(text):
        if i >= 20:
            break
        embedding[i] = ord(char)
    return embedding

# Tokenize the sentences into words and create skipgram Word2Vec
def tokenize_sentence(sentence):
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens if token.lower() not in string.punctuation]

    # Create skipgram Word2Vec model for the sentence
    model = Word2Vec(sentences=[tokens], min_count=1, workers=2, sg=1, window=5)

    return model

# Loop through files in directory
for file in files:
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"
        sentences = []

        # Loop through sentences
        for sentence in root.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'):
            tokens = []
            for token in sentence.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'):
                tokens.append(token.text.strip())

            # Checks if the sentence contains the specific tags
            if sentence.find(".//webanno.custom.Judgmentsentity") is not None:
                annotated_word = sentence.find(
                    ".//webanno.custom.Judgmentsentity/de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token").text.strip()

                sentence_text = ' '.join(tokens)

                # Checks if the sentence has already been added
                if sentence_text not in sentences:
                    sentences.append(sentence_text)

                    # Tokenize the sentences
                    sentences_list = sent_tokenize(sentence_text)

                    # Prints the sentences and the annotated word
                    for sent_idx, sent in enumerate(sentences_list[:5]):  # Select up to 5 sentences
                        tokenized_sent = tokenize_sentence(sent)
                        annotated_index = tokenized_sent.wv.key_to_index.get(
                            annotated_word.lower(), -1)
                        context_start = max(0, annotated_index - 5)
                        context_end = min(annotated_index + 6, len(tokenized_sent.wv.key_to_index))
                        context_words = list(tokenized_sent.wv.key_to_index.keys())[context_start:context_end]
                        context_words.reverse()  # Reverse the word order

                        # Print the Instance and Value attributes
                        for element in root.iter("webanno.custom.Judgmentsentity"):
                            if (
                                    "sofa" in element.attrib and
                                    "begin" in element.attrib and
                                    "end" in element.attrib and
                                    "Instance" in element.attrib and
                                    "Value" in element.attrib
                            ):
                                sofa = element.attrib["sofa"]
                                begin = element.attrib["begin"]
                                end = element.attrib["end"]
                                instance = element.attrib["Instance"]
                                value = element.attrib["Value"]

                        context_text = ' '.join(context_words)
                        context_text = context_text.replace(annotated_word, f"[annotation]{annotated_word}[annotation]")
                        output_html += f"<p>Sentença {slot_number}: {context_text}</p>"
                        output_html += f"<p>Annotated Word: {annotated_word}</p>"
                        output_html += f"<p>Instance: {instance}</p>"
                        output_html += f"<p>Value: {value}</p>"

                        # Print the token vector
                        output_html += f"<p>Slot de Tokens {slot_number}: {context_words}</p>"

                        # Word Embeddings
                        output_html += f"<p>Word Embeddings {slot_number}: </p>"
                        output_html += "<pre>"
                        for word in context_words:
                            word_embedding = tokenized_sent.wv[word].reshape((100, 1))
                            output_html += f"<p>{word}: {word_embedding}</p>"
                        output_html += "</pre>"

                        # Char Embeddings
                        output_html += f"<p>Char Embeddings {slot_number}: </p>"
                        output_html += "<pre>"
                        for word in context_words:
                            output_html += f"<p>{word}: {char_embedding(word)}</p>"
                        output_html += "</pre>"

                        # Word and Char Embedding concatenated
                        output_html += "<p>Word and Char Embedding concatenated:</p>"
                        output_html += "<pre>"
                        for word in context_words:
                            word_embedding = tokenized_sent.wv[word].reshape((100, 1))
                            char_emb = char_embedding(word)
                            concatenated_emb = np.concatenate((word_embedding, char_emb))
                            output_html += f"<p>{word}: {concatenated_emb}</p>"
                        output_html += "</pre>"

                        # Bidirectional LSTM model
                        input_size = concatenated_emb.shape[-1]
                        hidden_size = 64
                        num_classes = 10
                        sequence_length = 1

                        # Transpose input
                        concatenated_emb = np.transpose(concatenated_emb, (1, 0))

                        # Generate example data
                        num_samples = 1
                        # Reshape the input data
                        X = concatenated_emb.reshape((num_samples, 1, 120))
                        y = tf.random.uniform((num_samples, num_classes))

                        # Create Bidirectional LSTM model
                        lstm_model = tf.keras.Sequential()
                        lstm_model.add(Dense(units=32))
                        lstm_model.add(
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                                hidden_size, input_shape=(1, 120), dropout=0.1)))
                        lstm_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

                        # Learning rate
                        learning_rate = 0.01
                        rho = 0.9

                        # Optimizer
                        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho)

                        # Compile o modelo
                        lstm_model.compile(
                            loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                        # Definir paciência (patience) e EarlyStopping
                        patience = 10
                        early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

                        # Treinar o modelo com EarlyStopping
                        lstm_model.fit(X, y, epochs=60, batch_size=32, callbacks=[early_stopping])

                        # Print LSTM model results
                        output_html += "<p>Bidirectional LSTM Model Results:</p>"
                        lstm_results = lstm_model.predict(X)
                        output_html += "<pre>"
                        # Get the indices of the words in the Slot de Tokens
                        word_indices = [tokenized_sent.wv.key_to_index[word.lower()] for word in context_words]
                        # Create a dictionary mapping word indices to LSTM results
                        results_dict = dict(zip(word_indices, lstm_results[0]))
                        # Iterate over the words in the Slot de Tokens and print the corresponding LSTM result
                        for word in context_words:
                            word_index = tokenized_sent.wv.key_to_index[word.lower()]
                            result = results_dict.get(word_index, 0.0)  # Default to 0.0 if word index not found
                            if word == annotated_word:
                                result = results_dict.get(word_index, 0.0)
                            output_html += f"<p>{word}: {result}"
                            if word != annotated_word:
                                output_html += " - [O]"
                            if word == annotated_word:
                                output_html += " - [B-ReqTreatment]"
                            output_html += "</p>"

                        output_html += "</pre>"
                        slot_number += 1

# Output files path
output_file_txt = os.path.join(output_dir, "output.txt")
output_file_html = os.path.join(output_dir, "output.html")

# Save the result to the output TXT file
with open(output_file_txt, "w", encoding="utf-8") as f:
    f.write(output_html)

# Save the result to the HTML output file
with open(output_file_html, "w", encoding="utf-8") as f:
    f.write(output_html)

# Opens the HTML file in the browser
webbrowser.open(output_file_html)

print("Results saved in folder C://Outputs")
