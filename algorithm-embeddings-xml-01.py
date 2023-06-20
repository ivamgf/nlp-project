# Algorithm 2 - dataset cleaning, pre-processing XML and create embeddings
# Results in file and browser

# Imports
import os
import nltk
import xml.etree.ElementTree as ET
import spacy
from gensim.models import Word2Vec
from datetime import datetime
import hashlib
import webbrowser

nltk.download('punkt')

path = "C:\\Dataset-TRT"
files = os.listdir(path)

nlp = spacy.load("pt_core_news_sm")

def replace_words(text):
    word_replacements = {
        "LimitaÃ§Ã£o da condenaÃ§Ã£o aos valores dos pedidos": "Limitacao da condenacao aos valores dos pedidos",
        "AssÃ©dio moral": "Assedio moral",
        "HonorÃ¡rios sucumbenciais": "Honorarios sucumbenciais",
        "Estabilidade acidentÃ¡ria": "Estabilidade acidentaria",
        "DoenÃ§a ocupacional": "Doenca ocupacional"
    }
    for old_word, new_word in word_replacements.items():
        text = text.replace(old_word, new_word)
    return text

def replace_expression(text):
    expressions = {
        "LimitaÃ§Ã£o da condenaÃ§Ã£o aos valores dos pedidos": "Limitacao da condenacao aos valores dos pedidos",
        "AssÃ©dio moral": "Assedio moral",
        "HonorÃ¡rios sucumbenciais": "Honorarios sucumbenciais",
        "Estabilidade acidentÃ¡ria": "Estabilidade acidentaria",
        "DoenÃ§a ocupacional": "Doenca ocupacional"
    }
    for expression, replacement in expressions.items():
        text = text.replace(expression, replacement)
    return text

output_dir = "C:\\Outputs"
current_datetime = datetime.now()
timestamp = current_datetime.strftime("%Y%m%d%H%M%S")
hash_value = hashlib.md5(current_datetime.isoformat().encode()).hexdigest()

output_file_txt = os.path.join(output_dir, f"output_{timestamp}_{hash_value}.txt")
output_file_html = os.path.join(output_dir, f"output_{timestamp}_{hash_value}.html")

os.makedirs(output_dir, exist_ok=True)

output_html = ""

output_html += "<h3>Arquivos encontrados no diretório:</h3>"
for file in files:
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"
        sentences = []
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

                instance = replace_words(instance)
                value = replace_words(value)

                for child in element:
                    if child.text:
                        text = child.text.strip().lower()
                        text = replace_words(text)
                        text = replace_expression(text)
                        doc = nlp(text)
                        tokens = [token.text for token in doc]
                        sentences.append(tokens)

        if len(sentences) > 0:
            model = Word2Vec(min_count=1, workers=2)
            model.build_vocab(sentences)
            model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

            output_html += "<h4>Vocabulary:</h4>"
            for word in model.wv.key_to_index.keys():
                output_html += f"<p>Token: {word}</p>"
                output_html += f"<p>Instance: {instance}</p>"
                output_html += f"<p>Value: {value}</p>"
                output_html += "<p>Embedding:</p>"
                output_html += f"<pre>{model.wv[word]}</pre>"
        else:
            output_html += "<p>Nenhuma sentença encontrada para treinar o modelo Word2Vec.</p>"

# Salva o resultado no arquivo de saída TXT
with open(output_file_txt, "w", encoding="utf-8") as f:
    f.write(output_html)

# Salva o resultado no arquivo de saída HTML
with open(output_file_html, "w", encoding="utf-8") as f:
    f.write(output_html)

# Abre o arquivo HTML no navegador
webbrowser.open(output_file_html)

print("Results saved in folder C://Outputs")
