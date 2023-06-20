# Algorithm 3 - dataset cleaning and pre-processing XML

# Imports
import os
import nltk
import xml.etree.ElementTree as ET

# Downloads
nltk.download('punkt')

# Use double backslashes in file or directory path
path = "C:\\Dataset-TRT"

# Get a list of files in the directory
files = os.listdir(path)

# Print the file list
print("Files found in the directory:")
for file in files:
    if file.endswith(".xml"):
        print(file)

        # Open the XML file and parse the contents
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the XML tags and text content within the specific tags
        print("Content of the file " + file + ":")
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

                print(
                    f"<webanno.custom.Judgmentsentity "
                    f"sofa='{sofa}' "
                    f"begin='{begin}' "
                    f"end='{end}' "
                    f"Instance='{instance}' "
                    f"Value='{value}'>"
                )
                # Iterate over the element's children and print their text content
                for child in element:
                    if child.text:
                        print(child.text.strip())

                print("</webanno.custom.Judgmentsentity>")