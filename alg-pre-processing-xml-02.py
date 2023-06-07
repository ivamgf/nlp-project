# Algorithm 4 - dataset cleaning and pre-processing XML

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
        print("Annotated file content " + file + ":")

        # Get all elements in the XML
        all_elements = list(root.iter())

        # Find the indices of the target elements
        target_indices = [
            i for i, element in enumerate(all_elements)
            if element.tag == "webanno.custom.Judgmentsentity"
        ]

        for index in target_indices:
            element = all_elements[index]
            if (
                    "sofa" in element.attrib and
                    "begin" in element.attrib and
                    "end" in element.attrib and
                    "Instance" in element.attrib and
                    "Value" in element.attrib
            ):
                # Get the previous and next elements
                previous_index = index - 1 if index > 0 else None
                next_index = index + 1 if index < len(all_elements) - 1 else None

                previous_element = all_elements[previous_index] if previous_index is not None else None
                next_element = all_elements[next_index] if next_index is not None else None

                # Print the previous element if it exists
                if previous_element is not None:
                    print("Previous Tag:")
                    print("<" + previous_element.tag + ">")
                    print("Previous Content:")
                    print(previous_element.text.strip())

                # Print the current element's content
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

                # Print the next element if it exists
                if next_element is not None:
                    print("Next Tag:")
                    print("<" + next_element.tag + ">")
                    print("Next Content:")
                    print(next_element.text.strip())
                    print("---------------------------------------------------")