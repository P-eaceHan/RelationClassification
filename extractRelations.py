'''
This is a program to pre-process relation extraction data

feature vector:
[
shortest dependency path - raw
  shortest dependency path - POS
  distance to main verb/root
]

@author Peace Han
'''
import xml.etree.ElementTree as ET

path = 'clean/train_data/'
file = path + '1.1.text.xml'

# with open(file) as f:
#     for line in f:
#         print(line)

tree = ET.parse(file)
root = tree.getroot()
print(root)
print(root.tag)

textCount = 0
entCount = 0
for child in root:  # iterate over every text element
    # print(child.tag, child.attrib)
    print("text id=", child.tag, child.attrib)
    textCount += 1
    # print(child[0])  # this is the title element
    # print(child[1])  # this is the abstract text element
    abs = child[1]
    print('\t', abs.text)  # this only gets the bit right before the first entitiy
    # TODO: figure out above issue (see asgmt1)
    for ent in abs:
        print('\t', ent.attrib)
        print('\t', ent.text)
        entCount += 1
        # print()

print("total texts: ", textCount)
print("total entities: ", entCount)

