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
import regex as re
import xml.etree.ElementTree as ET

path = 'clean/train_data/'
file = path + '1.1.text.xml'
outfile = path + '/1.1.text_abs.txt'
entfile = path + '/1.1.text_ents.csv'
o = open(outfile, 'w')
p = open(entfile, 'w')

tree = ET.parse(file)
root = tree.getroot()
print(root)
print(root.tag)

textCount = 0
entCount = 0
for text in root.findall('text'):
    textCount += 1
    # print(text.find('abstract'))
    abstract = text.find('abstract')
    abstractString = ET.tostring(text.find('abstract'))
    abstractString = abstractString.decode('UTF-8').strip()
    abstractString = ' '.join(abstractString.split('\n'))
    abstractString = ' '.join(re.split('  +', abstractString))
    # [x.strip() for x in abstractString]
    print("Original abstract: \n\t", abstractString)

    pattern = "<entity.*?<\/entity>"
    abstractList = re.split(pattern, abstractString)
    print(abstractList)

    print("entities:")
    res = re.findall(pattern, abstractString)
    # [print(x) for x in res]
    pattern2 = '<entity id="(.*?)">'
    pattern3 = '<entity.*?>(.*?)</entity>'
    for ent in res:
        id = re.match(pattern2, ent).group(1)
        t = re.match(pattern3, ent).group(1)
        # print(id, t)
        p.write(id + ',')
        p.write(t + '\n')
    entCount += len(res)

    scrubbed = []
    for i in range(len(abstractList)):
        # print(abstractList[i])
        scrubbed.append(abstractList[i].strip())
        if i < len(res):
            # print(re.match(pattern2, res[i]).group(1))
            id = re.match(pattern2, res[i]).group(1)
            scrubbed.append(id)
    scrubbed = ' '.join(scrubbed)
    scrubbed = re.match('<abstract>(.*)</abstract>', scrubbed).group(1)
    scrubbed = scrubbed.strip()
    print(scrubbed)
    o.write(scrubbed + '\n')

o.close()

print("total texts: ", textCount)
print("total entities: ", entCount)

