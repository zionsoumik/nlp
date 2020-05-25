from os import listdir
import os
import csv
import re
from os.path import isfile, join
input=[]
mypath=join(os.getcwd(),'train')
mypath=join(mypath,'pos')
print(mypath)
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

for f in os.listdir(mypath):
    dict={}
    file=open(mypath+"\\"+f,'r',encoding='utf-8')
    #print(file.read())
    dict["id"]=f.split("_")[0]
    dict["rating"]=1
    text=REPLACE_NO_SPACE.sub("", file.read().lower())
    text=REPLACE_WITH_SPACE.sub("", text.lower())
    dict["text"] = text
    #print(dict)
    input.append(dict)
mypath=join(os.getcwd(),'train')
mypath=join(mypath,'neg')
print(mypath)
for f in os.listdir(mypath):
    dict={}
    file=open(mypath+"\\"+f,'r',encoding='utf-8')
    #print(file.read())
    dict["id"]=f.split("_")[0]
    dict["rating"]=0
    text = REPLACE_NO_SPACE.sub("", file.read().lower())
    text = REPLACE_WITH_SPACE.sub("", text.lower())
    dict["text"] = text
    #print(dict)
    input.append(dict)
with open('train.tsv', 'w',encoding='utf-8') as csvfile:
    fieldnames = ['id', 'text','rating']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')

    writer.writeheader()
    for i in input:
        writer.writerow(i)
#print(input)