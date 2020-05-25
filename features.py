import csv
import string
data=csv.DictReader(open("train.tsv",encoding="utf-8"),delimiter='\t')
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet
from nltk.wsd import lesk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
import re
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
ftext=[]
punct=set(["?",".","!"])
data=list(data)
for row in data:
    text=row["text"].lower()
    sents=text.translate(text.maketrans("","", string.punctuation))
    pos_score=0
    neg_score=0
    obj_score=0

    #
    texts=sent_tokenize(sents)
    for i in range(0, len(texts)):

        index_b = texts[i].find(" but ")
        if (index_b != -1):
            texts[i] = texts[i][index_b + 5:len(texts[i])]
        index_although = texts[i].find("Although ")
        if (index_although != -1):
            for k in range(index_although, len(texts[i])):
                if texts[i][k] == "." or texts[i][k] == ";" or texts[i][k] == "!" or texts[i][k] == ",":
                    break
            index_p = k
            #print(index_p)
            texts[i] = texts[i][index_p + 1:len(texts[i])]

        #print(texts[i])
    for sent in texts:
        words=word_tokenize(text)
        pos = pos_tag(words)
        newwords=""
        m_pos=1
        m_neg=1
        m_obj=1

        #newword=[]
        for i in range(0,len(words)):
            lala = 0
            stop = set(stopwords.words('english'))
            if words[i] not in stop:
                #newword=wordnet_lemmatizer.lemmatize(word)
                #print(newword)
                #print(text)
                sense=lesk(text,words[i])    ##word sense disambiguation

                #print(sense)
                if sense is not None:
                    senti=sentiwordnet.senti_synset(str(sense)[8:-2])
                    #print("senti:",senti)
                    if senti is not None:

                        newwords=newwords+" "+ str(sense)[8:-2]
                        #print(words[i-1],pos[i-1])
                        if wn.synsets(words[i-1])==[]:
                            lala=1
                        #print(wn.synsets(words[i - 1]))
                        #print(lala)
                        if (pos[i-1][1][:2] =='JJ' or pos[i-1][1][:2] =='RB') and lala==0:   #trying out the pos tagger of NLTK as it is very efficient

                                m_pos = sentiwordnet.senti_synset(str(lesk(text,words[i-1]))[8:-2]).pos_score()
                                #m_obj = sentiwordnet.senti_synset(str(lesk(text,words[i-1]))[8:-2]).pos_score()
                                m_neg = sentiwordnet.senti_synset(str(lesk(text,words[i-1]))[8:-2]).neg_score()
                                if words[i-1] == "not":
                                    m_pos = -1
                                    m_neg=-1
                                    m_obj=-1
                                else:
                                    if m_pos>0 or m_neg>0:
                                        #print(m_neg)
                                        if m_pos>m_neg:
                                            m_pos=1/m_pos  #scaling up
                                            #m_neg=1
                                        else:
                                            m_neg=1/m_neg   #scaling up
                                            #m_pos=1

                                #print(m_pos,m_neg,m_obj)

                        #else:
                        #print(m_obj)
                        obj_score_temp=senti.obj_score()
                        pos_score_temp=senti.pos_score()
                        neg_score_temp=senti.neg_score()
                        if words[i - 1] == "occasionally" or words[i - 1] == "less" or words[i - 1] == "only" or \
                                        words[i - 1] == "just" or words[i - 1] == "barely":
                            m_pos = 0.5
                            m_neg = 0.5
                            if words[i - 2] == "not":
                                #f = m_pos

                                f=pos_score_temp
                                pos_score_temp=neg_score_temp
                                neg_score_temp=pos_score_temp
                        if words[i - 1] == "very" or words[i - 1] == "so" or words[i - 1] == "too":
                            m_pos = 1.5
                            m_neg= 1.5
                            # print(score)
                            # print(c)
                            if words[i - 2] == "not":
                                f = pos_score_temp
                                pos_score_temp = neg_score_temp
                                neg_score_temp = pos_score_temp
                        # transforming the scores with prior weights

                        obj_score = obj_score + obj_score_temp
                        pos_score = pos_score + m_pos * pos_score_temp
                        neg_score = neg_score + m_neg * neg_score_temp
                        m_pos = 1
                        m_neg = 1
                        m_obj = 1

                #newword.append(word)
        #print(len(words))
        #print("obj_score:", obj_score)
        #print("pos_score:", pos_score)
        #print("neg_score:", neg_score)
        #print(words[i], sense)
        row["synset"]=newwords
        if len(words)!=0 and (pos_score+neg_score+obj_score/len(words))!=0:
            row["pscore"]=pos_score/(pos_score+neg_score+obj_score/len(words))
            row["nscore"]=neg_score/(pos_score+neg_score+obj_score/len(words))
            row["obscore"]=(obj_score/len(words))/(pos_score+neg_score+obj_score/len(words))
        else:
            row["pscore"] = 0
            row["nscore"] = 0
            row["obscore"] = 0
with open('newsenti1.csv', 'w', encoding="utf-8", newline="") as csvfile:
    fieldnames=data[1].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for d in data:
        writer.writerow(d)
