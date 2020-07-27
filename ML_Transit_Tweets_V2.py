#training data
#get alot of tweets! ~200k
#get top words + top bigrams 
#20 top transit + 20 top transit bigrams + 20 top nontransit + 20 top nontransit bigrams + confusing terms: train of thought etc (maybe 10)
#length of tweet=<1/3 = 0 1/3 to 2/3 = 1 >2/3 =2
#take top 220,000 tweets for ttc from moein
#remove duplicates: 165,558 remain
#remove pregnancy stuff 164,455
#take top 55,000 tweets from translink from moein
#remove duplicates 45,135 remain
#take top 21,000 tweets from gotransit from moein
#remove duplicates 16,470 remain
#put these 226060 tweets here: C:\Users\Omar\OneDrive\Python_sync\ML_Transit_Tweets_V2\Training_tweets\training_pos.txt

#----------------

#ttc processing
#only keep english tweets
#remove antalya
#remove resort
#remove euro
#remove partir

import os
import csv
import nltk
import re
import numpy as np
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize
from functools import reduce
from collections import Counter
import operator
#nltk.download('stopwords')
stemmer=LancasterStemmer()
stop_words = stopwords.words('english')

#function to remove URLs
def remove_urls (vTEXT):
    vTEXT =re.sub(r'http\S+', '', vTEXT)
    return(vTEXT)

def bagofwords(sentence, words): #function to create bag of words
    sentence_words = sentence
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1                
    return np.array(bag)

corpus=[]
tok_corp2=[]
temparray=[]
loading=0

inputpreprocessingtransit=input("Do you want to preprocess transit tweets?, Y for yes | N for no: ")
inputpreprocessingnontransit=input("Do you want to preprocess transit tweets?, Y for yes | N for no: ")
if inputpreprocessingtransit=="Y":
    file = open(r"C:\Users\Omar\OneDrive\Python_sync\ML_Transit_Tweets_V2\Training_tweets\training_pos.txt",'r',encoding="utf-8")
    lines=file.readlines()
    file.close()
    f= open("ProcessedTransitTweets.txt","w")
    for line in lines:
        print("Preprocessing:",loading)
        loading=loading+1
        #line=line[2:]
        line=line.replace("RT","") #remove the word RT
        line=remove_urls(line) #remove URLs
        line=' '.join(word for word in line.split(' ') if not word.startswith('@')) #remove mentions
        line=line.lower() #make lowercase
        line=re.sub(r'[^\w]', ' ', line) #remove symbols
        text_tokens = word_tokenize(line)
        #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()] #remove stopwords
        #tokens_without_sw = [word for word in tokens_without_sw if word.startswith('u04')==False] #remove u04
        #instead
        tokens_without_sw = [word for word in text_tokens if word.startswith('u04')==False] #remove u04
        filtered_sentence = (" ").join(tokens_without_sw)
        corpus.append(filtered_sentence) #put the processed tweets in corpus 
        f.write(filtered_sentence)
        f.write("\n")
    f.close() #we have produced a txt file of the processed tweets
elif inputpreprocessingtransit=="N":
    file = open("ProcessedTransitTweets.txt",'r',encoding="utf-8")
    lines=file.readlines()
    file.close()
    for line in lines:
        corpus.append(line)

print(corpus)

tok_corp=[nltk.word_tokenize(sent) for sent in corpus]

for sentence in tok_corp:
    for word in sentence:
       temparray.append(word)
    tok_corp2.append(temparray) #tokenize each sentence in corpus
    temparray=[]

print(tok_corp2) #basically corpus, but tokenized 

flatcorpus=reduce(operator.concat, tok_corp2) #put all the words from tok_corp2 into 1 list (later used to get word counts)
#print(flatcorpus)

counter_obj = Counter(flatcorpus) #count the unique words

bagofwords_list=[]

numberoftopwords=40 #this is how many top words do we need to identify

for i in np.arange(numberoftopwords): #
    bagofwords_list.append(counter_obj.most_common()[i][0]) #create a list of the most frequent words

print(bagofwords_list)
print("------------")
#print(counter_obj.most_common()) #this gives a count of the unique words

f= open("MostCommonTransitWords.txt","w")
for i in np.arange(len(bagofwords_list)):
     f.write(str(bagofwords_list[i]))
     f.write("\n")
f.close() #we have produced a txt file of the most common words
'''
templist=[]
newcorpus=[]
temparray2=np.zeros((numberoftopwords,)) 
counter=0

for i in np.arange(len(tok_corp2)):
    templist=list(bagofwords(tok_corp2[i],bagofwords_list))
    templist=list(templist)
    newcorpus.append(templist)

#print(newcorpus) #this is a list of all the vectorized tweets

exportcorpus=newcorpus # i don't know why i did this

f= open("Corpus.txt","w") #this step outputs a txt file only so that we can easily remove the [ and ] from the list and output just numbers into a txt file
for i in np.arange(len(tok_corp2)): #this file will be deleted later, will be replaced by outputtransit.txt
     f.write(str(exportcorpus[i]))
     f.write("\n")
f.close()

with open("Corpus.txt",encoding="utf-8") as infile, open("OutputTransit.txt", 'w',encoding="utf-8") as outfile:
    for line in infile:
        line2=line.replace("[", "") #remove [ from start of list 
        line3=line2.replace("]", "") #remove ] from end of list
        outfile.write(line3)  
outfile.close()
infile.close()

os.remove("Corpus.txt") #deleted, we don't need it
'''
#now we need to create vectors for bigrams
bigrams=[]
tempbigrams=[]
for tweet in corpus:
    _bigrams=ngrams(tweet.split(),2)
    __bigrams=list(_bigrams)
    for bigram in __bigrams:
        bigram_=bigram[0]+" "+bigram[1]
        tempbigrams.append(bigram_)
    bigrams.append(tempbigrams)
    tempbigrams=[]

print(bigrams) #this is a list of the tweets (tokenized into bigrams)

flatcorpus=reduce(operator.concat, bigrams) #put all the bigrams from bigrams into 1 list (later used to get bigram counts)
#print(flatcorpus)

counter_obj = Counter(flatcorpus) #count the unique bigrams

bagofbigrams_list=[]

numberoftopbigrams=40 #this is how many top bigrams we need to identify

for i in np.arange(numberoftopbigrams): 
    bagofbigrams_list.append(counter_obj.most_common()[i][0]) #create a list of the most frequent bigrams

print(bagofbigrams_list)
print("------------")
#print(counter_obj.most_common()) #this gives a count of the unique bigrams

f= open("MostCommonTransitBigrams.txt","w")
for i in np.arange(len(bagofbigrams_list)):
     f.write(str(bagofbigrams_list[i]))
     f.write("\n")
f.close() #we have produced a txt file of the most common bigrams
'''
templist=[]
newcorpus=[]
temparray2=np.zeros((numberoftopbigrams,)) 
counter=0

for i in np.arange(len(bigrams)):
    templist=list(bagofwords(bigrams[i],bagofbigrams_list))
    templist=list(templist)
    newcorpus.append(templist)

print(newcorpus) #this is a list of all the vectorized tweets

exportcorpus=newcorpus # i don't know why i did this

f= open("BigramsCorpus.txt","w") #this step outputs a txt file only so that we can easily remove the [ and ] from the list and output just numbers into a txt file
for i in np.arange(len(bigrams)): #this file will be deleted later, will be replaced by outputtransit.txt
     f.write(str(exportcorpus[i]))
     f.write("\n")
f.close()

with open("BigramsCorpus.txt",encoding="utf-8") as infile, open("OutputTransitBigrams.txt", 'w',encoding="utf-8") as outfile:
    for line in infile:
        line2=line.replace("[", "") #remove [ from start of list 
        line3=line2.replace("]", "") #remove ] from end of list
        outfile.write(line3)  
outfile.close()
infile.close()

os.remove("BigramsCorpus.txt") #deleted, we don't need it
'''
#now we need to vectorize the nontransit corpus
file = open(r"C:\Users\Omar\OneDrive\Python_sync\ML_Transit_Tweets_V2\Training_tweets\training_neg.txt",'r',encoding="utf-8")
lines=file.readlines()
file.close()

corpus=[]
tok_corp2=[]
temparray=[]
loading=0
if inputpreprocessingnontransit=="Y":
    file = open(r"C:\Users\Omar\OneDrive\Python_sync\ML_Transit_Tweets_V2\Training_tweets\training_neg.txt",'r',encoding="utf-8")
    lines=file.readlines()
    file.close()
    f= open("ProcessedNonTransitTweets.txt","w")
    for line in lines:
        print("Preprocessing:",loading)
        loading=loading+1
        line=line[2:]
        line=line.replace("RT","") #remove the word RT
        line=remove_urls(line) #remove URLs
        line=' '.join(word for word in line.split(' ') if not word.startswith('@')) #remove mentions
        line=line.lower() #make lowercase
        line=re.sub(r'[^\w]', ' ', line) #remove symbols
        text_tokens = word_tokenize(line)
        #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()] #remove stopwords
        #tokens_without_sw = [word for word in tokens_without_sw if word.startswith('u04')==False] #remove u04
        #instead
        tokens_without_sw = [word for word in text_tokens if word.startswith('u04')==False] #remove u04
        filtered_sentence = (" ").join(tokens_without_sw)
        corpus.append(filtered_sentence) #put the processed tweets in corpus 
        f.write(filtered_sentence)
        f.write("\n")
    f.close() #we have produced a txt file of the processed tweets
elif inputpreprocessingnontransit=="N":
    file = open("ProcessedNonTransitTweets.txt",'r')
    lines=file.readlines()
    file.close()
    for line in lines:
        corpus.append(line)
print(corpus)

tok_corp=[nltk.word_tokenize(sent) for sent in corpus]

for sentence in tok_corp:
    for word in sentence:
       temparray.append(word)
    tok_corp2.append(temparray) #tokenize each sentence in corpus
    temparray=[]

print(tok_corp2) #basically corpus, but tokenized 

flatcorpus=reduce(operator.concat, tok_corp2) #put all the words from tok_corp2 into 1 list (later used to get word counts)
#print(flatcorpus)

counter_obj = Counter(flatcorpus) #count the unique words

#print(counter_obj.most_common()) #this gives a count of the unique words

nontransitbagofwords_list=[]

for i in np.arange(numberoftopwords): #
    nontransitbagofwords_list.append(counter_obj.most_common()[i][0]) #create a list of the most frequent words

f= open("MostCommonNonTransitWords.txt","w")
for i in np.arange(len(nontransitbagofwords_list)):
     f.write(str(nontransitbagofwords_list[i]))
     f.write("\n")
f.close() #we have produced a txt file of the most common words
'''
templist=[]
newcorpus=[]
temparray2=np.zeros((numberoftopwords,)) 
counter=0

for i in np.arange(len(tok_corp2)):
    templist=list(bagofwords(tok_corp2[i],bagofwords_list))
    templist=list(templist)
    newcorpus.append(templist)

#print(newcorpus) #this is a list of all the vectorized tweets

exportcorpus=newcorpus # i don't know why i did this

f= open("Corpus.txt","w") #this step outputs a txt file only so that we can easily remove the [ and ] from the list and output just numbers into a txt file
for i in np.arange(len(tok_corp2)): #this file will be deleted later, will be replaced by outputtransit.txt
     f.write(str(exportcorpus[i]))
     f.write("\n")
f.close()

with open("Corpus.txt",encoding="utf-8") as infile, open("OutputTransit.txt", 'w',encoding="utf-8") as outfile:
    for line in infile:
        line2=line.replace("[", "") #remove [ from start of list 
        line3=line2.replace("]", "") #remove ] from end of list
        outfile.write(line3)  
outfile.close()
infile.close()

os.remove("Corpus.txt") #deleted, we don't need it
'''
#now we need to create vectors for bigrams
bigrams=[]
tempbigrams=[]
for tweet in corpus:
    _bigrams=ngrams(tweet.split(),2)
    __bigrams=list(_bigrams)
    for bigram in __bigrams:
        bigram_=bigram[0]+" "+bigram[1]
        tempbigrams.append(bigram_)
    bigrams.append(tempbigrams)
    tempbigrams=[]

print(bigrams) #this is a list of the tweets (tokenized into bigrams)

flatcorpus=reduce(operator.concat, bigrams) #put all the bigrams from bigrams into 1 list (later used to get bigram counts)
#print(flatcorpus)

counter_obj = Counter(flatcorpus) #count the unique bigrams

#print(counter_obj.most_common()) #this gives a count of the unique bigrams

nontransitbagofbigrams_list=[]

for i in np.arange(numberoftopbigrams): 
    nontransitbagofbigrams_list.append(counter_obj.most_common()[i][0]) #create a list of the most frequent bigrams

f= open("MostCommonNonTransitBigrams.txt","w")
for i in np.arange(len(nontransitbagofbigrams_list)):
     f.write(str(nontransitbagofbigrams_list[i]))
     f.write("\n")
f.close() #we have produced a txt file of the most common nontransit bigrams
'''
templist=[]
newcorpus=[]
temparray2=np.zeros((numberoftopbigrams,)) 
counter=0

for i in np.arange(len(bigrams)):
    templist=list(bagofwords(bigrams[i],bagofbigrams_list))
    templist=list(templist)
    newcorpus.append(templist)

print(newcorpus) #this is a list of all the vectorized tweets

exportcorpus=newcorpus # i don't know why i did this

f= open("BigramsCorpus.txt","w") #this step outputs a txt file only so that we can easily remove the [ and ] from the list and output just numbers into a txt file
for i in np.arange(len(bigrams)): #this file will be deleted later, will be replaced by outputtransit.txt
     f.write(str(exportcorpus[i]))
     f.write("\n")
f.close()

with open("BigramsCorpus.txt",encoding="utf-8") as infile, open("OutputTransitBigrams.txt", 'w',encoding="utf-8") as outfile:
    for line in infile:
        line2=line.replace("[", "") #remove [ from start of list 
        line3=line2.replace("]", "") #remove ] from end of list
        outfile.write(line3)  
outfile.close()
infile.close()

os.remove("BigramsCorpus.txt") #deleted, we don't need it
'''