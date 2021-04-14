from nltk.corpus import stopwords

import numpy as np
import math
import re
import string
import networkx as nx   # FOR(PAGE RANK)

#from nltk.cluster.util import cosine_distance

def readarticle(filedata):
    #file=open(filename,'r')
    #filedata=file.read()
    article=filedata.split(".")
    sentences=[]
    for sentence in article:
        sentences.append(sentence.strip().replace("[^a-zA-Z]", " ").split(" "))
    return sentences

def get_uniquewords(sentences):
    stop_words=stopwords.words('english')
    unque=set()
    for sentence in sentences:
        for word in sentence:
            if word not in stop_words:
                unque.add(word)
    return sorted(list(unque))

def loadEmbeddingMatrix(EMBEDDING_FILE):
    embeddings_index = dict()
    #Transfer the embedding weights i dictionary by iterating through every line of the file.
    f = open(EMBEDDING_FILE,'r',encoding='utf-8')
    for line in f:
        #split up line into an indexed array
        values = line.split()
        #first index is word
        word = values[0]
        #store the rest of the values in the array as a new array
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs #50 dimensions
    f.close()
    #print('Loaded %s word vectors.' % len(embeddings_index))

    return embeddings_index #, embedding_matrix

## Loading 'glove' words
emb_index= loadEmbeddingMatrix('very_large_data/glove.6B.50d.txt')


################################################################################################################################################################################################

## TEXT RANK 

def cosine_similarity(A,B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    #return 1 - cosine_distance(vector1, vector2) ## this is using nltk function
    return cosine_similarity(vector1,vector2)
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        sent1=[i.lower() for i in sentences[idx1]]
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            sent2=[i.lower() for i in sentences[idx2]]
            similarity_matrix[idx1][idx2] = sentence_similarity(sent1, sent2, stop_words)

    return similarity_matrix
def TextRank_summary(file_name,ABSTRACT_SIZE=0.3):
    stop_words=stopwords.words('english')
    summary=[]
    sentences=readarticle(file_name)
    similarityMatrix=build_similarity_matrix(sentences,stop_words)
    
    #ranking the sentences using Page Rank
    sentence_similarity_graph = nx.from_numpy_array(similarityMatrix)
    scores = nx.pagerank(sentence_similarity_graph,max_iter=1000)
    
    #sorting according to the ranking
    ranked_sentence = sorted(((scores[i]," ".join(s)) for i,s in enumerate(sentences)), reverse=True)

    
    return ".".join([i[1] for i in ranked_sentence[:round(len(sentences)*ABSTRACT_SIZE)]])


################################################################################################################################################################################################

## LUHN Method

def top_words(sentences):
    record = {}
    common_words =  stopwords.words('english')  #load_common_words()
    for sentence in sentences:
        words = sentence.split()
        for word in words:  #sentences is already a list of words so no need to split again. .
            w = word.strip('.!?,()\n').lower()
            record[w]= record.get(w,0)+1

    for word in record.keys():
        if word in common_words:
            record[word] = -1     
    occur = [key for key in record.keys()]
    occur.sort(reverse=True, key=lambda x: record[x])
    return set(occur[: len(occur) // 10 ])
def calculate_score(sentence, metric):
    words = sentence.split()
    imp_words, total_words, begin_unimp, end, begin = [0]*5
    for word in words:
        w = word.strip('.!?,();').lower()
        end += 1
        if w in metric:
            imp_words += 1
            begin = total_words
            end = 0
        total_words += 1
    unimportant = total_words - begin - end
    if(unimportant != 0):
        return float(imp_words**2) / float(unimportant)
    return 0.0
def Luhn_summary(file_name,ABSTRACT_SIZE=0.3):
    sentences = readarticle(file_name)
    sentences = [" ".join(sentence) for sentence in sentences] #to make words list to sentence
    metric = top_words(sentences)
    scores = {}
    for sentence in sentences:
        scores[sentence] = calculate_score(sentence, metric)
    top_sentences =list(sentences) # make a copy
    top_sentences.sort(key=lambda x: scores[x], reverse=True)      # sort by score
    top_sentences = top_sentences[:round(len(scores)*ABSTRACT_SIZE)] # get top 5% (in persentage)
    top_sentences.sort(key=lambda x: sentences.index(x))           # sort by occurrence
    return '. '.join(top_sentences) 

#################################################################################################################################################################################

## LSA method

def modified_tfidf(sentences , unique_words):
    tf_idf= np.zeros((len(sentences),len(unique_words)))
    tot_frequency=dict()
    i=0
    # frequency matrix or TF values
    for sentence in sentences:
        for word in sentence :
            if word in unique_words:
                j =unique_words.index(word)
                freq = tf_idf[i][j]
                if freq==0 :
                    tot_frequency[word]=tot_frequency.get(word,0)+1
                tf_idf[i][j]=freq+1
        i=i+1
    #print(tot_frequency)
    #binary=tf_idf
    # calculating IDF values for all the unique values
    x,y = tf_idf.shape
    idf={}
    for i in tot_frequency.keys():
        idf[i]=math.log(x/tot_frequency[i])
    #print(idf)
    # calculating tf_idf values 
    for i in range(x):
        for j in range(y):
            tf_idf[i][j] = tf_idf[i][j]*idf[unique_words[j]]
    
    # modified Tf_IDF approch making the less than average values to zero to remove noise
    sent_avg = np.mean(tf_idf,axis=1)
    #print("average= ",sent_avg)
    res=[]
    for i in range(x):
        res.append(list(np.greater(tf_idf[i],sent_avg[i]).astype("int")))
    #print(np.count_nonzero(tf_idf==0) , np.count_nonzero((res*tf_idf)==0))
    return res*tf_idf
def LSA_summary(filename,ABSTRACT_SIZE=0.3):
    sentences = readarticle(filename)
    uniqueWords=get_uniquewords(sentences)
    tf_idf_vectors=modified_tfidf(sentences,uniqueWords)
    #print(len(sentences), len(uniqueWords), tf_idf_vectors.shape)
    U,s,V = np.linalg.svd(np.transpose(tf_idf_vectors))
    V_avg=np.mean(V,axis=1)
    #print("avg= " ,V_avg)
    
    # redusing the noices again 
    res=[]
    for i in range(len(V_avg)):
        res.append(list(np.greater(V[i],V_avg[i]).astype("int")))
    V= V*res
    
    # geting the sentence length values
    Lengths = np.sum(V,axis=0)
    #print(Lengths)
    # Selecting the top sentences
    sents_ord=sorted(sentences,key=lambda x: Lengths[sentences.index(x)] , reverse=True)
    return (".".join([" ".join(i) for i in sents_ord[:round(len(sentences)*ABSTRACT_SIZE) ]]))

###############################################################################################################################################################################################################################################

## FUZZY logic

from fuzzyLogic.summerize import fuzzy_summary



##################################################################################################################################################################

## Sentence embidding 

### DATA SET AMAZON REVIEWS DATA SET (OPTIONAL)
### PRE TRIENDED 50 DIMENSSIONS EMBIDINGS(GLOVE)


### k-means clustering and selection based on the distance to the center of the cluster.
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def get_sent_embedding(wordlist):
    """
    This function calculates the embedding of each sentence in the review. Checks if the sentence being passed is a valid one, 
    removing the punctuation and emojis etc.
    """
    sent_emb = []
    for i in wordlist:
        i = i.lower()
        try :
            res=list(emb_index[i])
        except:
            res=list(emb_index['unknown'])
        sent_emb.append(res)

    #calculating the mean 
    sent_emb=np.mean(sent_emb,axis=0)
    return np.array(sent_emb)
def Embeding_summary(file_name):
    sentences =readarticle(file_name)
    emb_sents=[get_sent_embedding(sent) for sent in sentences]
    sentences=[" ".join(sent) for sent in sentences]
    n_clusters = int(np.ceil(len(emb_sents)**0.5))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(emb_sents)
    avg = []
    closest = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        #print("IDX is: ", idx)
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_,emb_sents)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])
    return summary

def summerize(id,text):
    if id=="TextRank":
        return TextRank_summary(text);
    elif id=="Luhn":
        return Luhn_summary(text);
    elif id=="LSA":
        return LSA_summary(text);
    elif id=="Embeding":
        return Embeding_summary(text);
    elif id=="fuzzy":
        return fuzzy_summary(text);
    else:
        return "<h3>THIS MODEL IS CORRENTLY NOT READY </h3>"+TextRank_summary(text);
