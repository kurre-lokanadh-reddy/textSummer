from nltk.corpus import stopwords

import numpy as np
import math
import re
import string
import networkx as nx   # FOR(PAGE RANK)

#from nltk.cluster.util import cosine_distance


### bellow are the pre-processing methods that are used in the model development as necessary.

def readarticle(filedata):
    #file=open(filename,'r')
    #filedata=file.read()
    article=filedata.split(".")
    sentences=[]
    for sentence in article:
        sentences.append(sentence.strip().replace("[^a-zA-Z]", " ").split(" "))
    return sentences

from nltk.stem import WordNetLemmatizer, LancasterStemmer
def readarticle_lema(file_text):
    #file=open(filename,'r',encoding="utf-8")
    #file_text=file.read()
    
    org_text = file_text
    org_text=re.sub("[()]",".",org_text)
    orginal = org_text.split(".")
    if len(orginal[-1])<5:
        orginal.pop(-1)
        
    
    file_text=re.sub("[()]",".",file_text)
    file_text=re.sub("[^a-zA-Z0-9.]"," ",file_text)
    file_text=re.sub('["]',' ',file_text)
    sents= file_text.split(".")
    if len(sents[-1])<5:
        sents.pop(-1)
    words=list()
    wordnet_lemmatizer = WordNetLemmatizer()
    for sent in sents:
        sent=sent.strip()
        lema_words=[wordnet_lemmatizer.lemmatize(word.lower(), pos="v") for word in sent.split()]
        words.append([word for word in lema_words if word not in list(stopwords.words('english'))])
    return orginal,words

def readarticle_stma(file_text):
    #file=open(filename,'r',encoding="utf-8")
    #file_text=file.read()
    
    org_text = file_text
    org_text=re.sub("[()]",".",org_text)
    orginal = org_text.split(".")
    if len(orginal[-1])<5:
        orginal.pop(-1)
    
    file_text=re.sub("[()]",".",file_text)
    file_text=re.sub("[^a-zA-Z0-9.]"," ",file_text)
    file_text=re.sub('["]',' ',file_text)
    sents= file_text.split(".")
    if len(sents[-1])<5:
        sents.pop(-1)
    words=list()
    lancaster=LancasterStemmer()
    for sent in sents:
        sent=sent.strip()
        stema_words=[lancaster.stem(word.lower()) for word in sent.split()]
        words.append([word for word in stema_words if word not in list(stopwords.words('english'))])
    return orginal,words

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


## for able to upload to github divided glove to two parts code is for loading that
def loadEmbeddingHalfs(ad1,ad2):
	f1=open(ad1,'r',encoding='utf-8')
	emb=dict()
	for line in f1:
	    values=line.split(" ")
	    word = values[0]
	    emb[word]=np.asarray(values[1:],dtype='float32')
	f1.close()
	f2=open(ad2,'r',encoding='utf-8')
	for line in f2:
	    values=line.split(" ")
	    word = values[0]
	    emb[word]=np.asarray(values[1:],dtype='float32')
	f2.close()
	emb.pop('\n')
	return emb

#emb_index= loadEmbeddingHalfs('very_large_data/glove.6B.50d.1.txt','very_large_data/glove.6B.50d.1.txt')


################################################################################################################################################################################################

## TEXT RANK 

def cosine_similarity(A,B):
    denom =np.linalg.norm(A)*np.linalg.norm(B)
    if denom==0:
        return 0.7
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
# main function
def TextRank_summary(file_name,ABSTRACT_SIZE=0.3):
    stop_words=stopwords.words('english')
    summary=[]
    act_sentences,sentences=readarticle_stma(file_name)
    similarityMatrix=build_similarity_matrix(sentences,stop_words)
    
    
    #ranking the sentences using Page Rank
    sentence_similarity_graph = nx.from_numpy_array(similarityMatrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    sorted_scores=list(sorted(scores.items(),key= lambda item:item[1],reverse=True))[:max(1,math.floor(len(act_sentences)*ABSTRACT_SIZE))]
    
    return ".".join([act_sentences[i[0]] for i in sorted(sorted_scores[:])])


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
# main method
def Luhn_summary(file_name,ABSTRACT_SIZE=0.3):
    actual,sentences = readarticle_stma(file_name)
    sentences = [" ".join(sentence) for sentence in sentences] #to make words list to sentence
    metric = top_words(sentences)
    scores = {}
    for i,sentence in enumerate(sentences):
        scores[i]=calculate_score(sentence, metric)
    

    #sorting according to the ranking
    sorted_scores=list(sorted(scores.items(),key= lambda item:item[1],reverse=True))[:max(1,math.floor(len(actual)*ABSTRACT_SIZE))]
    
    return ".".join([actual[i[0]] for i in sorted(sorted_scores)])

#################################################################################################################################################################################

## LSA method

def modified_tfidf(sentences , unique_words,modified=True):
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
    
    if modified:
        # modified Tf_IDF approch making the less than average values to zero to remove noise
        sent_avg = np.mean(tf_idf,axis=1)
        #print("average= ",sent_avg)
        res=[]
        for i in range(x):
            res.append(list(np.greater(tf_idf[i],sent_avg[i]).astype("int")))
        #print(np.count_nonzero(tf_idf==0) , np.count_nonzero((res*tf_idf)==0))
        return res*tf_idf
    else:
        return tf_idf
# main method
def LSA_summary(filename,ABSTRACT_SIZE=0.3):
    orginal,sentences = readarticle_stma(filename)
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
    scores = {i:j for i,j in enumerate(Lengths)}

    #sorting according to the ranking
    sorted_scores=list(sorted(scores.items(),key= lambda item:item[1],reverse=True))[:max(1,math.floor(len(orginal)*ABSTRACT_SIZE))]
    #print(sorted_scores)
    # Selecting the top sentences
    
    return ".".join([orginal[i[0]] for i in sorted(sorted_scores)])

###############################################################################################################################################################################################################################################

## FUZZY logic

#from fuzzyLogic.summerize import fuzzy_summary



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

    #for empty sentences
    if len(sent_emb)<1:
        return np.zeros(50)
    sent_emb=np.mean(sent_emb,axis=0)
    return np.array(sent_emb)
def Embeding_summary(file_name,ABSTRACT_SIZE=0.30):
    actual , sentences =readarticle_lema(file_name)
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
    print(closest)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([actual[closest[idx]] for idx in ordering[:min(max(2,math.floor(len(actual)*ABSTRACT_SIZE)),n_clusters)]])
    return summary

######################################################################################################################################

### OWN model (in simple its a combination of embedding with LSA technique )

''' we use the svd on whole text to remove the bottom 40% text and svd on topic clusters to draw attentio to the important topics.'''
def SVD_elimenation_clusterWise(tf_idf_vectors,indx,elimiate=False):
    U,s,V = np.linalg.svd(np.transpose(tf_idf_vectors))
    V_avg=np.mean(V,axis=1)
    #print("avg= " ,V_avg)
    
    # redusing the noices again 
    res=[]
    for i in range(len(V_avg)):
        res.append(list(np.greater(V[i],V_avg[i]).astype("int")))
    if elimiate:
        V= V*res
    
    # geting the sentence length values
    Lengths = np.sum(V,axis=0)
    return {i:j for i,j in zip(indx,Lengths)}

def get_abstract_ratio(counts,ABSTRACT_SIZE):
    f_c=[]
    c=[]
    for _,i in counts.items():
        f_c.append(math.floor(i*ABSTRACT_SIZE))
        c.append(float(i)*ABSTRACT_SIZE)
    return f_c,c

def remove_bottom_30(selected,scores):
    scores = sorted(scores.items(),key=lambda item:item[1])[:round(len(selected)*0.40)]
    for i in scores:
        selected[i[0]]=-1
    return selected

def get_sent_embedding_tfidf(wordlist,tf_idf,unique_words):
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
            
        sent_emb.append(np.array(res)*tf_idf[unique_words.index(i)])

    # multiplying with the tf_idf values

    # if sentence is empty
    if len(sent_emb)<1:
        return np.zeros(50)
    #calculating the mean
    sent_emb=np.mean(sent_emb,axis=0)
    return np.array(sent_emb)


def select_sentences_high(selected,scores,ratios):
    '''
    need to the cluster preference order based on the average sentence value.
    '''
    sele=0
    for i in range(len(ratios)):
        j=0
        #print("cluster []",i)
        sorted_ist = sorted(scores[i].items(),key=lambda item:item[1],reverse=True)
        while j<ratios[i]:
            top = sorted_ist.pop(0)
            index = top[0]
            if selected[index]==0:
                sele=sele+1
                selected[index]=1
                j=j+1
    return selected,sele

def select_sentences_low(selected , scores ,get_ratio):
    min_cover = [float(math.ceil(i))-i for i in get_ratio]
    min_index =min_cover.index(min(min_cover))
    j=0
    sorted_list=sorted(scores[min_index].items(),key=lambda item:item[1],reverse=True)
    while j<1:
        top = sorted_list.pop(0)
        index = top[0]
        if selected[index]==0:
            selected[index]=1
            j=j+1
    return selected,min_index

def OWN_summary(filename,ABSTRACT_SIZE=0.3):
    actual,sentences = readarticle_lema(filename)
    uniqueWords=get_uniquewords(sentences)
    tf_idf = modified_tfidf(sentences,uniqueWords,modified=True)
    print("loaded text and converted to matrix: done")
    # sentence wise normalizationof tf_idf values
    res=[]
    for i in tf_idf:
        denom=max(i)-min(i)
        if denom==0:
            res.append(i)
        else:
            res.append((i-min(i))/(max(i)-min(i)))
    res= np.array(res)
    res = res+1
    print("done1")
    # getting the sentence embedding 
    emb_sents = [get_sent_embedding_tfidf(sentences[i],res[i],uniqueWords) for i in range(len(sentences))]
    
    # clustering the sentences
    print("started clustering  :",end=" ")
    n_clusters = int(np.ceil(len(emb_sents)**0.5))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(emb_sents)
    print("done")
    
    print("applying SVD alogo :",end=" ")
    # implementation of the SvD method to give attention to the required sentences in each clusters.
    tf_idf_cluster=np.array(tf_idf)
    scores_cluster = {}
    counts={}
    for i in range(n_clusters):
        idx = list(np.where(kmeans.labels_ == i)[0])
        scores_cluster[i]=SVD_elimenation_clusterWise(tf_idf_cluster[idx],idx,elimiate=True)
        counts[i]=len(idx)
    # implementation of the SvD method to give attention on global level.
    scores_global = SVD_elimenation_clusterWise(tf_idf_cluster,[i for i in range(len(actual))])
    print("done")
    
    selected=[0 for i in range(len(actual))]
    
    # the bellow process eliminates bottom 30% of text before the selection step.
    selected = remove_bottom_30(selected,scores_global)
    #print(selected)
    print("removed bottom 40 percentel using globel level : done")
    # get top ABSTRACT_SIZE percent sente form each cluster
    
    print("constructing summary from top :",end=" ")
    
    get_ratio_f,get_ratio= get_abstract_ratio(counts,ABSTRACT_SIZE)
    
    to_be_selected=math.floor(len(actual)*ABSTRACT_SIZE)
    selected,cur_selected = select_sentences_high(selected,scores_cluster,get_ratio_f)
    print("done")
    while(cur_selected<to_be_selected):
        print("....calling function to rescue")
        selected , cl= select_sentences_low(selected , scores_cluster ,get_ratio)
        get_ratio[cl]=math.ceil(get_ratio[cl])
        cur_selected=cur_selected+1
    print("\n\n",selected,"\n")
    
    summary=[]
    for i in range(len(selected)):
        if selected[i]==1:
            summary.append(actual[i])
    return ".".join(summary)


# the below method calls the respective method for summarizaion as per the input given from the flask app.

def summerize(id,text,ABSTRACT_SIZE=0.3):
    if id=="TextRank":
        return TextRank_summary(text,ABSTRACT_SIZE);
    elif id=="Luhn":
        return Luhn_summary(text,ABSTRACT_SIZE);
    elif id=="LSA":
        return LSA_summary(text,ABSTRACT_SIZE);
    elif id=="Embeding":
        return Embeding_summary(text,ABSTRACT_SIZE);
    #elif id=="fuzzy":
    #    return fuzzy_summary(text,ABSTRACT_SIZE);
    elif id=="own":
    	return OWN_summary(text,ABSTRACT_SIZE);
    else:
        return "<h3>THIS MODEL IS CORRENTLY NOT READY </h3>"+TextRank_summary(text,ABSTRACT_SIZE);
