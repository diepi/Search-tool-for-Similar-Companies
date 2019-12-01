#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:41:08 2019

@author: diepy
"""

import pandas as pd
import math
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string

import numpy as np

from scipy import spatial

from warnings import filterwarnings
filterwarnings('ignore')




    
# convert text to lower-case, tokenization and strip punctuation
def normalize_text(text):
    """
    text cleaning
    """
    text = str(text)
    norm_text = text.lower()
    norm_text = word_tokenize(norm_text)
    result = [] # store the text without punctuation
    for i, w in enumerate(norm_text):
        if w not in string.punctuation:
            result.append(w)
            
    return [" ".join(result)]



def DataPrepare(text):
    """
    text preparation for modeling
    """
    norm_text = [normalize_text(text.iloc[i]) for i in range(len(text))]
    tagged_data = [TaggedDocument(words=word_tokenize(d[0]), tags=[str(i)]) \
               for i, d in enumerate(norm_text)]
    return tagged_data

'''
Doc2vec Training:
1. epochs: The number of epochs. It defines the number times that the learning algorithm 
               will work through the entire training dataset. 
2. vec_size:   The size of Vector you want to have.
3. alpha:      The initial learning rate
4. min_alpha:  Learning rate will linearly drop to min_alpha as training progresses
5. min_count:  Ignores all words with total frequency lower than this.We set 1 because we want
               words can be count into the vector. Otherwise, some unique sentence labels
               may be out of consideration, which is not what we want.
6. dm:         Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. 
               Otherwise, distributed bag of words (PV-DBOW) is employed.

More details visit: https://radimrehurek.com/gensim/models/doc2vec.html
'''

# train the model
def similaritymodel(preparedtext):
    """
    modelling data
    """
    max_epochs = 500
    vec_size = 20
    alpha = 0.025

    # dm = 1 means ‘distributed memory’ (PV-DM)
    # dm = 0 means ‘distributed bag of words’ (PV-DBOW)
    model = Doc2Vec(vector_size =vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                window=10,
                dm=1)

    model.build_vocab(preparedtext)

    for epoch in range(max_epochs):
    #print('iteration {0}'.format(epoch))
        model.train(preparedtext,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
        model.alpha -= 0.0002
    # fix the learning rate, no decay
        model.min_alpha = model.alpha
        
    cosine_matrix = np.zeros((len(preparedtext), len(preparedtext)))

    for i in range(len(preparedtext)):
        for j in range(len(preparedtext)):
            cosine_matrix[i][j] = spatial.distance.cosine(model.docvecs[i], model.docvecs[j])
    cosine_matrix = pd.DataFrame(cosine_matrix)
    #save the similarity matrix for later use
    cosine_matrix.to_csv('cosine_matrix.csv',index=False)
    
    # output the representationl vector
    vec = []
    for i in np.arange(0,len(preparedtext),1):
        vec.append(list(model.docvecs[str(i)]))
    vec = np.matrix(vec)
    # save vectors into .csv file
    pd.DataFrame(vec).to_csv("vec.csv",index=False)
    


def get_similarities(cosine_matrix, company_id, textdata, data):
    """
    Get the similar campanies info by the given company id
    """

    similarities  = 1 - cosine_matrix[company_id, :]
    similarities_df = pd.DataFrame({'company_id': textdata.index, 'similarity':similarities})
    
    # sort the values
    similarities_df.sort_values(by='similarity', ascending=False, inplace=True)
    similarities_df.reset_index(drop=True, inplace=True)

    # drop itself
    similarities_df = similarities_df.iloc[1:,:]
    similarities_df.reset_index(drop=True, inplace=True)

    # merge the company name into the dataframe
    similar_name = data.iloc[similarities_df['company_id'],:]['Name'].reset_index(drop=True)
    similar_sector = data.iloc[similarities_df['company_id'],:]['Sector'].reset_index(drop=True)
    market_cap = data['Marketcap']
    similarities_df = pd.DataFrame({'Similar_company': similar_name, 'Sector':similar_sector, 'MarketCap':market_cap})      
    return similarities_df

def search_1(cm,data):
    """
    searches for similar company
    """
    df = data['Aggregate_text']
    name = data['Name']
    while True:
        company = input("Company name:").lower()
        if company != "":
                matching = [s for s in name if company in s.lower()]
                if len(matching)==0:
                    print("There is no company with this name in the system.")
                    print("Please enter again the company name.")
                elif len(matching)>1:
                    print("Please specify one of the companies below:")
                    print(matching)
                else:
                    answer = input("Do you mean: {}?[y/n]".format(matching[0]))
                    if answer !="y":
                        print("Please enter again the company name.")
                    else:
                        while True:
                            a1 = input("Do you want to narrow down the search by company size?[y/n]")
                            try:
                                if a1 == 'n':
                                    a2 = input("How many companies do you want to find? Enter number:")   
                                    try:
                                        val = int(a2)
                                        company_id = [i for i, n in enumerate(name) if n == matching[0]]
                                        company_id = int(company_id[0])
                                        similar_df_1 = get_similarities(cm, company_id, df, data)
                                        similar_df_1 = similar_df_1[['Similar_company', 'MarketCap']]
                                        print('Search: {0},  Market Capital: {1}'.format(data.iloc[company_id,:]['Name'],data['Marketcap'][company_id]))
                                        print(similar_df_1[:val])
                                        return
                                    except ValueError:
                                        print("This is invalid number.")
                                if a1 == 'y':
                                    print('ddd')
                                    a21 = input("Please enter the lower bound for size (in millions):")
                                    a22 = input("Please enter the upper bound for size (in millions):")
                                    try:
                                        low = int(a21)
                                        up = int(a22)
                                        if up - low < 0:
                                            raise ValueError("lower bound or upper bound does not match", up - low)
                                    except ValueError:
                                        print("This is invalid input.")
                                        continue
                                    else:
                                        company_id = [i for i, n in enumerate(name) if n == matching[0]]
                                        company_id = int(company_id[0])
                                        similar_df = get_similarities(cm, company_id, df, data)
                                        similar_df_1 = similar_df[similar_df['MarketCap'] > low]
                                        similar_df_1 = similar_df_1[similar_df_1['MarketCap'] < up]
                                        a23 = input("How many companies do you want to find? Enter number:")
                                        try:
                                            val1 = int(a23)
                                            similar_df_2 = similar_df_1[['Similar_company', 'MarketCap']]
                                            print('Search: {0},  Market Capital: {1}'.format(data.iloc[company_id,:]['Name'],data['Marketcap'][company_id]))
                                            print(similar_df_2[:val1])
                                            return
                                        except ValueError:
                                            print("This is invalid number.")
                                            break
                                else:
                                    raise AssertionError("Unexpected value", a1)
                            except AssertionError:
                                print("Wrong enter.")

        


def count(dictionary, dt):
    """
    counts frequency of each word in dictionary for each documents
    """
    dictionary = dictionary.split()
    mat = pd.DataFrame()
    for j in range(len(dt)):
        freq = np.array([])
        for i in range(len(dictionary)):
            n=0
            for word in dt.Aggregate_text[j].split():
                if word == dictionary[i]:
                    n += 1
            freq = np.append(freq, n)
        mat[dt.Name[j]] = freq
    mat.index = dictionary
    return mat


def idf(freqmatrix, word):
    """
    calculate inverse document frequency, overall number of documents is the number of columns in the matrix. 
    """
    _, n_docs = freqmatrix.shape
    df = np.count_nonzero(freqmatrix.loc[word])
    try:
        idf = math.log(float(n_docs) / df)
    except ZeroDivisionError:
        idf = 0
    return idf

def cosine_similarity(a, b):
    """
    return cosine similarity between 2 vectors
    """
    return np.dot(a,b) / (math.sqrt(np.dot(a, a)) * math.sqrt(np.dot(b, b)))

def query_vector(freqmatrix, dictionary):
    """
    computes the inverse document frequency between the dictionary and document
    """
    dictionary = set(dictionary.split())
    n_terms, _ = freqmatrix.shape
    query_vector = np.zeros(n_terms)
    
    for idx, term in enumerate(freqmatrix.index):
        if term in dictionary:
            query_vector[idx] = idf(freqmatrix, term)

    return query_vector

def query(index, query_terms):
    """
    computes tf idf similarity between dictionary and document, decides which sector has highest probability for each company and saves it to csv.
    """
    q = query_vector(index, query_terms)
    n_terms, _ = index.shape

    results = np.array([])
    for doc in index:
        doc_vec = np.zeros(n_terms)
    
        for (idx, (term, tf)) in enumerate(index[doc].iteritems()):
            doc_vec[idx] = tf * idf(index, term)
    
        results = np.append(results, cosine_similarity(q, doc_vec))
    return results

def tfidf(dictionary,data):
    """
    find similarity business defined by the dictionary of words
    """
    matrixlist = []
    for i in range(len(dictionary)):
        matrixlist.append(count(dictionary.Dictionary[i],data))
    similarity = pd.DataFrame()
    for i in range(len(dictionary)):
        similarity[i] = query(matrixlist[i],dictionary.Dictionary[i])
    similarity.index = matrixlist[0].columns
    similarity.columns = ['Finance & Banking', 'Information Technology', 'Constructions', 'Energy', 'Mining', 'Support Service', 'Healthcare', 'Chemical', 'Manufacturing', 'Real Estate', 'Trave & Leisure', 'Wholesale & Retail', 'Media', 'Food', 'Logistic & Transport', 'Aerospace & Defense']
    sectortfidf = list()
    for i in range(len(similarity)):
        sectortfidf.append(np.argmax(similarity.iloc[i]))
    sectortfidf = np.asarray(sectortfidf)
    similarity['sectortfidf']=sectortfidf 
    similarity = similarity['sectortfidf']
    similarity.to_csv('similarityidf.csv')
    
def search_2(dictionary, similarityidf):
    """
    finds all companies in the sector based on the keyword in dictionary.
    """
    while True:
        key = input('Type any keywords for to define the company activities or company products: ')
        matching = list()
        for i in range(len(dictionary)):
            if key.lower() in dictionary.Dictionary[i].lower():
                matching.append(dictionary.Sector[i])
        if len(matching)==0:
            print("Keywords not found in the record.")
            print(" Please input different keywords.")
        elif len(matching)>1:
            print("The keywords were found in one of the sectors below:")
            print(matching)
            while True:
                sector=input('Please specify the sector:').lower()
                sectorlist = [matching.index(sec) for sec in matching if sector in sec.lower()]
                try:   
                    if len(sectorlist)==0:
                        raise AssertionError('There is no such sector in the list.', sector)
                    elif len(sectorlist)>1:
                        raise AssertionError('The input has too broad meaning.', sector)
                    else:
                        print('The companies defined with the keywords "{}" under sector "{}" are shown below.'.format(key, matching[sectorlist[0]]))
                        print(similarityidf.Name[similarityidf['Sector']==matching[sectorlist[0]]])
                        return
                except AssertionError:
                    print('Please input the sector again.')             
        else:
            print('The keywords "{}", were found under sector "{}". The companies defined with these keywords are shown below.'.format(key, matching[0]))
            print(similarityidf.Name[similarityidf['Sector']==matching[0]])
            break
        
        
    







# ------------------- Main function --------------------

def FindSimilarBusinesses():
    """
    main engine
    """
    print("*"*55)
    print("***"+" "*8+"WELCOME TO PEAK SEARCH TOOL!"+" "*8+"***")
    print("*"*55,"\n")
    data=pd.read_csv("compdata.csv")
    df = data['Aggregate_text']
    dictionary = pd.read_csv("dictionary.csv")
    #first step - building model or not?
    print("Is this you first time using this tool?")
    while True:
        try:
            answer=input("[y/n]:")
            if answer == "y":
                print("Please, wait ...")
                tagged_data =DataPrepare(df)
                similaritymodel(tagged_data)
                tfidf(dictionary,data)
                cm = pd.read_csv('cosine_matrix.csv')
                cm = cm.values
                similarityidf = pd.read_csv("similarityidf.csv", names=['Name', 'Sector'], header = None)
                break
            elif answer == "n":
                print("Loading the model ...")
                cm = pd.read_csv('cosine_matrix.csv')
                cm = cm.values
                similarityidf = pd.read_csv("similarityidf.csv", names=['Name', 'Sector'], header = None)
                break
            else:
                print("Is this you first time using this tool?")
        except FileNotFoundError:
            print("The record was not found, please try again.")
    
    #search
    while True:
        answer = input('Which search do u wish to turn on? 1 for company search, 2 for sector search:')
        try:
            answer=int(answer)
            if answer == 1:
                search_1(cm,data)
            elif answer == 2:
                search_2(dictionary, similarityidf)
            else:
                raise ValueError('Invalid number.')
        except ValueError:
            print('Please enter 1 or 2.')
            
        while True:
            try:
                print("Do you wish to start another search?")                     
                a2=input("[y/n]:")
                if a2 == "y":
                    break
                elif a2 == "n":
                    return
                else:
                    raise AssertionError("Unexpected value of 'distance'!", a2)
            except AssertionError:
                print("Wrong enter.")
                            
                            
                            
                            
                       
 
                        
                            
                            
                            
if __name__ == '__main__' or __name__ == 'builtins':
    FindSimilarBusinesses()       
  