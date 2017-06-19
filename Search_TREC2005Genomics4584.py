'''
Created on Aug 1, 2015
Last updated on Aug 6, 2015

@author: Wei Wei
'''

import os,sys,lucene,pickle, nltk, string, time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# import multiprocessing as mp
from Bio import Entrez
from pprint import pprint 

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version

from org.apache.lucene.analysis.core import LowerCaseFilter, StopFilter, StopAnalyzer
from org.apache.lucene.analysis.en import PorterStemFilter
from org.apache.lucene.analysis.standard import StandardTokenizer, StandardFilter
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.search.similarities import BM25Similarity

Entrez.email = "your@email.com"

class PorterStemmerAnalyzer(PythonAnalyzer):
    def createComponents(self, fieldName, reader):
        source = StandardTokenizer(Version.LUCENE_CURRENT, reader)
        filter = StandardFilter(Version.LUCENE_CURRENT, source)
        filter = LowerCaseFilter(Version.LUCENE_CURRENT, filter) # normalize token text to lower case
        filter = PorterStemFilter(filter) # transform the token stream as per the Porter stemming algorithm
        filter = StopFilter(Version.LUCENE_CURRENT, filter,
                            StopAnalyzer.ENGLISH_STOP_WORDS_SET)
        return self.TokenStreamComponents(source, filter)

class RetBM25(object):
    '''This class retrieves documents from the indexed corpus.
       The default similarity metric is BM25. Terms in the queries are either stemmed using Porter stemmer or not stemmed.
    '''
    def __init__(self,base_dir, index_dir,index_file,queryDict):
        self.baseDir = base_dir
        self.indexFile = os.path.join(index_dir,index_file)
        lucene.initVM(vmargs=['-Djava.awt.headless=true']) # uncomment when run Retrieve separately
        directory = SimpleFSDirectory(File(self.indexFile))
        searcher = IndexSearcher(DirectoryReader.open(directory))
        self.BM25(searcher,queryDict)
        del searcher
            
    def BM25(self,searcher,queryDict):
        '''Retrieve similar documents and rank them using BM25'''
        # set up the searcher
        searcher.setSimilarity(BM25Similarity(1.2,0.75)) # set BM25 as the similarity metric, k=1.2, b=0.75
        results = {}
        for pmid,ab in queryDict.iteritems():
            retRes = self.__BM25(searcher, ab)
            results[pmid] = retRes
        ## save BM25 retrieval results
        stemmer = self.indexFile.split("/")[-1].split(".")[0].split("_")[-1]
        bm25OutFile = "BM25_similarArticles_%s.pkl"%(stemmer)
        pickle.dump(results,file(os.path.join(self.baseDir,"BM25_results",bm25OutFile),"w"))
        
    def __BM25(self,searcher,rawQuery):
        '''retrieve documents with a single query'''
        if 'Standard' in self.indexFile:
            analyzer = StandardAnalyzer(Version.LUCENE_CURRENT) # build a standard analyzer with default stop words
        if 'Porter' in self.indexFile:
            analyzer = PorterStemmerAnalyzer()

        query = QueryParser(Version.LUCENE_CURRENT, "contents", analyzer).parse(QueryParser.escape(rawQuery)) # escape special characters
        scoreDocs = searcher.search(query, 100).scoreDocs
        docList = []
        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)
            docList.append(doc.get("name"))
        return docList
            
def prepQueries(inputFile):
    '''Output a list of queries. Every item is an abstract from the 4584 articles'''
    texts = filter(None,file(inputFile).read().split("\n\n"))
    query = {}
    for t in texts:
        pair = filter(None,t.split("\n"))
        if len(pair)>1:
            query[pair[0]] = pair[1]
    return query

if __name__ == '__main__':
    base_dir = "/home/w2wei/Research/mesh/data/TREC/2005/4584rel"
    rawPMID_dir = "/home/w2wei/Research/mesh/data/TREC/2005/genomics.qrels.large.txt"
    medline_dir = os.path.join(base_dir,"medline")
    database_dir = os.path.join(base_dir,"database")
    index_dir = os.path.join(base_dir,"index")
#     analyzerType = "Standard"
    analyzerType = "PorterStemmer"
    index_file = "4584_MEDLINE_%s.index"%analyzerType
    
    # Retrieve documents
    queries = prepQueries(os.path.join(base_dir,"query.txt"))
    ret = RetBM25(base_dir, index_dir, index_file, queries)
