'''
Collect MEDLINE of 4.5 M articles and store the records on disk, multiple articles per file.
Extract PMID, titles and abstracts. Save the results in a new database.
Build a vocabulary from extracted titles and abstracts.
Vectorize every article. Save vectors/sparse matrices in multiple pickle files.
 a) Load all vectors using the sparse matrix
 b) Load a subset of vectors/matrices
 c) Load subsets using the sparse matrix
Similarity comparison. Use matrix manipulation.
 a) If all vectors can be loaded into the memory, compute similarities directly.
 b) If use block-wise comparison, iterate over all blocks.
Save similarity scores to DB.
 a) For every PMID:knnPMID, make an insertion
 b) For every block of PMID:knnPMID, make an insertion.

Created on Aug 10, 2015
Last updated Aug. 20, 2015

08/20 edits: Output the approximate matches and the contribution of the match to the final similarity score, i.e., why two documents are similar.


@author: Wei Wei
'''
from Bio import Entrez
import string, pickle, re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
import numpy as np
from gensim import models
from ParameterSetting import *
# from TREC2005.Index_TREC2005Genomics4584 import Corpus
from collections import Counter
import SkipGramGenerator as sgg
Entrez.email = "your@email.com"

class PRC(object):
    '''This class re-rank the BM25 results using Lin and Wilbur's PRC algorithm.'''
    def __init__(self, pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir):
        self.dbDir = database_dir # /home/w2wei/Research/mesh/data/TREC/2005/4584rel/database/
        self.baseDir = base_dir
        self.stemmed_corpusDir = stemmed_corpus_dir
        self.vocabDir = vocab_dir
        self.resDir = result_dir
        self.pmidList = pairDict.values()[0]
        
        self.vocab = []
        self.corpus = []
        self.stemmed_corpus = []
        self.df = None # doc freq vector
        self.doclen = None # doc len vector
        self.doc_term_matrix = None # doc-term count matrix
        self.prc_matrix = None # PRC score matrix
        self.sim_matrix = None # Similarity score matrix
        self.prc_rankedHits = []

    def run_PRC(self):
        '''run the experiment to get PRC top hits'''
        self.getVocab()
        self.vectorizeText()
        self.buildDocFreq()
        self.calPRCscores()
        self.cal_PRC_Similarity()
     
    def getVocab(self): 
        try:
            self.vocab = pickle.load(file(os.path.join(self.vocabDir,self.pmidList[0])))
            self.stemmed_corpus = pickle.load(file(os.path.join(self.stemmed_corpusDir,str(self.pmidList[0]))))
        except Exception as e:
            print e
            self.buildVocab() # including both self.vocab and self.stemmed_corpus
            
    def buildVocab(self):
        '''Build a vocabulary for the selected documents (from dir database).'''
        ## Note: The source of text should be Lucene processed field values. Lucene tokenized the text, remove stop words, and may take other unknown steps.
        ## Right now the vocabulary is built on the raw text with NLTK based stopwords removal, and tokenization. This should be improved.
        # collect contents from /database/ for each of these doc
        for pmid in self.pmidList: # self.pmidList includes the query and the 99 most similar articles selected by BM25
            self.corpus.append(file(os.path.join(self.dbDir,pmid)).read()) # corpus contains raw text (MH, title*2, abstract)
        for text in self.corpus:
            sent_tokenize_list = sent_tokenize(text.strip().lower(), "english") # tokenize an article text
            stemmed_text = []
            if sent_tokenize_list: # if sent_tokenize_list is not empty
                porter_stemmer = PorterStemmer()
                for sent in sent_tokenize_list:
                    words = TreebankWordTokenizer().tokenize(sent) # tokenize the sentence
                    words = [word.strip(string.punctuation) for word in words]
                    words = [word for word in words if not word in stopwords.words("english")]               
                    words = [word for word in words if len(word)>1] # remove single letters and non alphabetic characters               
                    words = [word for word in words if re.search('[a-zA-Z]',word)]                        
                    words = [porter_stemmer.stem(word) for word in words] # apply Porter stemmer                     
                    stemmed_text.append(" ".join(words))
                    self.vocab+=words
            self.stemmed_corpus.append(". ".join(stemmed_text)) # append a stemmed article text
        # save stemmed corpus
        pickle.dump(self.stemmed_corpus, file(os.path.join(self.stemmed_corpusDir,str(self.pmidList[0])),"w"))
        # remove low frequency tokens and redundant tokens
        tokenDist = Counter(self.vocab)
        lowFreqList = []
        for token, count in tokenDist.iteritems():
            if count<2:
                lowFreqList.append(token)
        self.vocab = list(set(self.vocab)-set(lowFreqList))
        # save vocabulary
        pickle.dump(self.vocab,file(os.path.join(self.vocabDir,str(self.pmidList[0])),"w"))
 
    def vectorizeText(self):
        '''This function converts every article (title and abstract) into a list of vocabulary count'''
        vectorizer = CountVectorizer(analyzer='word', vocabulary=self.vocab,dtype=np.float64) # CountVectorizer cannot deal with terms with hyphen inside, e.g. k-ras. CountVectorizer will not count such terms.
        self.doc_term_matrix = vectorizer.fit_transform(self.stemmed_corpus) # for Porter stemmer
#         self.doc_term_matrix = vectorizer.fit_transform(self.corpus) # for Standard analyzer, no stemmer
#         self.doclen = self.doc_term_matrix.sum(1) # self.doclen format is CSC, not numpy
        self.doc_term_matrix = self.doc_term_matrix.A
        self.doclen = np.sum(self.doc_term_matrix,1)
        self.doclen = self.doclen.reshape((len(self.doclen),1))
        
    def buildDocFreq(self):
        '''Count documents that contain particular words'''
        vectorizer = CountVectorizer(analyzer='word', vocabulary=self.vocab, binary=True)
#         doc_term_bin_matrix = vectorizer.fit_transform(self.corpus) # for standard analyzer, no stemmer
        doc_term_bin_matrix = vectorizer.fit_transform(self.stemmed_corpus) #for Porter stemmer
        self.df = doc_term_bin_matrix.sum(0)

    def calPRCscores(self):
        '''Calculate the weight of every term per document, using PubMed Related Citation (PRC) algorithm, Jimmy Lin and John Wilbur 2007.
           input: idf vector, docLen vector, occurrence count matrix (n documents, all terms in the vocabulary)
           output: a matrix of PRC scores.
        '''
        la = 0.022
        mu = 0.013
        div = mu/la
        ## generate m1
        reciSqrtIdf = np.reciprocal(np.sqrt(np.log(len(self.stemmed_corpus)*2.0/(self.df+1)))) # dim 1*19, conversion verified
        expDoclen = np.exp(self.doclen*(la-mu)) # dim 10*1, conversion verified
        m1 = np.dot(expDoclen,reciSqrtIdf) # dim 10*19, product verified
        ## generate m2: matrix
        matrix = np.power(div,self.doc_term_matrix)/div
        ## Hadamard product
        matrix = np.multiply(matrix,m1)
        ## offset
        offset = np.dot(np.ones((matrix.shape[0],1)),reciSqrtIdf)
        ## matrix+offset
        matrix = matrix+offset
        ## reciprocal of recWt
        raw_prc_matrix = np.reciprocal(matrix)
        ## reset scores for the terms that do not occur
        label = (self.doc_term_matrix>0)
        self.prc_matrix = np.multiply(label, raw_prc_matrix)

    def cal_PRC_Similarity(self):
        '''Measure the similarity between every pair of articles using PRC scores of terms in common between documents'''
        self.sim_matrix = np.dot(self.prc_matrix,self.prc_matrix.T)
        ## get a ranked similar doc list
        scoreList = self.sim_matrix[0,:].tolist()[0]
        self.prc_rankedHits = [pmid for (score,pmid) in sorted(zip(scoreList,self.pmidList),reverse=True)]
        if self.prc_rankedHits[0]!= self.pmidList[0]:
            print self.prc_rankedHits[:5]
        ## save results
        outDir = self.resDir # os.path.join(self.baseDir,"prc_ranks","PorterStemmer")
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        outFile1 = os.path.join(outDir,self.pmidList[0]+".txt")
        fout = file(outFile1,"w")
        fout.write("\n".join(self.prc_rankedHits))
        outFile2 = os.path.join(outDir,self.pmidList[0]+"_"+"score.pkl")
        fout = file(outFile2,"w")
        pmidwScore = sorted(zip(scoreList, self.pmidList),reverse=True)
        pickle.dump(pmidwScore,fout)
        
class MPRC_SKG(PRC):
    '''Adjust the count of terms in every article, according to the similarity of two terms in a skip bigram. No expansion is used. SKG for skip-gram '''
    def __init__(self,pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir, knnterm_dir, model):
        super(MPRC_SKG,self).__init__(pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir)
        self.model = models.Word2Vec.load(model) # Load trained Word2Vec model
        self.mprcSKG_rankedHits = []
        self.mprcSKG_matrix = None # MPRC score matrix

    def run_MPRC_SKG(self):
        '''run the experiment to get PRC top hits'''
        self.getVocab() # get the vocabulary of a query article and 99 similar articles selected by BM25
        self.vectorizeText() # get the word count per document
        self.adjustWeights()
        self.buildDocFreq() # get the document frequency for every word in the vocabulary
        self.calPRCscores() # calculate the weights
#         self.adjustScores() # remove comments if apply weight adjustments to PRC score matrix.
        self.cal_PRC_Similarity() # calculate the similarity

    def adjustScores(self): 
        '''Parse every article in the stemmed corpus, generate skip-grams from all sentences in every article.
           Calculate the similarity of two terms in a bigram according to trained Word2Vec model.
           Divide the similarity score and assign half of the score to each term in the pair. 
           Apply weight adjustments to the PRC score matrix'''        
        for article in self.stemmed_corpus:
            weightDict = self.getWeightFromSKG(article)
            articleIndex = self.stemmed_corpus.index(article)
            self.updateSimScoreMatrix(weightDict, articleIndex)
    
    def updateSimScoreMatrix(self, weightDict, articleIndex):
        validTerms = list(set(weightDict.keys()).intersection(set(self.vocab))) # terms such as 'k-ra' is included, but its count in doc-term matrix is 0 because CountVectorizer cannot process terms with punctuation inside
        for term in validTerms: # consider replace - with some combination of letters, such as aaa, hyphen, etc.
            self.prc_matrix[articleIndex,self.vocab.index(term)] = self.prc_matrix[articleIndex,self.vocab.index(term)]*weightDict[term]
    
    def adjustWeights(self): 
        '''Parse every article in the stemmed corpus, generate skip-grams from all sentences in every article.
           Calculate the similarity of two terms in a bigram according to trained Word2Vec model.
           Divide the similarity score and assign half of the score to each term in the pair.'''        
        for article in self.stemmed_corpus:
            weightDict = self.getWeightFromSKG(article)
            articleIndex = self.stemmed_corpus.index(article)
            self.updateDocTermCountMatrix(weightDict,articleIndex) # update every row in doc-term count matrix            
    
    def updateDocTermCountMatrix(self,weightDict,articleIndex):
        '''Modify doc-term count matrix self.doc_term_matrix by multiplying the weights from SKG to the original counts.'''
        validTerms = list(set(weightDict.keys()).intersection(set(self.vocab))) # terms such as 'k-ra' is included, but its count in doc-term matrix is 0 because CountVectorizer cannot process terms with punctuation inside
        for term in validTerms: # consider replace - with some combination of letters, such as aaa, hyphen, etc.
            self.doc_term_matrix[articleIndex,self.vocab.index(term)] = self.doc_term_matrix[articleIndex,self.vocab.index(term)]*weightDict[term]
    
    def getWeightFromSKG(self, article):
        '''Calculate the normalized weights from skip-grams'''
        skgList = self.getSkipGrams(article) # all the skip-grams of an article
        pairSimScore = self.getSKGtermSimilarity(skgList)# calculate the similarity between every pair of terms in the skgList
        rawWeights = self.calWeights(pairSimScore)# get the weight adjustment for every term
        tokenCount = self.countTerm(article)
        weightDict =self.normalizeWeights(rawWeights, tokenCount)
        return weightDict
    
    def normalizeWeights(self, rawWeights, tokenCount):
        '''Normalize the weight of every term according to its occurrence in the text'''
        for term in rawWeights.keys():
            rawWeights[term] = rawWeights[term]*1.0/tokenCount[term]
        return rawWeights
    
    def countTerm(self,article):
        '''Count the occurrence of every token in the text. Single token sentences (MH) are ignored because they do not produce skip grams.'''
        sents = article.split(". ")
        sents = [sent for sent in sents if len(sent.split(" "))>1]
        tokenList = [sent.split(" ") for sent in sents]
        tokenList = [item for sublist in tokenList for item in sublist]
        tokenCount = Counter(tokenList)
        return tokenCount
        
    def calWeights(self, scores):
        '''Calculate the weights assigned to every term'''
        scoreDict = {}
        for pairScore in scores:
            term1, term2 = pairScore[0]
            simScore = pairScore[1]
            if term1 not in scoreDict.keys():
                scoreDict[term1]=simScore
            else:
                scoreDict[term1]+=simScore
            if term2 not in scoreDict.keys():
                scoreDict[term2]=simScore
            else:
                scoreDict[term2]+=simScore
        return scoreDict
        
    def getSKGtermSimilarity(self, skgList):
        '''Get the similarity of two terms according to the trained Word2Vec model'''
        for pair in skgList:
            try:
                score = self.model.similarity(pair[0], pair[1])
                yield pair, score
            except Exception as e:
#                 print "Missed pair ", pair # if a term is not in Word2Vec vocabulary, any pair contains this term will come to exception.
                pass
       
    def getSkipGrams(self, article):
        skgList = []
        sents = article.split(". ")
        for sent in sents:
            tokenList = sent.split(" ")
            skgList+=sgg.kskipngrams(tokenList, len(tokenList), 2) # skip-grams are collected from the whole sentence. May expand this to the entire text.
        return skgList

class MPRC(PRC):
    '''This class re-rank the BM25 results using the MPRC algorithm.'''
    def __init__(self,pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir, knnterm_dir, model):
        super(MPRC,self).__init__(pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir)
        self.knntermDir = knnterm_dir
        self.model = models.Word2Vec.load(model) # Load trained Word2Vec model
        self.knnTermDict = {}
        self.mprc_rankedHits = []
        self.mprc_matrix = None # MPRC score matrix
            
    def run_MPRC(self):
        '''run the experiment to get MPRC top hits'''
        self.getVocab()
        self.vectorizeText()
        self.buildDocFreq()
        self.getKNNterms()
        self.calMPRCscores()
        self.cal_MPRC1_similarity()
    
    def getKNNterms(self):
        '''prepare knn terms of the QUERY TEXT VOCABULARY from the trained word2vec model. '''
        try:
            if "PorterStemmer" in self.resDir: # if Word2Vec model was trained on stemmed texts
                self.knnTermDict = pickle.load(file(os.path.join(self.knntermDir, "knnTermDict_PorterStemmer_%s.pkl"%self.pmidList[0])))
            if "Standard" in self.resDir: # if Word2Vec model was trained on the raw texts
                self.knnTermDict = pickle.load(file(os.path.join(self.knntermDir, "knnTermDict_Standard_%s.pkl"%self.pmidList[0])))
        except:
            for term in self.vocab:
                try:
                    knnTerms = self.model.most_similar(term, topn=5)
                    knnTerms = [t[0] for t in knnTerms]
                    self.knnTermDict[term]=knnTerms
                except:
                    pass
            if "PorterStemmer" in self.resDir: # if Word2Vec model was trained on stemmed texts
                print os.path.join(self.knntermDir, "knnTermDict_PorterStemmer_%s.pkl"%self.pmidList[0])
                pickle.dump(self.knnTermDict,file(os.path.join(self.knntermDir, "knnTermDict_PorterStemmer_%s.pkl"%self.pmidList[0]),"w"))
            if "Standard" in self.resDir: # if Word2Vec model was trained on the raw texts
                print os.path.join(self.knntermDir, "knnTermDict_PorterStemmer_%s.pkl"%self.pmidList[0])
                pickle.dump(self.knnTermDict,file(os.path.join(self.knntermDir, "knnTermDict_Standard_%s.pkl"%self.pmidList[0]),"w"))
    
    def calMPRCscores(self):
        '''Calculate Modified PRC score matrix'''
        ## count matrix 
        d1_vocab = np.where(self.doc_term_matrix[0,:]>0)[0].tolist() # d1_vocab is a list of index, not acutal terms. These terms verified. 
#         query_vocab_index = np.where(self.doc_term_matrix[self.pmidList.index(self.query),:]>0)[0].tolist() # query_vocab is a list of index, not acutal terms
        curr_doclen = np.sum(self.doc_term_matrix[0,:]) # the length of the query text
        newMx  = self.doc_term_matrix # a numpy matrix
        for ind in d1_vocab:
            ori_t =self.vocab[ind]
            if ori_t in self.knnTermDict.keys():
                knn_t = self.knnTermDict[ori_t]
                knn_t = [t for t in knn_t if t in self.vocab]
                knn_index = [self.vocab.index(t) for t in knn_t]
                t_index = [ind]+knn_index
                subMx = self.doc_term_matrix[:,t_index] # The columns of subMx include the original terms in d0 and their similar terms
                newMx[1:,ind] = np.sum(subMx,axis=1)[1:] # Count the occurrence of the original term and its associated similar terms in all other documents
                newMx[:,ind] = newMx[:,ind]*(newMx[0,ind]/curr_doclen) # weight every term by the percentage of this term in the original text
        newMx = newMx[:,d1_vocab] # the new count matrix weighted by percentage
        self.doc_term_matrix = newMx
        
        ## MPRC
        la = 0.022
        mu = 0.013
        div = mu/la
        ## generate m1
        reciSqrtIdf = np.reciprocal(np.sqrt(np.log(len(self.stemmed_corpus)*2.0/(self.df+1)))) # dim 1*19, conversion verified
        reciSqrtIdf = reciSqrtIdf[0,d1_vocab]
        expDoclen = np.exp(self.doclen*(la-mu)) # dim 10*1, conversion verified
        m1 = np.dot(expDoclen,reciSqrtIdf) # dim 10*19, product verified
        ## generate m2: matrix
        matrix = np.power(div,self.doc_term_matrix)/div
        ## Hadamard product
        matrix = np.multiply(matrix,m1)
        ## offset
        offset = np.dot(np.ones((matrix.shape[0],1)),reciSqrtIdf)
        ## matrix+offset
        matrix = matrix+offset
        ## reciprocal of recWt
        raw_prc_matrix = np.reciprocal(matrix)
        ## reset scores for the terms that do not occur
        label = (self.doc_term_matrix>0)
        self.mprc_matrix = np.multiply(label, raw_prc_matrix)

    def cal_MPRC1_similarity(self):
        '''MPRC 1 keeps word count k the same, but increase the number of words in common N'''
        self.sim_matrix = np.dot(self.mprc_matrix,self.mprc_matrix.T)
        ## get a ranked similar doc list
        scoreList = self.sim_matrix[0,:].tolist()[0]
        self.mprc_rankedHits = [pmid for (score,pmid) in sorted(zip(scoreList,self.pmidList),reverse=True)]
        if self.mprc_rankedHits[0]!= self.pmidList[0]:
            print "Similarity metric error: The most similar article to PMID %s is not itself."%self.pmidList[0],self.mprc_rankedHits[:10]
        ## save results
        outDir = self.resDir #os.path.join(self.baseDir,"mprc1_ranks")
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        outFile1 = os.path.join(outDir,self.pmidList[0]+".txt")
        fout = file(outFile1,"w")
        fout.write("\n".join(self.mprc_rankedHits))
        outFile2 = os.path.join(outDir,self.pmidList[0]+"_"+"score.pkl")
        fout = file(outFile2,"w")
        pmidwScore = sorted(zip(scoreList, self.pmidList),reverse=True)
        pickle.dump(pmidwScore,fout)

class MPRC_WT(MPRC):
    '''The similar terms are weighted according to their cosine distance to the original term. '''
    def __init__(self,pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir, knnterm_dir, model):
        super(MPRC_WT,self).__init__(pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir, knnterm_dir, model)

    def run_MPRC_WT(self):
        '''run the experiment to get MPRC_WT top hits'''
        self.getVocab()
        self.vectorizeText()
        self.buildDocFreq()
        self.getKNNterms()
        self.calMPRC_WTscores()
        self.cal_MPRC1_similarity()

    def calMPRCscores(self):
        '''Calculate Modified PRC score matrix'''
        ## count matrix 
        d1_vocab = np.where(self.doc_term_matrix[0,:]>0)[0].tolist() # d1_vocab is a list of index, not acutal terms
        curr_doclen = np.sum(self.doc_term_matrix[0,:]) # the length of the query text
        newMx  = self.doc_term_matrix # a numpy matrix
        for ind in d1_vocab:
            ori_t =self.vocab[ind]
            if ori_t in self.knnTermDict.keys():
                knn_t = self.knnTermDict[ori_t]
                knn_t = [t for t in knn_t if t in self.vocab]
                knn_index = [self.vocab.index(t) for t in knn_t]
                t_index = [ind]+knn_index
                subMx = self.doc_term_matrix[:,t_index] # The columns of subMx include the original terms in d0 and their similar terms
                newMx[1:,ind] = np.sum(subMx,axis=1)[1:] # Count the occurrence of the original term and its associated similar terms in all other documents
                newMx[:,ind] = newMx[:,ind]*(newMx[0,ind]/curr_doclen) # weight every term by the percentage of this term in the original text
        newMx = newMx[:,d1_vocab] # the new count matrix weighted by percentage
        self.doc_term_matrix = newMx
        
        ## MPRC
        la = 0.022
        mu = 0.013
        div = mu/la
        ## generate m1
        reciSqrtIdf = np.reciprocal(np.sqrt(np.log(len(self.stemmed_corpus)*2.0/(self.df+1)))) # dim 1*19, conversion verified
        reciSqrtIdf = reciSqrtIdf[0,d1_vocab]
        expDoclen = np.exp(self.doclen*(la-mu)) # dim 10*1, conversion verified
        m1 = np.dot(expDoclen,reciSqrtIdf) # dim 10*19, product verified
        ## generate m2: matrix
        matrix = np.power(div,self.doc_term_matrix)/div
        ## Hadamard product
        matrix = np.multiply(matrix,m1)
        ## offset
        offset = np.dot(np.ones((matrix.shape[0],1)),reciSqrtIdf)
        ## matrix+offset
        matrix = matrix+offset
        ## reciprocal of recWt
        raw_prc_matrix = np.reciprocal(matrix)
        ## reset scores for the terms that do not occur
        label = (self.doc_term_matrix>0)
        self.mprc_matrix = np.multiply(label, raw_prc_matrix)

class MPRC_Filter(PRC):
    '''This class filters out all the terms with PRC score under a threshold, i.e., only keep the most important terms.'''
    def __init__(self,pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir):
        super(MPRC_Filter,self).__init__(pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir)
        self.query = pairDict.keys()[0]
        self.mprc_rankedHits = []
        self.mprc_matrix = None # MPRC score matrix
    
    def run_MPRC_Filter(self):
        '''run the experiment to get PRC top hits'''
        self.getVocab()
        self.vectorizeText()
        self.buildDocFreq()
        self.calMPRC_Fileter_scores()
        self.cal_MPRC_Filter_Similarity()

    def calMPRC_Fileter_scores(self):
        '''Calculate the weight of every term per document, using PubMed Related Citation (PRC) algorithm, Jimmy Lin and John Wilbur 2007.
           input: idf vector, docLen vector, occurrence count matrix (n documents, all terms in the vocabulary)
           output: a matrix of PRC scores.
        '''
        la = 0.022
        mu = 0.013
        score_threshold = 0.5 # the PRC weight threshold 
        div = mu/la
        ## generate m1
        reciSqrtIdf = np.reciprocal(np.sqrt(np.log(len(self.stemmed_corpus)*2.0/(self.df+1)))) # dim 1*19, conversion verified
        expDoclen = np.exp(self.doclen*(la-mu)) # dim 10*1, conversion verified
        m1 = np.dot(expDoclen,reciSqrtIdf) # dim 10*19, product verified
        ## generate m2: matrix
        matrix = np.power(div,self.doc_term_matrix)/div
        ## Hadamard product
        matrix = np.multiply(matrix,m1)
        ## offset
        offset = np.dot(np.ones((matrix.shape[0],1)),reciSqrtIdf)
        ## matrix+offset
        matrix = matrix+offset
        ## reciprocal of recWt
        raw_prc_matrix = np.reciprocal(matrix)
        ## reset scores for the terms that do not occur
        label = (self.doc_term_matrix>0)
        self.prc_matrix = np.multiply(label, raw_prc_matrix)
        
        ## modify the score matrix, remove terms with low scores
        keyword_index_vec = np.where(self.prc_matrix.A[self.pmidList.index(self.query),:]>score_threshold)[0].tolist()
        self.prc_matrix = self.prc_matrix.A[:,keyword_index_vec]

    def cal_MPRC_Filter_Similarity(self):
        '''Measure the similarity between every pair of articles using PRC scores of terms in common between documents'''
        self.sim_matrix = np.dot(self.prc_matrix,self.prc_matrix.T)
        ## get a ranked similar doc list
        scoreList = self.sim_matrix[0,:].tolist()
        self.mprc_rankedHits = [pmid for (score,pmid) in sorted(zip(scoreList,self.pmidList),reverse=True)]
        if self.mprc_rankedHits[0]!= self.pmidList[0]:
            print "Most similar one is not itself: ",self.mprc_rankedHits[:10]
        ## save results
        outDir = self.resDir # os.path.join(self.baseDir,"prc_ranks","PorterStemmer")
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        outFile1 = os.path.join(outDir,self.pmidList[0]+".txt")
        fout = file(outFile1,"w")
        fout.write("\n".join(self.mprc_rankedHits))
        outFile2 = os.path.join(outDir,self.pmidList[0]+"_"+"score.pkl")
        fout = file(outFile2,"w")
        pmidwScore = sorted(zip(scoreList, self.pmidList),reverse=True)
        pickle.dump(pmidwScore,fout)    

def loadData(base_dir,analyzerType,sampleSize,bm25SimilarDoc):
    '''Load bm25 selected similar articles and the sample queries'''
    # Load 4584 articles and their top 100 similar articles selected by BM25
    bm25SimilarDocDict = pickle.load(file(bm25SimilarDoc))
    keyList = bm25SimilarDocDict.keys()[:sampleSize]
    sample = {}
    for key in keyList:
        sample[key] = bm25SimilarDocDict[key]
    return sample

def rerank(base_dir, database_dir,stemmed_corpus_dir, vocab_dir, prc_result_dir, mprc_result_dir, mprc_skg_result_dir, mprc_weighted_result_dir, mprc_filter_result_dir, knnterm_dir, sample, model):
    '''Re-rank similar articles using MPRC and PRC algorithms'''
    for query, similars in sample.iteritems():
        print query
        prc  = PRC({query:similars}, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, prc_result_dir)
        prc.run_PRC()
#         mprc = MPRC({query:similars}, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, mprc_result_dir, knnterm_dir, model)
#         mprc.run_MPRC()
#         mprc_skg = MPRC_SKG({query:similars}, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, mprc_skg_result_dir, knnterm_dir, model)
#         mprc_skg.run_MPRC_SKG()
#         mprc_wt = MPRC_WT({query:similars}, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, mprc_weighted_result_dir, knnterm_dir, model)
#         mprc_wt.run_MPRC_WT()        
#         mprc_filter  = MPRC_Filter({query:similars}, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, mprc_filter_result_dir)
#         mprc_filter.run_MPRC_Filter()

if __name__ == '__main__':
    # Parameters are in ParameterSetting.py
    
    # Load 4584 articles and their top 100 similar articles selected by BM25
    sample = loadData(base_dir,analyzerType,sampleSize, bm25SimilarDoc)
    
    # Prepare the text of 4584 articles
#     if not os.listdir(database_dir):
#         corpus = Corpus(base_dir, rawPMID_dir, medline_dir, database_dir)
#         corpus.run()

    # Re-rank BM25 retrieval results using MPRC and PRC
    rerank(base_dir, database_dir,stemmed_corpus_dir, vocab_dir, prc_result_dir, mprc_result_dir, mprc_skg_result_dir, mprc_weighted_result_dir, mprc_filter_result_dir,knnterm_dir, sample, model)
    print "Re-ranking completed.\n"
