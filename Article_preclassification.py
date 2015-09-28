'''
Classify articles into two categories: PRC good performance and PRC poor performance. 
Every article will be represented by the weights on its stemmed vocabulary. 
Start with LR

Created on Sep 8, 2015

@author: Wei Wei
'''

from ParameterSetting import *
import numpy as np
from pprint import pprint
import pickle, os, time
from RetKNN_PRC_analysis import CompMPRCvsPRC
import pickle
from RetKNN_MPRC import PRC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, cross_validation
from Evaluation import Evaluation
from RetKNN_PRC_analysis import Analysis
from RetKNN_MPRC import MPRC
from scipy import stats

class Label(CompMPRCvsPRC):
    '''If PRC achieves 100% MAP on an article, this article is labeled with 0, otherwise it is labeled with 1
       Prepare training and test data in numpy array format.'''
    def __init__(self,gsdir,mprcDir,prcDir,medline_dir,eval_dir,sampleSize, classifier_data_dir):
        super(Label,self).__init__(gsdir,mprcDir,prcDir,medline_dir,eval_dir,sampleSize,classifier_data_dir)
           
    def run(self):
        self.getGoldStd() # 49 topics (50 claimed)
        self.loadPRChits()
        self.evalMAP()
        self.selectLowAvePrecArticles(threshold=1.0) # self.low are the positive examples; 4236 docs, 1215 positive docs, 3021 negative docs
        self.save()
    
    def save(self):
        outFile = os.path.join(self.outDir,"1215_positive_pmids.pkl")
        pickle.dump(self.low,file(outFile,"w"))
        
class Data(PRC):
    '''Format data in numpy array'''
    def __init__(self, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir, classifier_data_dir):
        super(Data,self).__init__({"void":0}, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir)
        self.stemmed_corpus_dir = stemmed_corpus_dir
        self.resDir = classifier_data_dir
        self.vocabFile = os.path.join(self.resDir,"vocab.pkl")
        self.corpusFile = os.path.join(self.resDir,"corpus.pkl")
        self.dataFile = os.path.join(self.resDir,"data.npy")
        self.targetFile = os.path.join(self.resDir,"target.npy")
        if not os.path.exists(self.resDir):
            os.mkdir(self.resDir)
        
        self.pmidList = []
        self.vocab = []
        self.stemmed_corpus = []
        self.target = None # label array for all the articles
    
    def run(self):
        self.getPMID()
        self.getVocabAndCorpus()
        self.vectorizeText()
        self.buildDocFreq()
        self.calPRCscores()        
        self.buildTargets()
        self.save()
  
    def getPMID(self):
        for doc in os.listdir(self.stemmed_corpus_dir):
            self.pmidList.append(doc)
        pickle.dump(self.pmidList,file(os.path.join(self.resDir,"all_pmids.pkl"),"w"))
    
    def getVocabAndCorpus(self):
        try:
            self.vocab = pickle.load(file(self.vocabFile))
            self.stemmed_corpus = pickle.load(file(self.corpusFile))
        except:
            self.buildVocabAndCorpus()
    
    def buildVocabAndCorpus(self):
        for pmid in self.pmidList: # self.pmidList includes the query and the 99 most similar articles selected by BM25
            self.stemmed_corpus.append(pickle.load(file(os.path.join(self.stemmed_corpus_dir,pmid)))[0]) # corpus contains raw text (MH, title*2, abstract)
        for doc in self.stemmed_corpus:
            sent = doc.split(". ")
            vocabList = [s.split(" ") for s in sent]
            vocab = [item for sublist in vocabList for item in sublist]
            self.vocab+=vocab
            self.vocab = list(set(self.vocab))
        pickle.dump(self.stemmed_corpus,file(self.corpusFile,"w"))
        pickle.dump(self.vocab,file(self.vocabFile,"w"))

    def buildTargets(self):
        '''Assign labels to every article'''
        pos_pmids = pickle.load(file(os.path.join(self.resDir,"1215_positive_pmids.pkl")))
        target = []
        for pmid in self.pmidList:
            if pmid in pos_pmids:
                target.append(1)
            else:
                target.append(0)
        self.target = np.array(target)                
        
    def save(self):
        np.save(file(self.dataFile,"w"),self.prc_matrix)
        np.save(file(self.targetFile,"w"), self.target)
    
class Classifier(object):
    '''Classify articles using different models'''
    def __init__(self,classifier_data_dir):
        self.wkdir = classifier_data_dir
        self.data = None
        self.target = None
        self.train_data = None
        self_train_target = None
        self.test_data = None
        self.test_target = None
    
    def run(self):
        self.loadData()
        self.logistic_regression()
#         self.logistic_regression_evaluation()
    
    def logistic_regression(self):
        '''Train a LR on half the data, and make predictions on the remaining data.'''
        lr = linear_model.LogisticRegression()
        self.train_data = self.data[:0.5*len(self.target),:]
        self.train_target = self.target[:0.5*len(self.target)]
        self.test_data = self.data[0.5*len(self.target):,:]
        self.test_target = self.target[0.5*len(self.target):]
        lr = lr.fit(self.train_data,self.train_target)
        scores = lr.score(self.test_data, self.test_target)
        preds = lr.predict(self.test_data)
        np.save(file(os.path.join(self.wkdir,"lr_predictions.npy"),"w"),preds)
    
    def logistic_regression_evaluation(self):
        '''Using cross validation'''
        lr = linear_model.LogisticRegression()
        accuracyList = cross_validation.cross_val_score(lr, self.data, self.target, scoring='accuracy', cv=10, n_jobs=4)
        print "Accuracy: ", accuracyList.mean(),accuracyList.std()*2
        precisionList = cross_validation.cross_val_score(lr, self.data, self.target, scoring='precision', cv=10, n_jobs=4)
        print "Precision: ", precisionList.mean(),precisionList.std()*2
        recallList = cross_validation.cross_val_score(lr, self.data, self.target, scoring='recall', cv=10, n_jobs=4)
        print "Recall: ", recallList.mean(),recallList.std()*2
        f1List = cross_validation.cross_val_score(lr, self.data, self.target, scoring='f1', cv=10, n_jobs=4)
        print "F1: ", f1List.mean(),f1List.std()*2
        aucList = cross_validation.cross_val_score(lr, self.data, self.target, scoring='roc_auc', cv=10, n_jobs=4)
        print "AUC: ", aucList.mean(),aucList.std()*2
        
    def loadData(self):
        self.data = np.load(file(os.path.join(self.wkdir,"data.npy")))
        self.target = np.load(file(os.path.join(self.wkdir,"target.npy")))
#         self.train_data = self.data[:0.9*len(self.target),:]
#         self.train_target = self.target[:0.9*len(self.target)]
#         self.test_data = self.data[0.9*len(self.target):,:]
#         self.test_target = self.target[0.9*len(self.target):]

def load_KL_Data(goldstd_dir,mprc_result_dir, prc_result_dir, medline_dir, mprc_eval_dir,sampleSize):
    '''Load PRC top hits and MPRC top hits'''
    eval = Evaluation(goldstd_dir,mprc_result_dir, prc_result_dir, medline_dir, mprc_eval_dir,sampleSize)
    eval.loadMPRChits()
    eval.loadPRChits()
    prc_tophits = eval.PRCtophits
    mprc_tophits = eval.tophits
    sample = {}
    for key in prc_tophits.keys():
        sample[key] = [mprc_tophits[key],prc_tophits[key]]
    return sample

def run_kl_classifier(kl_data,base_dir, database_dir, kl_stemmed_corpus_dir, kl_vocab_dir, mprc_result_dir, prc_result_dir, knnterm_dir, model):
    if not os.path.exists(kl_vocab_dir):
        os.makedirs(kl_vocab_dir)
    if not os.path.exists(kl_stemmed_corpus_dir):
        os.makedirs(kl_stemmed_corpus_dir)
    for pmid in kl_data.keys():
        mprc_tophits, prc_tophits = kl_data[pmid]
        klc = KL_classifier({pmid:[pmid]+mprc_tophits},base_dir, database_dir, kl_stemmed_corpus_dir, kl_vocab_dir, mprc_result_dir, knnterm_dir, model)
        klc.run()
        raw_input("wait...")

class KL_classifier(MPRC):
    '''Merge PRC top hits and MPRC top hits. Select the top 5 according to the KL distance to the query'''
    def __init__(self,pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir, knnterm_dir, model):
        super(KL_classifier,self).__init__(pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, result_dir, knnterm_dir, model)
        self.mprc_klDict = {} # {similar PMID: KL distance}
        self.prc_klDict = {}
        
    def run(self):
        print self.pmidList[0]
        self.run_MPRC()
#         self.run_PRC()
        
    def run_MPRC(self):
        self.getVocab()
        self.vectorizeText()
        self.buildDocFreq()
        self.getKNNterms()
        self.calMPRCscores()
        self.cal_MPRC_KL()
        print "MPRC: ", self.mprc_klDict
    
    def cal_MPRC_KL(self):
        '''calculate the KL distance between similar articles and the query'''
        query_non_zero_cols = np.where(self.mprc_matrix[0,:]>0)[1].tolist()[0]
        print query_non_zero_cols
        self.prc_matrix = self.mprc_matrix[:,query_non_zero_cols]        
        query = self.mprc_matrix[0,:]
        print "query"
        print query
        for i in range(1,6):
            sim = self.mprc_matrix[i,:]
            matched_query_term_wts = query[sim>0]
            matched_similar_term_wts = sim[sim>0]
#             print matched_query_term_wts
#             print len(matched_query_term_wts.tolist()[0]) 
#             print matched_similar_term_wts
#             print len(matched_similar_term_wts.tolist()[0])            
            norm_matched_query_term_wts = matched_query_term_wts/np.sum(matched_query_term_wts)
            norm_matched_similar_term_wts = matched_similar_term_wts/np.sum(matched_similar_term_wts)
            print "MPRC term count ", len(norm_matched_query_term_wts.tolist()[0])
            print "norm_matched_query_term_wts: ",np.max(norm_matched_query_term_wts)
            print "norm_matched_similar_term_wts: ",np.max(norm_matched_similar_term_wts)
            print "KL: ", stats.entropy(norm_matched_similar_term_wts.tolist()[0], norm_matched_query_term_wts.tolist()[0])
            self.mprc_klDict[self.pmidList[i]]=stats.entropy(norm_matched_similar_term_wts.tolist()[0], norm_matched_query_term_wts.tolist()[0]) # KL distance
            
    def run_PRC(self):
        self.getVocab()
        self.vectorizeText()
        self.buildDocFreq()
        self.calPRCscores()
        self.cal_PRC_Similarity()
        self.cal_PRC_KL()
        print "PRC: ",self.prc_klDict
            
    def cal_PRC_KL(self):
        # remove columns that query term weight is 0
        query_non_zero_cols = np.where(self.prc_matrix[0,:]>0)[1].tolist()[0]
        self.prc_matrix = self.prc_matrix[:,query_non_zero_cols]
        query = self.prc_matrix[0,:]
        print "query"
        print query
        for i in range(1,6):
            sim = self.prc_matrix[i,:]
            matched_query_term_wts = query[sim>0]
            matched_similar_term_wts = sim[sim>0]
#             print matched_query_term_wts
#             print len(matched_query_term_wts.tolist()[0]) 
#             print matched_similar_term_wts
#             print len(matched_similar_term_wts.tolist()[0])
            norm_matched_query_term_wts = matched_query_term_wts/np.sum(matched_query_term_wts)
            norm_matched_similar_term_wts = matched_similar_term_wts/np.sum(matched_similar_term_wts)
            print "PRC term count ", len(norm_matched_query_term_wts.tolist()[0])
            print "norm_matched_query_term_wts: ",np.max(norm_matched_query_term_wts)
            print norm_matched_query_term_wts
            print "norm_matched_similar_term_wts: ",np.max(norm_matched_similar_term_wts)
            print norm_matched_similar_term_wts
            print "KL: ", stats.entropy(norm_matched_similar_term_wts.tolist()[0], norm_matched_query_term_wts.tolist()[0])
            self.prc_klDict[self.pmidList[i]]=stats.entropy(norm_matched_similar_term_wts.tolist()[0], norm_matched_query_term_wts.tolist()[0]) # KL distance

if __name__ == '__main__':
    # prepare labels, determine positive examples and negative examples
    label = Label(goldstd_dir,mprc_result_dir, prc_result_dir, medline_dir, mprc_eval_dir,sampleSize,prc_summary_dir)
#     label.run()
    data = Data(base_dir, database_dir, stemmed_corpus_dir, vocab_dir, prc_summary_dir, classifier_data_dir)
#     data.run()
    
    # run logistic regression
    classifier = Classifier(classifier_data_dir)
#     classifier.run()
    
    # run kl-divergence based Classification
    kl_data = load_KL_Data(goldstd_dir,mprc_result_dir, prc_result_dir, medline_dir, mprc_eval_dir,sampleSize)
    run_kl_classifier(kl_data,base_dir, database_dir, kl_stemmed_corpus_dir, kl_vocab_dir, mprc_result_dir, prc_result_dir, knnterm_dir, model)
    