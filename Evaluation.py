'''
This script evaluates the performance of PRC, MPRC, and MPRC_SKG, in terms of average precision, mean average precision, recall and F1-score.

Created on Sep 2, 2015
Last updated on Sep 2, 2015
@author: Wei Wei
'''

import os, pickle, random
from ParameterSetting import *
from pprint import pprint
import numpy as np
from RetKNN_MPRC import *

class Evaluation(object):
    '''Evaluate the performance of retrieval results'''
    def __init__(self,gsdir,mprcDir,prcDir,medline_dir,eval_dir,sampleSize): # database_dir contains individual MEDLINE files, one article per file
        self.gsdir = gsdir
        self.resDir = mprcDir # MPRC results
        self.prcDir = prcDir # PRC results
        self.medlineDir = medline_dir
#         self.evalFile = os.path.join(eval_dir,"%d_for_reviewer.txt"%sampleSize) # output for reviewers
#         self.evalOutFile = os.path.join(eval_dir,"%d_prec.txt"%sampleSize) # eval_doc
        self.rawFile = "/home/w2wei/Research/mesh/data/TREC/2005/genomics.qrels.large.txt"
        
        self.clusters={}
        self.cluster_index = {}
        self.tophits = {}
        self.PRCtophits = {}
        
    def run(self):
        self.getGoldStd() # 49 topics (50 claimed)
        self.loadMPRChits()
        self.loadPRChits()
        self.evalP5()
        self.evalPrec()
        self.evalRecall()
        self.evalFscore()
    
    def evalP5(self):
        '''Evaluate MAP@5'''
        mprc_map5 = 0
        prc_map5 = 0
        for k,v in self.tophits.iteritems(): # k is the query, v is a list of similar article PMIDs
            if self.tophits[k]==[]:
                continue
            ap_k = self.averPrec({k:v})
            mprc_map5+=ap_k
        print "MPRC MAP5 ", mprc_map5*1.0/len(self.tophits)
        for k,v in self.PRCtophits.iteritems(): # k is the query, v is a list of similar article PMIDs
            if self.PRCtophits[k]==[]:
                continue
            ap_k = self.averPrec({k:v})
            prc_map5+=ap_k
        print "PRC MAP5 ", prc_map5*1.0/len(self.PRCtophits)
    
    def averPrec(self, pairDict):
        '''Evaluate the average precision at 5 of an article'''
        query=pairDict.keys()[0]
        similars = pairDict[query] 
        relDocs = [self.clusters[c] for c in self.cluster_index[query]]
        relDocs = [item for sublist in relDocs for item in sublist]
        relDocs = set(relDocs)

        # MAP method 1 based on www.kaggle.com/wiki/MeanAveragePrecision
        weightedPrecSum = 0
        for k in xrange(len(similars)):
            if similars[k] in relDocs:
                preds = similars[:k+1] # top k predictions
                correct_preds = len(set(preds).intersection(set(relDocs)))
            else:
                correct_preds = 0
            p_k = correct_preds*1.0/(k+1)  # precision at cut-off k
            weightedPrecSum+=p_k
        aveP = weightedPrecSum/len(similars)
        return aveP

        # MAP method 2 based on wikipedia
#         weightedPrecSum = 0
#         for k in xrange(len(similars)):
#             preds = similars[:k+1] # top i predictions                
#             correct_preds = len(set(preds).intersection(set(relDocs)))
#             p_k = correct_preds*1.0/(k+1) # precision at cut-off k         
#             if similars[k] in relDocs:
#                 rel_k=1 # an indicator function, equal 1 if the item at rank k is a relevant doc
#             else:
#                 rel_k=0 # an indicator function, equal 0 if the item at rank k is NOT a relevant doc
#             weighted_prec_k = p_k*rel_k
#             weightedPrecSum += weighted_prec_k
#         AveP = weightedPrecSum*1.0/len(similars)
#         return AveP
        
    def evalPrec(self):
        '''Evaluate the precision of model outcomes'''
        mprc_ave_prec = 0
        prc_ave_prec = 0
#         fout = file(self.evalOutFile,"w")
        for k,v in self.tophits.iteritems():
#             print "Target doc: ",k
            if self.tophits[k]==[]:
                continue
            relDocs = [self.clusters[c] for c in self.cluster_index[k]]
            relDocs = [item for sublist in relDocs for item in sublist]
            relDocs = set(relDocs)        
#             print "MPRC predictions: ",v, "correct predictions: ", sorted(list(set(v).intersection(relDocs)))
#             print "PRC predictions: ",self.PRCtophits[k], "correct predictions: ", sorted(list(set(self.PRCtophits[k]).intersection(relDocs)))
#             print "MRPC precision: ", len(set(v).intersection(relDocs))*1.0/len(v)
#             print "PRC precision: ", len(set(self.PRCtophits[k]).intersection(relDocs))*1.0/len(v)
#             print
            content = "PMID\tMPRC\tPRC\n"
            content = k+"\t%.3f\t%.3f\n"%(len(set(v).intersection(relDocs))*1.0/len(v), len(set(self.PRCtophits[k]).intersection(relDocs))*1.0/len(v))
#             fout.write(content)
            mprc_ave_prec+=len(set(v).intersection(relDocs))*1.0/len(v)
            prc_ave_prec+=len(set(self.PRCtophits[k]).intersection(relDocs))*1.0/len(v)
        print "MPRC average precision ", mprc_ave_prec/len(self.tophits)
        print "PRC average precision ", prc_ave_prec/len(self.PRCtophits)
#         fout.write("MPRC average precision %.3f\nPRC average precision %.3f"%(mprc_ave_prec/len(self.tophits),prc_ave_prec/len(self.PRCtophits)))
        
    def evalRecall(self):
        '''Evaluate the average recall of model outcomes'''
        mprc_ave_recall=0
        prc_ave_recall=0
        for k,v in self.tophits.iteritems():
#             print "Query ",k
            if self.tophits[k]==[]:
                continue
            relDocs = [self.clusters[c] for c in self.cluster_index[k]]
            relDocs = [item for sublist in relDocs for item in sublist]
            relDocs = set(relDocs)
#             print "Rel doc num ", len(relDocs)            
#             print "MPRC predictions: ",v, "correct predictions: ", sorted(list(set(v).intersection(relDocs)))
#             print "PRC predictions: ",self.PRCtophits[k], "correct predictions: ", sorted(list(set(self.PRCtophits[k]).intersection(relDocs)))
#             print "MRPC recall: ", len(set(v).intersection(relDocs))*1.0/len(relDocs)
#             print "PRC recall: ", len(set(self.PRCtophits[k]).intersection(relDocs))*1.0/len(relDocs)
#             print
            mprc_ave_recall+=len(set(v).intersection(relDocs))*1.0/len(relDocs)
            prc_ave_recall+=len(set(self.PRCtophits[k]).intersection(relDocs))*1.0/len(relDocs)
        print "MPRC average recall ", mprc_ave_recall/len(self.tophits)
        print "PRC average recall ", prc_ave_recall/len(self.PRCtophits)
        
    def evalFscore(self):
        '''Evaluate the F score for every pair'''
        mprc_ave_F1 = 0
        prc_ave_F1 = 0
        for k,v in self.tophits.iteritems():
            if self.tophits[k]==[]:
                continue
            relDocs = [self.clusters[c] for c in self.cluster_index[k]]
            relDocs = [item for sublist in relDocs for item in sublist]
            relDocs = set(relDocs)        
            mprc_prec = len(set(v).intersection(relDocs))*1.0/len(v)
            prc_prec = len(set(self.PRCtophits[k]).intersection(relDocs))*1.0/len(v)
            mprc_recall = len(set(v).intersection(relDocs))*1.0/len(relDocs)
            prc_recall = len(set(self.PRCtophits[k]).intersection(relDocs))*1.0/len(relDocs)
            
            if mprc_prec+mprc_recall==0 or prc_prec+prc_recall==0:
                continue
            # MPRC F1
            mprc_F1 = 2.0*mprc_prec*mprc_recall/(mprc_prec+mprc_recall)
            # PRC F1
            prc_F1 = 2.0*prc_prec*prc_recall/(prc_prec+prc_recall)
            
            mprc_ave_F1 += mprc_F1
            prc_ave_F1 += prc_F1

        print "MPRC average F1 ", mprc_ave_F1/len(self.tophits)
        print "PRC average F1 ", prc_ave_F1/len(self.PRCtophits)
        
    def eval4review(self):
        '''Save the evaluation results for review'''
        gs = Corpus("", "", self.medlineDir, "")
        TiAbDict = gs.extractTiAb(os.path.join(self.medlineDir,"4584_medline.txt"))
        poolsize=20
        falseset = list(set(TiAbDict.keys())-set(self.tophits.keys())-set(self.PRCtophits.keys()))
        fout = file(self.evalFile,"w")
        for pmid,mprc_res in self.tophits.iteritems():
#             print "Target doc: ",pmid #, TiAbDict[pmid]
            fout.write("=====================\n")
            fout.write("Query: \n")
            fout.write("PMID: "+pmid+"\n")
            fout.write("Title: "+TiAbDict[pmid][0]+"\n")
            fout.write("Abstract: "+TiAbDict[pmid][1]+"\n")
            fout.write("---------------------\n")
            
            prc_res = self.PRCtophits[pmid]
            sim_res = list(set(prc_res).union(set(mprc_res)))
            ## randomly select false cases
            falsesize = poolsize - len(sim_res)
            falsecases = random.sample(falseset,falsesize)
            sim_res_pool  = sim_res + falsecases
            random.shuffle(sim_res_pool)
            for res in sim_res_pool:
#                 print "Similar doc: ",res, TiAbDict[res]
                fout.write("Similar article: \n")
                fout.write("PMID: "+res+"\n")
                fout.write("Title: "+TiAbDict[res][0]+"\n")
                fout.write("Abstract: "+TiAbDict[res][1]+"\n")
                fout.write("---------------------\n")
   
    def loadPRChits(self):
        '''Load PRC predictions'''
        docs = os.listdir(self.prcDir)
        docs = [doc for doc in docs if doc.endswith(".txt")]
        for doc in docs:
            target = doc.split(".txt")[0]         
            ranks = file(os.path.join(self.prcDir,doc)).read().split("\n")
            if target==ranks[0]: ## if the most similar one is the target article itself (supposed)
                self.PRCtophits[target] = ranks[1:6]
            else:
                self.PRCtophits[target] = ranks[:5]
            
    def loadMPRChits(self):
        '''Load MPRC predictions'''
        docs = os.listdir(self.resDir)
        docs = [doc for doc in docs if doc.endswith(".txt")]
        for doc in docs:
            target = doc.split(".txt")[0]
            ranks = file(os.path.join(self.resDir,doc)).read().split("\n")
            if target==ranks[0]: ## if the most similar one is the target article itself (supposed)
                self.tophits[target] = ranks[1:6]
            else:
                self.tophits[target] = ranks[:5]

    def buildGoldStd(self):
        '''Build a gold standard dictionary'''
        clusters = {}
        for line in file(self.rawFile):
            cat,tmp,pmid,label = line.split("\t")
            label = label[:-1]
#             if label!=0: # both possible relevant and definite relevant are counted
            if label>1: # only definite relevant is counted
                if cat not in clusters.keys():
                    clusters[cat]=[pmid]
                else:
                    clusters[cat].append(pmid)
        for cat in clusters.keys():
            clusters[cat] = list(set(clusters[cat]))
        cluster_index = {}
        for k,pmids in clusters.iteritems():
            for p in pmids:
                if p not in cluster_index.keys():
                    cluster_index[p] = [k]
                else:
                    cluster_index[p].append(k)
        for pmid in cluster_index.keys():
            cluster_index[pmid]=list(set(cluster_index[pmid]))
        
        self.clusters = clusters
        self.cluster_index = cluster_index
        fout1 = file(os.path.join(self.gsdir,"clusters.pkl"),'w')
        pickle.dump(clusters,fout1)
        fout2 = file(os.path.join(self.gsdir,"cluster_index.pkl"),"w")
        pickle.dump(cluster_index,fout2)
    
    def getGoldStd(self):
        '''Load the a gold standard'''
        try:
            self.clusters = pickle.load(file(os.path.join(self.gsdir,"clusters.pkl")))
            self.cluster_index = pickle.load(file(os.path.join(self.gsdir,"cluster_index.pkl")))
        except:
            self.buildGoldStd()

class CompMixVsPRC(object):
    '''Compare the mixture model (LR+PRC+MPRC) and PRC'''
    def __init__(self,classifier_data_dir,base_dir, database_dir, stemmed_corpus_dir, vocab_dir, medline_dir, mprc_eval_dir, knnterm_dir, goldstd_dir, sampleSize, model):
        self.base_dir = base_dir
        self.database_dir = database_dir
        self.stemmed_corpus_dir = stemmed_corpus_dir
        self.vocab_dir = vocab_dir
        self.medline_dir = medline_dir
        self.mprc_eval_dir = mprc_eval_dir
        self.knnterm_dir = knnterm_dir
        self.goldstd_dir = goldstd_dir
        self.wkdir = classifier_data_dir
        self.prc_res_dir = os.path.join(self.wkdir,"prc")
        self.mix_res_dir = os.path.join(self.wkdir,"mix")
        if not os.path.exists(self.prc_res_dir):
            os.mkdir(self.prc_res_dir)
        if not os.path.exists(self.mix_res_dir):
            os.mkdir(self.mix_res_dir)
        
        self.pmidList = []
        self.prcList = []
        self.mprcList = []
        self.sampleSize = sampleSize
        self.model = model
        print "analyzer type ", analyzerType
        print "bm25 ",bm25SimilarDoc
    
    def run(self):
        # load LR predictions
        lr_preds = np.load(file(os.path.join(self.wkdir,"lr_predictions.npy")))
        all_pmids = pickle.load(file(os.path.join(self.wkdir,"all_pmids.pkl")))
        self.pmidList = all_pmids[len(all_pmids)-len(lr_preds):]
        # split pmidList
        self.splitPMIDs(lr_preds)
        # run all the PMIDs using PRC (baseline)
        if not os.listdir(self.prc_res_dir):
            self.run_all_samples_with_PRC()
#         run MPRC or PRC according to LR predictions
        if not os.listdir(self.mix_res_dir):
            self.run_mix()
        # evaluate PRC performance
        self.eval()
    
    def eval(self):
        eva = Evaluation(self.goldstd_dir,self.mix_res_dir, self.prc_res_dir, self.medline_dir, self.mprc_eval_dir,self.sampleSize)
        eva.run()        
            
    def run_mix(self):
        '''Run prc and mprc according to LR predictions'''
        sample = loadData(self.base_dir,analyzerType, self.sampleSize, bm25SimilarDoc)
        for query, similars in sample.iteritems():
            if query in self.prcList:
                fin = file(os.path.join(prc_result_dir,query+'.txt'))
                text = fin.read()
                fout = file(os.path.join(self.mix_res_dir,query+'.txt'),"w")
                fout.write(text)
#                 prc  = PRC({query:similars}, self.base_dir, self.database_dir, self.stemmed_corpus_dir, self.vocab_dir, self.mix_res_dir)
#                 prc.run_PRC()
            if query in self.mprcList:
                fin = file(os.path.join(mprc_result_dir,query+'.txt'))
                text = fin.read()
                fout = file(os.path.join(self.mix_res_dir,query+'.txt'),"w")
                fout.write(text)
#                 mprc = MPRC({query:similars}, self.base_dir, self.database_dir, self.stemmed_corpus_dir, self.vocab_dir, self.mix_res_dir, self.knnterm_dir, self.model)
#                 mprc.run_MPRC()

    
    def run_all_samples_with_PRC(self):
        '''Run all the samples using PRC and evaluate its MAP'''
        sample = loadData(base_dir,analyzerType,sampleSize, bm25SimilarDoc)
        for query, similars in sample.iteritems():
            if query not in self.pmidList:
                continue
            fin = file(os.path.join(prc_result_dir,query+'.txt'))
            text = fin.read()
            fout = file(os.path.join(self.prc_res_dir,query+'.txt'),"w")
            fout.write(text)
#             prc  = PRC({query:similars}, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, self.prc_res_dir)
#             prc.run_PRC()

    def splitPMIDs(self,lr_preds):
        '''Split self.pmids according to the LR prediction'''
        for i in xrange(len(self.pmidList)):
            if lr_preds[i]==1:
                self.mprcList.append(self.pmidList[i])
            else:
                self.prcList.append(self.pmidList[i])
        
    
if __name__ == '__main__':
    # Parameters are in ParameterSetting.py
#     eva = Evaluation(goldstd_dir,mprc_filter_result_dir, prc_result_dir, medline_dir, mprc_filter_eval_dir,sampleSize)
    eva = Evaluation(goldstd_dir,mprc_result_dir, prc_result_dir, medline_dir, mprc_eval_dir,sampleSize)
#     eva.run()
    cmp = CompMixVsPRC(classifier_data_dir,base_dir, database_dir, stemmed_corpus_dir, vocab_dir, medline_dir, mprc_eval_dir, knnterm_dir, goldstd_dir, sampleSize, model)
    cmp.run()
