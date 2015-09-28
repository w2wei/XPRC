'''
Analyze the model outcomes and provide an interpretation.
Created on Sep 2, 2015
Last updated on Sep 2, 2015
@author: Wei Wei
'''


from ParameterSetting import *
from RetKNN_MPRC import *
from Evaluation import Evaluation
from pprint import pprint

# from TREC2005.Index_TREC2005Genomics4584 import Corpus
Entrez.email = "granitedewint@gmail.com"


class SimilarityAnalysis(MPRC):
    '''Interpret the similarity between two documents and identify matched terms and associated scores.'''
    def __init__(self,pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, knnterm_dir, model, goldstd_dir,mprc_result_dir,prc_result_dir,medline_dir,interpret_dir, sampleSize): 
        super(SimilarityAnalysis,self).__init__(pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, mprc_result_dir, knnterm_dir, model)
        self.query = pairDict.keys()[0] # current query PMID
        self.interpret_dir = interpret_dir
        self.interpret_file = os.path.join(interpret_dir,"%s"%self.query)
        self.eval = Evaluation(goldstd_dir,mprc_result_dir,prc_result_dir,medline_dir,self.interpret_file,sampleSize)
        self.output = {}
        self.pklout = {}
        self.knnTermDict = {}
        
    def run_MPRC_SKG(self):
        # get original PRC weights
        self.getVocab() # the vocabulary of the articles in pairDict
        self.vectorizeText()
#         self.getKNNterms()
        self.buildDocFreq() # get the document frequency for every word in the vocabulary
        self.calPRCscores() # calculate the weights
#         self.cal_PRC_Similarity() # calculate the similarity
        orig_wtMatrix = self.prc_matrix # the weight matrix from PRC
        self.adjustWeights()
        self.buildDocFreq() # get the document frequency for every word in the vocabulary
        self.calPRCscores() # calculate the weights
        skg_wtMatrix = self.prc_matrix # the weight matrix from MPRC_SKG
        # print the precision on this query
        self.eval.loadMPRChits()
        self.analzeResults_mprc_skg(orig_wtMatrix, skg_wtMatrix)
#         self.analyzeResults_mprc()
#         self.saveOutput()
        
    def run_PRC(self):
        # get original PRC weights
        self.getVocab() # the vocabulary of the articles in pairDict
        self.vectorizeText()
        self.buildDocFreq() # get the document frequency for every word in the vocabulary
        self.calPRCscores() # calculate the weights
        orig_wtMatrix = self.prc_matrix # the weight matrix from PRC
        # print the precision on this query
        self.eval.loadPRChits()
        self.analyzeResults_prc(orig_wtMatrix)
        self.saveOutput()
    
    def run_MPRC(self):
        '''Compare the difference between PRC's selection and MRPC's selections in terms of matched terms.'''
        self.getVocab() # the vocabulary of the articles in pairDict, 100 articles in the corpus, pmidList size 100
        self.vectorizeText()
        orig_doc_term_matrix = self.doc_term_matrix
        self.buildDocFreq() # get the document frequency for every word in the vocabulary
        self.getKNNterms()
        self.calMPRCscores() # calculate the weights
        self.eval.loadMPRChits()
        self.analyzeResults_mprc(orig_doc_term_matrix)   
        self.saveOutput()     
 
    def analyzeResults_prc(self, orig_wtMatrix):
        summary = ''
        if self.query not in self.eval.PRCtophits.keys():
            print "This query %s does not exist in pre-calculated PRC top hits."%self.query
            return
        for similar in self.eval.PRCtophits[self.query]: # PRC selected similar articles
            matchTermScoreDict = self.analyzeEachPair_prc(similar, orig_wtMatrix)
            self.pklout[similar] = (matchTermScoreDict)
            # output this pair of articles, their matched terms and weight changes
            summary += "Current pair: %s - %s\n" %(self.query, similar)
            for k,v in matchTermScoreDict.iteritems():
                summary += "%s: %s\n"%(k,str(v))
        if self.query not in self.output.keys():
            self.output[self.query] = [summary]
        else:
            self.output[self.query].append(summary)  
    
    def analyzeEachPair_prc(self, similar, orig_wtMatrix):
        '''Analyze PRC outputs'''
        query_vocab_index = np.where(self.doc_term_matrix[self.pmidList.index(self.query),:]>0)[0].tolist() # query_vocab is a list of index, not acutal terms
        # get the vocabulary of the similar text
        similar_vocab_index = np.where(self.doc_term_matrix[self.pmidList.index(similar),:]>0)[0].tolist() # similar article vocabulary indices
        similar_vocab = [self.vocab[index] for index in similar_vocab_index]
        match = {}
        # matched terms in the similar article
        for index in query_vocab_index:
            ori_term = self.vocab[index]
            match[ori_term]=[] # initialize match term dictionary            
        # term weights in the query
        for term in match.keys():
            if term in similar_vocab:
                query_orig_wt = orig_wtMatrix[0,self.vocab.index(term)]
                similar_orig_wt = orig_wtMatrix[self.pmidList.index(similar),self.vocab.index(term)]
                match[term] = [query_orig_wt, similar_orig_wt]
            else:
                query_orig_wt = orig_wtMatrix[0,self.vocab.index(term)]
                match[term] = [query_orig_wt, 0] # 0 means the similar article does not contain this term         
        return match

    def analyzeResults_mprc(self,orig_doc_term_matrix):
        '''Extract every pair of articles and call the analyzer function'''
        for similar in self.eval.tophits[self.query]: # MPRC selected similar articles
            if similar not in self.pmidList: # if MPRC's selection is not in the original BM25 top 100 selection. this should not happen.
                continue
            self.analyzeEachPair_mprc(similar,orig_doc_term_matrix)
        
    def analyzeEachPair_mprc(self,similar,orig_doc_term_matrix):
        '''Analyze a pair of query text and model predicted similar text''' 
        # get the vocabulary of the query text
        query_vocab_index = np.where(orig_doc_term_matrix[self.pmidList.index(self.query),:]>0)[0].tolist() # query_vocab is a list of index, not acutal terms
        query_vocab = [self.vocab[index] for index in query_vocab_index]
        # get the vocabulary of the similar text
        similar_vocab_index = np.where(orig_doc_term_matrix[self.pmidList.index(similar),:]>0)[0].tolist() # query_vocab is a list of index, not acutal terms
        similar_vocab = [self.vocab[index] for index in similar_vocab_index]
        match = {}
        # get the expanded vocabulary of the query text and the matched terms in the similar article
        for index in query_vocab_index:
            ori_term = self.vocab[index]
            overlap=[]
            if ori_term in self.knnTermDict.keys():
                knn_termList = self.knnTermDict[ori_term]
                knn_termList = [t for t in knn_termList if t in self.vocab]
                overlap = list(set([ori_term]+knn_termList).intersection(set(similar_vocab)))
            else:
                overlap = list(set([ori_term]).intersection(set(similar_vocab)))
            if overlap:
                match[ori_term] = overlap
        # output the summary of matched terms
        summary = "Current pair: %s - %s\n" %(self.query, similar)
        summary += "Word count of the query %s: %d\n"%(self.query,np.sum(self.doc_term_matrix[self.pmidList.index(self.query),:]))
        summary += "Word count of the similar article %s: %d\n"%(similar,np.sum(self.doc_term_matrix[self.pmidList.index(similar),:]))
        for k,v in match.iteritems():
            summary += "%s: %s\n"%(k,";".join(v))
        summary += "\n"
        if self.query not in self.output.keys():
            self.output[self.query] = [summary]
        else:
            self.output[self.query].append(summary)
        
    def analzeResults_mprc_skg(self, orig_wtMatrix, skg_wtMatrix):
        '''Extract every pair of articles and call the analyzer function'''
        summary = ''
        for similar in self.eval.tophits[self.query]: # MPRC_SKG selected similar articles
            matchTermScoreDict = self.analyzeEachPair_mprc_skg(similar, orig_wtMatrix, skg_wtMatrix)
            # output this pair of articles, their matched terms and weight changes
            summary += "Current pair: %s - %s\n" %(self.query, similar)
            for k,v in matchTermScoreDict.iteritems():
                summary += "%s: %s\n"%(k,str(v))
                summary += "\n"
        if self.query not in self.output.keys():
            self.output[self.query] = [summary]
        else:
            self.output[self.query].append(summary)
            
    def analyzeEachPair_mprc_skg(self, similar, orig_wtMatrix, skg_wtMatrix):
        '''Analyze MPRC_SKG outputs'''
        query_vocab_index = np.where(self.doc_term_matrix[self.pmidList.index(self.query),:]>0)[0].tolist() # query_vocab is a list of index, not acutal terms
#         query_vocab = [self.vocab[index] for index in query_vocab_index]
        # get the vocabulary of the similar text
        similar_vocab_index = np.where(self.doc_term_matrix[self.pmidList.index(similar),:]>0)[0].tolist() # query_vocab is a list of index, not acutal terms
        similar_vocab = [self.vocab[index] for index in similar_vocab_index]
        match = {}
        # matched terms in the similar article
        for index in query_vocab_index:
            ori_term = self.vocab[index]
            if ori_term in similar_vocab:
                match[ori_term]=[] # initialize match term dictionary
        # term weights in the query
        for term in match.keys():
            query_orig_wt = orig_wtMatrix[0,self.vocab.index(term)]
            query_new_wt = skg_wtMatrix[0,self.vocab.index(term)]
            similar_orig_wt = orig_wtMatrix[self.pmidList.index(similar),self.vocab.index(term)]
            similar_new_wt = skg_wtMatrix[self.pmidList.index(similar),self.vocab.index(term)]
            match[term] = [query_new_wt/query_orig_wt,similar_new_wt/similar_orig_wt]
        return match
                    
    def saveOutput(self):
        fout = file(self.interpret_file,"w")
        for summary  in self.output.values():
            for s in summary:
                fout.write(s)
        pklFile = self.interpret_file+".pkl"
        pickle.dump(self.pklout,file(pklFile,"w"))        

class Interpretation(SimilarityAnalysis):
    def __init__(self,pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, knnterm_dir, model, goldstd_dir,mprc_result_dir,prc_result_dir,medline_dir,interpret_dir, sampleSize):
        super(Interpretation,self).__init__(pairDict, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, knnterm_dir, model, goldstd_dir,mprc_result_dir,prc_result_dir,medline_dir,interpret_dir, sampleSize)
        
    def run_MPRC(self):
        # get original PRC weights
        self.getVocab() # the vocabulary of the articles in pairDict
        self.vectorizeText()
#         self.getKNNterms()
        self.buildDocFreq() # get the document frequency for every word in the vocabulary
        self.calPRCscores() # calculate the weights
#         self.cal_PRC_Similarity() # calculate the similarity
        orig_wtMatrix = self.prc_matrix # the weight matrix from PRC
        self.adjustWeights()
        self.buildDocFreq() # get the document frequency for every word in the vocabulary
        self.calPRCscores() # calculate the weights
        skg_wtMatrix = self.prc_matrix # the weight matrix from MPRC_SKG
        # print the precision on this query
        self.eval.loadMPRChits()
        self.analzeResults_mprc_skg(orig_wtMatrix, skg_wtMatrix)
#         self.analyzeResults_mprc()        

class MPRCresultAnalysis(Evaluation):
    '''This class outputs a summary of MPRC and PRC performance'''
    def __init__(self,gsdir,mprcDir,prcDir,medline_dir,eval_dir,sampleSize, interpret_dir):
        super(MPRCresultAnalysis,self).__init__(gsdir,mprcDir,prcDir,medline_dir,eval_dir,sampleSize)
        if not os.path.exists(interpret_dir):
            os.mkdir(interpret_dir)
        self.outFile = os.path.join(interpret_dir,str(sampleSize)+".txt")
        self.output = {}
        self.mprc_correct_prc_missed = []
        self.mprc_prc_both_correct = []
        self.prc_correct_mprc_missed = []
        self.mprc_incorrect = []
        self.prc_incorrect = []
        
    def run(self):
        self.getGoldStd() # 49 topics (50 claimed)
        self.loadMPRChits()
        self.loadPRChits()
        self.summarizeMPRCoutcomes()

    def summarizeMPRCoutcomes(self):
        '''Classify MPRC prediction into four categories'''
        for k in self.tophits.keys():
            if self.tophits[k]==[]:
                continue
            gsd = [self.clusters[c] for c in self.cluster_index[k]]
            gsd = [item for sublist in gsd for item in sublist]
            gsd = set(gsd)        
            mprc_outcome = set(self.tophits[k]) 
            prc_outcome = set(self.PRCtophits[k])
            self.output[k] = k+'\n' # initialize output[k]
            # MPRC's correct prediction, missed by PRC
            mprc_correct = mprc_outcome.intersection(gsd)
            mprc_correct_prc_missed = mprc_correct-prc_outcome
            self.mprc_correct_prc_missed+=list(mprc_correct_prc_missed)
            self.output[k]+="MPRC's correct prediction, but missed by PRC:\tCount: %d\t%s\n"%(len(mprc_correct_prc_missed), ",".join(list(mprc_correct_prc_missed)))
            # Both MPRC and PRC's correct prediction
            mprc_prc_both_correct = mprc_correct.intersection(prc_outcome)
            self.mprc_prc_both_correct += list(mprc_prc_both_correct)
            self.output[k] += "MPRC and PRC's correct prediction:            \tCount: %d\t%s\n"%(len(mprc_prc_both_correct), ",".join(list(mprc_prc_both_correct)))
            # PRC's correct prediction, missed by MPRC
            prc_correct = prc_outcome.intersection(gsd)
            prc_correct_mprc_missed = prc_correct-mprc_outcome
            self.prc_correct_mprc_missed += prc_correct_mprc_missed
            self.output[k] += "PRC's correct prediction, but missed by MPRC:\tCount: %d\t%s\n"%(len(prc_correct_mprc_missed), ",".join(list(prc_correct_mprc_missed)))
            # MPRC's incorrect prediction
            mprc_incorrect = mprc_outcome-gsd
            self.mprc_incorrect += list(mprc_incorrect)
            self.output[k] += "MPRC's incorrect prediction:                  \tCount: %d\t%s\n"%(len(mprc_incorrect), ",".join(list(mprc_incorrect)))
            # PRC's incorrect prediction
            prc_incorrect = prc_outcome-gsd
            self.prc_incorrect += prc_incorrect
            self.output[k] += "PRC's incorrect prediction:                   \tCount: %d\t%s\n"%(len(prc_incorrect), ",".join(list(prc_incorrect)))
            self.output[k] += "\n"
        # record article by category
        self.mprc_correct_prc_missed = list(set(self.mprc_correct_prc_missed))
        self.mprc_prc_both_correct = list(set(self.mprc_prc_both_correct))
        self.prc_correct_mprc_missed = list(set(self.prc_correct_mprc_missed))
        self.mprc_incorrect = list(set(self.mprc_incorrect))   
        self.prc_incorrect = list(set(self.prc_incorrect))
        # output
        print "mprc_correct_prc_missed ", len(self.mprc_correct_prc_missed)
        print "mprc_prc_both_correct ",len(self.mprc_prc_both_correct)
        print "prc_correct_mprc_missed ",len(self.prc_correct_mprc_missed)
        print "mprc_incorrect ", len(self.mprc_incorrect)
        print "prc_incorrect ",len(self.prc_incorrect)
        
        fout = file(self.outFile,"w")
        for k,v in self.output.iteritems():
            fout.write(v)

if __name__ == '__main__':
    # Generate a summary of MPRC and PRC performance (i.e., precision) and output to prc_summary_dir
    eva = MPRCresultAnalysis(goldstd_dir,mprc_result_dir, prc_result_dir, medline_dir, mprc_eval_dir,sampleSize,prc_summary_dir)
#     eva.run()
    # Interpret the similarity scores
    sample = loadData(base_dir,analyzerType,sampleSize, bm25SimilarDoc)
    for k,v in sample.iteritems():
        if k in os.listdir(prc_interpret_dir):
            continue  
        print k # v contains 100 candidates selected from BM25
        try:
            sa_mprc =SimilarityAnalysis({k:v}, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, knnterm_dir, model, goldstd_dir,mprc_result_dir,prc_result_dir,medline_dir,mprc_interpret_dir,sampleSize)
            sa_mprc.run_MPRC()
        except Exception as e:
            print "MPRC"
            print e
        try:
            sa_prc = SimilarityAnalysis({k:v}, base_dir, database_dir, stemmed_corpus_dir, vocab_dir, knnterm_dir, model, goldstd_dir,mprc_result_dir,prc_result_dir,medline_dir,prc_interpret_dir,sampleSize)
            sa_prc.run_PRC()
        except Exception as e:
            print "PRC"
            print e        