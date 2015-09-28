'''
This script learns vectors of words from the 4584 TREC2005 Genomics track dataset using module gensim.

Created on Aug 7, 2015
Last updated Aug 10, 2015
@author: Wei Wei
'''

import os, time, re, string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
from Index_TREC2005Genomics4584 import Corpus
from gensim import models#, similarities, corpora
import multiprocessing as mp

# os.system("taskset -p 0xff %d"%os.getpid())

class Corpus4Gensim(Corpus):
    def __init__(self,baseDir,pmidFile,medlineDir,databaseDir,gensimDir):
        super(Corpus4Gensim,self).__init__(baseDir,pmidFile,medlineDir,databaseDir)
        self.gensimDir = gensimDir
        
    def run4bioasq(self):
        '''Collect MEDLINE for 4.5M bioasq pmids and prepare sentences from these medline
           Completed 5000*621 records. Resume it.
        '''
        self.pmid = file(self.pmidFile).read().split("\n")
        if not os.listdir(self.medDir):
            print "Downloading MEDLINE..."
            super(Corpus4Gensim,self).prepMED()
        self.prepDatabase4gensim()
    
    def run(self):
        self.prepDatabase4gensim()
        
    def prepDatabase4gensim(self):
        '''extract information from medline and save it to disk. every mesh, abstract, title per line'''
        for doc in os.listdir(self.medDir):
            tiab = super(Corpus4Gensim,self).extractTiAb(os.path.join(self.medDir,doc))
            mesh = super(Corpus4Gensim,self).extractMH(os.path.join(self.medDir,doc))
            self.saveTiAbMH(tiab,mesh)

    def saveTiAbMH(self,TiAbDict,MeSHDict):
        for pmid in MeSHDict.keys():
            meshStr = ". ".join(MeSHDict[pmid])
            titleStr = TiAbDict[pmid][0].strip(".")+". "+TiAbDict[pmid][0].strip(".")
            absStr = TiAbDict[pmid][1]
            text = ". ".join([meshStr,titleStr,absStr])
            fout = file(os.path.join(self.gensimDir,pmid),"w")
            fout.write(text)

class Consumer(mp.Process):
    def __init__(self,task_queue): # result_queue
        mp.Process.__init__(self)
        self.task_queue = task_queue
#         self.result_queue = result_queue
        
    def run(self):
        '''Split texts into sentences for word2vec'''
#         proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print "%s: Exiting" %mp.current_process()
                self.task_queue.task_done()
                break
#             print "%s: %s"%(proc_name,next_task)
            next_task.__call__()
#             answer = next_task.__call__()
            self.task_queue.task_done()
#             self.result_queue.put(answer)
        return
    
class Task(object):
    def __init__(self,inFile,outFile):
        self.inputFile = inFile
        self.outputFile = outFile
    
    def __call__(self):
        '''Keep letter-digit combinations, and stem terms'''
        sentences = []
        text = file(self.inputFile).read()
        sent_tokenize_list = sent_tokenize(text.strip().lower(), "english") # a sentence list from doc 
        if sent_tokenize_list: # if sent_tokenize_list is not empty
            porter_stemmer = PorterStemmer()
            for sent in sent_tokenize_list:
                words = TreebankWordTokenizer().tokenize(sent) # tokenize the sentence
                words = [word.strip(string.punctuation) for word in words]
                words = [word for word in words if not word in stopwords.words("english")]
                words = [word for word in words if len(word)>1] # remove single letters and non alphabetic characters
                words = [word for word in words if re.search('[a-zA-Z]',word)]
                words = [porter_stemmer.stem(word) for word in words]
                sentences.append(words)
        self.__save__(sentences)
    
    def __save__(self,sentences):
        fout = file(self.outputFile,"w")
        texts = [" ".join(sent) for sent in sentences]
        fout.write("\n".join(texts))

    def __str__(self):
        return "%s "%(self.inputFile)

class Sentence(object):
    def __init__(self,wkdir,evalDir):
        self.wkdir = wkdir
        self.evalDir = evalDir
    
    def __iter__(self):
        for doc in os.listdir(self.wkdir)[:1000000]:
            if doc not in os.listdir(self.evalDir): # if this doc is not in the evaluation set, split it for word2vec training
                for line in file(os.path.join(self.wkdir,doc)):
                    yield line.split()
            
def splitTexts(base_dir,gensim_dir,sentence_dir):
    '''Prepare sentences for gensim. One sentence per line.'''
    print "Splitting texts..."
    tasks = mp.JoinableQueue()

    num_consumers = mp.cpu_count()*4
    print "creating %d consumers "%num_consumers
    consumers = [Consumer(tasks) for i in xrange(num_consumers)]
    
    for w in consumers:
        w.start()
    
    for doc in os.listdir(gensim_dir):
        inFile = os.path.join(gensim_dir,doc)
        outFile = os.path.join(sentence_dir,doc)
        tasks.put(Task(inFile,outFile))
        
    for i in xrange(num_consumers):
        tasks.put(None)
        
    tasks.join()
    
if __name__ == '__main__':
    ## Directory settings
    bioasq_base_dir = "/home/w2wei/Research/mesh/data/train/" # for collecting 4.5 million records
    bioasq_pmidFile = os.path.join(bioasq_base_dir,"allMeSH_limitjournals2014.pmids")
    bioasq_medline_dir = os.path.join(bioasq_base_dir,"allMeSH_limitjournals2014_medline")
    bioasq_gensim_dir = os.path.join(bioasq_base_dir, "gensim")
    bioasq_database_dir = os.path.join(bioasq_gensim_dir,"raw_sentences") # 3101418 files. One line per file.
    bioasq_sentence_dir = os.path.join(bioasq_gensim_dir,"allMH_sentences") # One sentence per line
    bioasq_sentence_stemmed_dir = os.path.join(bioasq_gensim_dir,"allMH_sentences_porterstemmed") # One sentence per line
    bioasq_model_dir = os.path.join(bioasq_gensim_dir,"model")
    eval_set_dir = "/home/w2wei/Research/mesh/data/TREC/2005/4584rel/database"
    if not os.path.exists(bioasq_database_dir):
        os.mkdir(bioasq_database_dir)
    if not os.path.exists(bioasq_model_dir):
        os.mkdir(bioasq_model_dir)
    if not os.path.exists(bioasq_sentence_stemmed_dir):
        os.mkdir(bioasq_sentence_stemmed_dir)
        
    # get BioASQ data ready for gensim word2vec
    t0=time.time()
    if not os.listdir(bioasq_database_dir):
        print "Preparing BioASQ MEDLINE files."
        bioasq_corpus = Corpus4Gensim(baseDir=bioasq_base_dir,pmidFile=bioasq_pmidFile,medlineDir=bioasq_medline_dir,databaseDir=bioasq_medline_dir,gensimDir=bioasq_database_dir)
        bioasq_corpus.run4bioasq()
    t1=time.time()
    print "BioASQ data are ready."
    print "Time cost: ",t1-t0

    print "Use bioasq_sentence_stemmed_dir, not bioasq_sentence_dir.\n"
    if not os.listdir(bioasq_sentence_stemmed_dir):
        splitTexts(bioasq_base_dir,bioasq_database_dir,bioasq_sentence_stemmed_dir) # for the BioASQ corpus, 3.1 M medline
    print "BioASQ sentences for Word2Vec are ready."

    ## train a Word2Vec model
    t0=time.time()
    feature_num = 100
    min_word_count = 40
    num_workers = mp.cpu_count()
    window = 10
    
#     print "Training data size: ",len(os.listdir(bioasq_sentence_stemmed_dir))
    sentences = Sentence(bioasq_sentence_stemmed_dir,eval_set_dir) # 3101418
    print "Start training..."
    t1 = time.time()
    model = models.Word2Vec(sentences,size=feature_num,window=window,min_count=min_word_count,workers=num_workers)
    t2 = time.time()
    print "Training completed."
    print "Training time: ",t2-t1
    vocab = model.vocab.keys()
    print "vocab size ",len(vocab)
    model.save(os.path.join(bioasq_model_dir,"model_bioasq_stemmed_vocab_%d.ml"%len(vocab)))
#     model = models.Word2Vec.load(os.path.join(bioasq_model_dir,"model_bioasq_stemmed_vocab_%d.ml"%len(vocab)))
    
    fsim = file(os.path.join(bioasq_model_dir,"model_bioasq_stemmed_vocab_%d.simwd"%len(vocab)),"w")
    for term in vocab:
        rec = model.most_similar(term,topn=3)
        rec = term+"\t"+str(rec)+"\n"
        fsim.write(rec)
        
    t3=time.time()
    print t3-t0
    
    
