'''
Created on Sep 2, 2015

@author: Wei Wei
'''

import os
# Experiment setting
sampleSize = 4584
expSetting = "lowfreq2_%dsample"%sampleSize # the experiment setting
analyzerType = "PorterStemmer"
model = r"/home/w2wei/Research/mesh/data/train/gensim/model/model_bioasq_stemmed_vocab_131075.ml"
bm25SimilarDoc = r"/home/w2wei/Research/mesh/data/TREC/2005/4584rel/BM25_results/BM25_similarArticles_%s.pkl"%analyzerType

# Parameter setting
base_dir = "/home/w2wei/Research/mesh/data/TREC/2005/4584rel"
rawPMID_dir = "/home/w2wei/Research/mesh/data/TREC/2005/genomics.qrels.large.txt"
medline_dir = os.path.join(base_dir,"medline")
database_dir = os.path.join(base_dir,"database")
index_dir = os.path.join(base_dir,"index")
stemmed_corpus_dir = os.path.join(base_dir,"stemmed_corpus")
vocab_dir = os.path.join(base_dir,"vocab")
experiment_dir = os.path.join(base_dir,"experiments")
mprc_skg_result_dir = os.path.join(experiment_dir,"mprc_skg_"+analyzerType,expSetting)
mprc_result_dir = os.path.join(experiment_dir,"mprc_"+analyzerType,expSetting)
mprc_weighted_result_dir = os.path.join(experiment_dir,"mprc_weighted_"+analyzerType,expSetting)
mprc_filter_result_dir = os.path.join(experiment_dir,"mprc_filter_"+analyzerType,expSetting)
prc_result_dir = os.path.join(experiment_dir,"prc_"+analyzerType,expSetting)
goldstd_dir = os.path.join(base_dir, "gensim","gold_std")
eval_dir = os.path.join(base_dir,"evaluation", analyzerType)
mprc_skg_eval_dir = os.path.join(eval_dir,"mprc_skg")
mprc_eval_dir = os.path.join(eval_dir,"mprc")
mprc_weighed_eval_dir = os.path.join(eval_dir,"mprc_weighted")
mprc_filter_eval_dir = os.path.join(eval_dir,"mprc_filter")
prc_eval_dir = os.path.join(eval_dir,"prc")
knnterm_dir = os.path.join(base_dir,"knn_terms")
interpret_dir = os.path.join(base_dir,"interpretation")
mprc_skg_interpret_dir = os.path.join(interpret_dir,"mprc_skg")
mprc_interpret_dir = os.path.join(interpret_dir,"mprc")
prc_interpret_dir = os.path.join(interpret_dir,"prc")
stat_interpret_dir = os.path.join(interpret_dir,"stat")
prc_precision_dir = os.path.join(base_dir,"evaluation", "PRC_precision")
prc_recall_dir = os.path.join(base_dir,"evaluation", "PRC_recall")
prc_f1_dir = os.path.join(base_dir,"evaluation", "PRC_f1")
prc_map_dir = os.path.join(base_dir,"evaluation", "PRC_map")
prc_summary_dir = os.path.join(base_dir,"evaluation", "PRC_summary")
prc_summary_file = os.path.join(prc_summary_dir,"%d.txt"%sampleSize)
classifier_dir = os.path.join(base_dir,"classifier")
classifier_data_dir = os.path.join(classifier_dir,"data")
logistic_regression_dir = os.path.join(classifier_dir,"LR")
kl_dir = os.path.join(classifier_dir,"KL")
kl_data_dir = os.path.join(kl_dir,"data")
kl_vocab_dir = os.path.join(kl_dir,"vocab")
kl_stemmed_corpus_dir = os.path.join(kl_dir,"stemmed_corpus")