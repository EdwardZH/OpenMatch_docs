Ad-hoc Search
=============

All experiments are measured on ndcg@20 with 5 fold cross-validation.

Data Statistics
---------------

Data can be downloaded from
`Datasets <https://cloud.tsinghua.edu.cn/d/77741ef1c1704866814a/>`__.
Note that we followed the settings of Zhuyun Dai's `work <https://dl.acm.org/doi/pdf/10.1145/3159652.3159659>`__.
We only use the subset of each dataset which was retrievaled by Indri's SDM for experiments.

+---------------------+-------------------+--------------------------+--------------------------------------+
| Datasets            | Queries/Anchors   | Query/Anchor-Doc Pairs   | Released Files                       |
+=====================+===================+==========================+======================================+
| **ClueWeb09-B**     | 200               | 47.1K                    | Queries, Q-D Relations, SDM scores   |
+---------------------+-------------------+--------------------------+--------------------------------------+
| **Robust04**        | 249               | 311K                     | Queries, Q-D Relations, SDM scores   |
+---------------------+-------------------+--------------------------+--------------------------------------+
| **ClueWeb12-B13**   | 100               | 28.9K                    | Queries, Q-D Relations, SDM scores   |
+---------------------+-------------------+--------------------------+--------------------------------------+

As we cannot release the document contents, the document IDs are used instead.

Tasks
-----

\* `ClueWeb09 <http://www.lemurproject.org/clueweb09/>`__. **Domain: Web Pages**.
The ClueWeb09 dataset was created to support research on information retrieval and related human
language technologies. It consists of about 1 billion web pages in ten languages that were collected
in January and February 2009. The dataset is used by several tracks of the TREC conference.

\* `ClueWeb12 <http://www.lemurproject.org/clueweb12.php/>`__. **Domain: Web Pages**.
The ClueWeb12 dataset was created to support research on information retrieval and related human
language technologies. The dataset consists of 733,019,372 English web pages, collected between
February 10, 2012 and May 10, 2012. ClueWeb12 is a companion or successor to the ClueWeb09 web
dataset. Distribution of ClueWeb12 began in January 2013.


\* `Robust04 <https://trec.nist.gov/data/t13_robust.html>`__. **Domain: News Articles**.
The goal of the Robust track is to improve the consistency of retrieval technology by focusing
on poorly performing topics. In addition, the track brings back a classic, ad hoc retrieval task
in TREC that provides a natural home for new participants. An ad hoc task in TREC investigates
the performance of systems that search a static set of documents using previously-unseen topics.
For each topic, participants create a query and submit a ranking of the top 1000 documents for that topic.

Models
------

In ad-hoc search experiments, we use
`KNRM <https://dl.acm.org/doi/pdf/10.1145/3077136.3080809/>`__,
`Conv-KNRM <https://dl.acm.org/doi/pdf/10.1145/3159652.3159659/>`__,
`EDRM <https://arxiv.org/pdf/1805.07591/>`__ and
`TK <https://arxiv.org/pdf/2002.01854.pdf/>`__ as neural IR models,
`BERT <https://arxiv.org/pdf/1810.04805.pdf/>`__ and
`ELECTRA <https://arxiv.org/pdf/2003.10555.pdf/>`__ as pretrained models.

Training
~~~~~~~~

For training neural ranking models, we use the KNRM model for example. The **-model** parameter can be set to any neural ranking model, such as **tk**, **cknrm** and **edrm**.

::

    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
            -task ranking \
            -model knrm \
            -train queries=./data/query.jsonl,docs=./data/doc.jsonl,qrels=./data/qrels,trec=./data/fold0/train.trec \
            -max_input 12800000 \
            -save ./checkpoints/knrm.bin \
            -dev queries=./data/query.jsonl,docs=./data/doc.jsonl,qrels=./data/qrels,trec=./data/fold0/dev.trec \
            -qrels ./data/qrels \
            -vocab ./data/glove.6B.100d.txt \
            -res ./results/knrm.trec \
            -metric ndcg_cut_20 \
            -n_kernels 21 \
            -max_query_len 16 \
            -max_doc_len 256 \
            -epoch 1 \
            -batch_size 32 \
            -lr 1e-3 \
            -n_warmup_steps 1000 \
            -eval_every 100

For training pretrained models, we use the BERT model for example. We can also use any pretrained model.

::

    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
            -task ranking \
            -model bert \
            -train queries=./data/query.jsonl,docs=./data/doc.jsonl,qrels=./data/qrels,trec=./data/fold0/train.trec \
            -max_input 12800000 \
            -save ./checkpoints/bert.bin \
            -dev queries=./data/query.jsonl,docs=./data/doc.jsonl,qrels=./data/qrels,trec=./data/fold0/dev.trec \
            -qrels ./data/qrels \
            -vocab bert-base-uncased \
            -pretrain bert-base-uncased \
            -res ./results/bert.trec \
            -metric ndcg_cut_20 \
            -max_query_len 32 \
            -max_doc_len 221 \
            -epoch 1 \
            -batch_size 16 \
            -lr 3e-6 \
            -n_warmup_steps 1000 \
            -eval_every 100

For getting classic IR features (e.g. boolean, language model, tfidf, bm25 ...), we need first read the document file to a dict (docs[docid] = doc), then for each given query-doc pair,
classic features can be compute as follows:

.. code:: python

    import OpenMatch as om

    corpus = om.Corpus(docs)
    docs_terms, df, total_df, avg_doc_len = corpus.cnt_corpus()
    query_terms, query_len = corpus.text2lm(query)
    doc_terms, doc_len = corpus.text2lm(doc)
    extractor = om.ClassicExtractor(query_terms, doc_terms, df, total_df, avg_doc_len)
    features = extractor.get_feature()


Inference
~~~~~~~~~

For neural ranking models:

::

    CUDA_VISIBLE_DEVICES=0 \
    python inference.py \
        -task ranking \
        -model knrm \
        -max_input 1280000 \
        -vocab ./data/glove.6B.300d.txt \
        -checkpoint ./checkpoints/knrm.bin \
        -test queries=./data/query.jsonl,docs=./data/doc.jsonl,qrels=./data/qrels,trec=./data/fold0/test.trec \
        -res ./results/knrm.trec \
        -max_query_len 16 \
        -max_doc_len 256 \
        -batch_size 512

For pretrained models:

::

    CUDA_VISIBLE_DEVICES=0 \
    python inference.py \
            -task ranking \
            -model bert \
            -max_input 12800000 \
            -test queries=./data/query.jsonl,docs=./data/doc.jsonl,qrels=./data/qrels,trec=./data/fold0/test.trec \
            -vocab bert-base-uncased \
            -pretrain bert-base-uncased \
            -checkpoint ./checkpoints/bert.bin \
            -res ./results/bert.trec \
            -max_query_len 32 \
            -max_doc_len 221 \
            -batch_size 256

Results
-------

\* `Ad-hoc Search <./docs/experiments-adhoc.md>`__

+-------------+----------------+---------------+-------------+------------+-------------+
| Retriever   | Reranker       | Coor-Ascent   | ClueWeb09   | Robust04   | ClueWeb12   |
+=============+================+===============+=============+============+=============+
| SDM         | KNRM           | \-            | 0.1880      | 0.3016     | 0.0968      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | Conv-KNRM      | \-            | 0.1894      | 0.2907     | 0.0896      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | EDRM           | \-            | 0.2015      | 0.2993     | 0.0937      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | TK             | \-            | 0.2306      | 0.2822     | 0.0966      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | BERT Base      | \-            | 0.2701      | 0.4168     | 0.1183      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | ELECTRA Base   | \-            | 0.2861      | 0.4668     | 0.1078      |
+-------------+----------------+---------------+-------------+------------+-------------+

MetaAdaptRank
---------------

Here provides the guiding code for the method of meta-learning to reweight synthetic weak supervision data, which uses target data to reweight contrastive synthetic data (CTSyncSup) during the learning to rank process. A detailed introduction to the technology can be found in the paper
`Few-Shot Text Ranking with Meta Adapted Synthetic Weak Supervision <https://arxiv.org/pdf/2012.14862.pdf>`__. This method contains two parts:

1. Contrastive Supervision Synthesis (CTSyncSup)
2. Meta Learning to Reweight


Contrastive Supervision Synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source-domain NLG training. We train two query generators (QG & ContrastQG) with the MS MARCO dataset using ``train_nlg.sh``:

::

    bash train_nlg.sh


Optional arguments:

::

    --generator_mode            choices=['qg', 'contrastqg']
    --pretrain_generator_type   choices=['t5-small', 't5-base']
    --train_file                The path to the source-domain nlg training dataset   
    --save_dir                  The path to save the checkpoints data


Target-domain NLG inference. The whole nlg inference pipline contains five steps:

-  1/ Data preprocess
-  2/ Seed query generation
-  3/ BM25 subset retrieval
-  4/ Contrastive doc pairs sampling
-  5/ Contrastive query generation

1\/ Data preprocess. convert target-domain documents into the nlg format using ``prepro_dataset.sh`` in the folder ``preprocess``:

::

    bash prepro_dataset.sh

Optional arguments:

::

--dataset_name          The name of the target dataset
--input_path            The path to the target dataset   
--output_path           The path to save the preprocess data


2\/ Seed query generation. utilize the trained QG model to generate seed queries for each target documents using ``qg_inference.sh`` in the folder ``run_shell``:

::

    bash qg_inference.sh

Optional arguments:

::

--generator_mode            choices='qg'   
--pretrain_generator_type   choices=['t5-small', 't5-base']   
--target_dataset_name       The name of the target dataset
--generator_load_dir        The path to the pretrained QG checkpoints


3\/ BM25 subset retrieval. utilize BM25 to retrieve document subset according to the seed queries using the following shell commands in the folder ``bm25_retriever``:

::
    
    bash build_index.sh
    bash retrieve.sh


Optional arguments:

::

    --dataset_name          The name of the target dataset
    --data_path          The name of the target dataset

4\/ Contrastive doc pairs sampling. pairwise sample contrastive doc pairs from the BM25 retrieved subset using ``sample_contrast_pairs.sh``:

::

    bash sample_contrast_pairs.sh

Optional arguments:

::

--dataset_name          choices=['clueweb09', 'robust04', 'trec-covid']   
--generator_folder      choices=['t5-small', 't5-base']


5\/ Contrastive query generation. utilize the trained ContrastQG model to generate new queries based on contrastive document pairs using ``nlg_inference.sh``:

::

    bash nlg_inference.sh

-  Optional arguments:

::

    --generator_mode            choices='contrastqg'   
    --pretrain_generator_type   choices=['t5-small', 't5-base']   
    --target_dataset_name       choices=['clueweb09', 'robust04', 'trec-covid']   
    --generator_load_dir        The path to the pretrained ContrastQG checkpoints



Meta Learning to Reweight
~~~~~~~~~

The code to run meta-learning is in the shell file

::

    bash meta_dist_train.sh

In the shell file, the code is written as

::
    
    export gpu_num=4 ## GPU Number
    export master_port=23900
    export job_name=MetaBERT

    ## ************************************
    export DATA_DIR= ## please set your dataset path here.
    export SAVE_DIR= ## please set your saving path here.

    ## ************************************
    CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -u -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port $master_port meta_dist_train.py \
    -job_name $job_name \
    -save_folder $SAVE_DIR/results \
    -model bert \
    -task ranking \
    -max_input 12800000 \
    -train queries=$DATA_DIR/queries.train.tsv,docs=$DATA_DIR/collection.tsv,qrels=$DATA_DIR/qrels.train.tsv,trec=$DATA_DIR/trids_bm25_marco-10.tsv \
    -dev queries=$DATA_DIR/queries.dev.small.tsv,docs=$DATA_DIR/collection.tsv,qrels=$DATA_DIR/qrels.dev.small.tsv,trec=$DATA_DIR/run.msmarco-passage.dev.small.100.trec \
    -target trec=$DATA_DIR/devids_bm25_marco.tsv \
    -qrels $DATA_DIR/qrels.dev.small.tsv \
    -vocab bert-base-uncased \
    -pretrain bert-base-uncased \
    -metric mrr_cut_10 \
    -max_query_len 32 \
    -max_doc_len 221 \
    -epoch 3 \
    -train_batch_size 8 \
    -target_batch_size 16 \
    -gradient_accumulation_steps 2 \
    -dev_eval_batch_size 1024 \
    -lr 3e-6 \
    -n_warmup_steps 160000 \
    -logging_step 2000 \
    -eval_every 10000 \
    -eval_during_train \

The tsv format of ``-target`` data is totally the same with the
``-train`` data.

::

    query_id \t pos_docid \t neg_docid

+-------------+-----------------+----------------+---------------+-------------+------------+--------------+
| Retriever   | Reranker        | Augment Data   | Coor-Ascent   | ClueWeb09   | Robust04   | TREC-COVID   |
+=============+=================+================+===============+=============+============+==============+
| SDM         | MetaAdaptRank   | CTSyncSup      | \+            | 0.3416      | 0.4916     | 0.8378       |
+-------------+-----------------+----------------+---------------+-------------+------------+--------------+
