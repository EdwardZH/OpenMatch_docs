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


+-------------+-----------------+----------------+---------------+-------------+------------+--------------+
| Retriever   | Reranker        | Augment Data   | Coor-Ascent   | ClueWeb09   | Robust04   | TREC-COVID   |
+=============+=================+================+===============+=============+============+==============+
| SDM         | ReInfoSelect    | CTSyncSup      | \+            | 0.3243      | 0.4816     | 0.8230       |
+-------------+-----------------+----------------+---------------+-------------+------------+--------------+
| SDM         | MetaAdaptRank   | CTSyncSup      | \+            | 0.3416      | 0.4916     | 0.8378       |
+-------------+-----------------+----------------+---------------+-------------+------------+--------------+
