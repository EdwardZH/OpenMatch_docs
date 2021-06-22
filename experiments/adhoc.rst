Ad-hoc Search
=============

All experiments are measured on ndcg@20 with 5 fold cross-validation.

Data Statistics
---------------

Data can be downloaded from
`Datasets <https://cloud.tsinghua.edu.cn/d/77741ef1c1704866814a/>`__.

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

Websites
--------

\* `ClueWeb09 <http://www.lemurproject.org/clueweb09/>`__.
\* `ClueWeb12 <http://www.lemurproject.org/clueweb12.php/>`__.
\* `Robust04 <https://trec.nist.gov/data/t13_robust.html/>`__.

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
--------

We use the KNRM model for example.

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

Inference
---------

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
        -batch_size 32

Results
-------

\* `Ad-hoc Search <./docs/experiments-adhoc.md>`__

+-------------+----------------+---------------+-------------+------------+-------------+
| Retriever   | Reranker       | Coor-Ascent   | ClueWeb09   | Robust04   | ClueWeb12   |
+=============+================+===============+=============+============+=============+
| SDM         | KNRM           | -             | 0.1880      | 0.3016     | 0.0968      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | Conv-KNRM      | -             | 0.1894      | 0.2907     | 0.0896      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | EDRM           | -             | 0.2015      | 0.2993     | 0.0937      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | TK             | -             | 0.2306      | 0.2822     | 0.0966      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | BERT Base      | -             | 0.2701      | 0.4168     | 0.1183      |
+-------------+----------------+---------------+-------------+------------+-------------+
| SDM         | ELECTRA Base   | -             | 0.2861      | 0.4668     | 0.1078      |
+-------------+----------------+---------------+-------------+------------+-------------+
