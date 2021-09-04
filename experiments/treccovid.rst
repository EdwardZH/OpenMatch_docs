TREC COVID
==========

-  **`Top Spot on TREC-COVID
   Challenge <https://ir.nist.gov/covidSubmit/about.html>`__** (May
   2020, Round2)

| The twin goals of the challenge are to evaluate search algorithms and
systems for helping scientists, clinicians, policy makers, and others
manage the existing and rapidly growing corpus of scientific literature
related to COVID-19, and to discover methods that will assist with
managing scientific information in future global biomedical crises.
|  `>> Reproduce Our Submit <./docs/experiments-treccovid.md>`__ `>>
About COVID-19 Dataset <https://www.semanticscholar.org/cord19>`__ `>>
Our Paper <https://arxiv.org/abs/2011.01580>`__

Data Statistics
---------------

Data can be downloaded from
`Datasets <https://ir.nist.gov/covidSubmit/data.html>`__. 
Each round (except Round1) only contains 5 new queries.

+---------------------+-----------+-------------------+
| Datasets            | Queries   | Valid Documents   |
+=====================+===========+===================+
| Round1              | 30        | 51.1K             |
+---------------------+-----------+-------------------+
| Round2              | 35        | 59.9K             |
+---------------------+-----------+-------------------+
| Round3              | 40        | 128.5K            |
+---------------------+-----------+-------------------+
| Round4              | 45        | 157.8K            |
+---------------------+-----------+-------------------+
| Round5              | 50        | 191.2K            |
+---------------------+-----------+-------------------+

Tasks
-----

\* `TREC-COVID <https://ir.nist.gov/covidSubmit/index.html/>`__. **Domain: BioMed Papers**.

Models
------

We use `SciBERT <https://arxiv.org/pdf/1903.10676.pdf>`__ base model for TREC-COVID experiments. 
`ReInfoSelect <https://arxiv.org/pdf/2001.10382.pdf>`__ and `MetaAdaptRank <https://arxiv.org/pdf/2012.14862.pdf>`__ frameworks 
are used to select more adaptive data for better performance of ranking models.

Results
------

+---------+-----------------+---------------------+-----------+----------+
| Round   | Method          | Pre-trained Model   | NDCG\@20   | P\@20     |
+=========+=================+=====================+===========+==========+
| 5       | MetaAdaptRank   | PudMedBERT          | 0.7904    | 0.9400   |
+---------+-----------------+---------------------+-----------+----------+

training
~~~~~~~~

Get training data from `google
drive <https://drive.google.com/file/d/1BT5gCOb1Kxkfh0BWqgUSgkxp2JPpRIWm/view?usp=sharing>`__.

Preprocess round1 data.

::

    python ./data/preprocess.py \
      -input_trec anserini.covid-r1.fusion1.txt \
      -input_qrels qrels-cord19-round1.txt \
      -input_queries questions_cord19-rnd1.txt \
      -input_docs cord19_0501_titabs.jsonl \
      -output ./data/dev_trec-covid-round1.jsonl

Train.

::

    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
            -task classification \
            -model bert \
            -train ./data/seanmed.train.320K-pairs.jsonl \
            -max_input 1280000 \
            -save ./checkpoints/scibert.bin \
            -dev ./data/dev_trec-covid-round1.jsonl \
            -qrels qrels-cord19-round1.txt \
            -vocab allenai/scibert_scivocab_uncased \
            -pretrain allenai/scibert_scivocab_uncased \
            -res ./results/scibert.trec \
            -metric ndcg_cut_10 \
            -max_query_len 32 \
            -max_doc_len 256 \
            -epoch 5 \
            -batch_size 16 \
            -lr 2e-5 \
            -n_warmup_steps 4000 \
            -eval_every 200

For ReInfoSelect training:

::

    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
            -task classification \
            -model bert \
            -reinfoselect \
            -reset \
            -train ./data/seanmed.train.320K-pairs.jsonl \
            -max_input 1280000 \
            -save ./checkpoints/scibert_rl.bin \
            -dev ./data/dev_trec-covid-round1.jsonl \
            -qrels qrels-cord19-round1.txt \
            -vocab allenai/scibert_scivocab_uncased \
            -pretrain allenai/scibert_scivocab_uncased \
            -checkpoint ./checkpoints/scibert.bin \
            -res ./results/scibert_rl.trec \
            -metric ndcg_cut_10 \
            -max_query_len 32 \
            -max_doc_len 256 \
            -epoch 5 \
            -batch_size 8 \
            -lr 2e-5 \
            -tau 1 \
            -n_warmup_steps 5000 \
            -eval_every 1

Inference
~~~~~~~~~

Get checkpoint. \*
`checkpoints <https://drive.google.com/drive/folders/1YHCMdSI7clFqPdfrRHA786PIecIxtKqA?usp=sharing>`__

Get data from Google Drive. \*
`round1 <https://drive.google.com/open?id=17CEoLecus232pCDwCECaJD4vNfh4OQao>`__
\*
`round2 <https://drive.google.com/open?id=1O6e8gXFnykkhN2icMCuWlMZkKUv6B3fV>`__

Filter round1 data from round2 data.

::

    python data/filter.py \
      -input_qrels qrels-cord19-round1.txt \
      -input_trec anserini.covid-r2.fusion2.txt \
      -output_topk 50 \
      -output_trec anserini.covid-r2.fusion2-filtered.txt

Preprocess round2 data.

::

    python ./data/preprocess.py \
      -input_trec anserini.covid-r2.fusion2-filtered.txt \
      -input_queries questions_cord19-rnd2.txt \
      -input_docs cord19_0501_titabs.jsonl \
      -output ./data/test_trec-covid-round2.jsonl

Reproduce scibert.

::

    CUDA_VISIBLE_DEVICES=0 \
    python inference.py \
            -task classification \
            -model bert \
            -max_input 1280000 \
            -test ./data/test_trec-covid-round2.jsonl \
            -vocab allenai/scibert_scivocab_uncased \
            -pretrain allenai/scibert_scivocab_uncased \
            -checkpoint ./checkpoints/scibert.bin \
            -res ./results/scibert.trec \
            -mode cls \
            -max_query_len 32 \
            -max_doc_len 256 \
            -batch_size 32

Reproduce reinfoselect scibert.

::

    CUDA_VISIBLE_DEVICES=0 \
    python inference.py \
            -task classification \
            -model bert \
            -max_input 1280000 \
            -test ./data/test_trec-covid-round2.jsonl \
            -vocab allenai/scibert_scivocab_uncased \
            -pretrain allenai/scibert_scivocab_uncased \
            -checkpoint ./checkpoints/reinfoselect_scibert.bin \
            -res ./results/reinfoselect_scibert.trec \
            -mode pooling \
            -max_query_len 32 \
            -max_doc_len 256 \
            -batch_size 32
