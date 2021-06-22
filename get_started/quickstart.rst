Quick Start
==============

Preliminary
~~~~~~~~~~~~~~~~


Given the query and document, ranking models aim to calculate the matching score between them.

.. code:: python

    import torch
    import OpenMatch as om

    query = "Classification treatment COVID-19"
    doc = "By retrospectively tracking the dynamic changes of LYM% in death cases and cured cases, this study suggests that lymphocyte count is an effective and reliable indicator for disease classification and prognosis in COVID-19 patients."


Traditional IR models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMatch provides several classic IR feature extractors. Document collection is needed for the creation of inverted dict. The parameter "docs" is a dict for all documents: "docs[docid] = doc".

.. code:: python

    corpus = om.Corpus(docs)
    docs_terms, df, total_df, avg_doc_len = corpus.cnt_corpus()
    query_terms, query_len = corpus.text2lm(query)
    doc_terms, doc_len = corpus.text2lm(doc)
    extractor = om.ClassicExtractor(query_terms, doc_terms, df, total_df, avg_doc_len)
    features = extractor.get_feature()


Pretrained IR models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMatch inherients parameters of pretrained language models from hugginface's trasnformers.

.. code:: python

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    input_ids = tokenizer.encode(query, doc)
    model = om.models.Bert("allenai/scibert_scivocab_uncased")
    ranking_score, ranking_features = model(torch.tensor(input_ids).unsqueeze(0))

Neural IR models
~~~~~~~~~~~~~~~~~~

.. code:: python

    tokenizer = om.data.tokenizers.WordTokenizer(pretrained="./data/glove.6B.300d.txt")
    query_ids, query_masks = tokenizer.process(query, max_len=16)
    doc_ids, doc_masks = tokenizer.process(doc, max_len=128)
    model = om.models.KNRM(vocab_size=tokenizer.get_vocab_size(),
                           embed_dim=tokenizer.get_embed_dim(),
                           embed_matrix=tokenizer.get_embed_matrix())
    ranking_score, ranking_features = model(torch.tensor(query_ids).unsqueeze(0),
                                            torch.tensor(query_masks).unsqueeze(0),
                                            torch.tensor(doc_ids).unsqueeze(0),
                                            torch.tensor(doc_masks).unsqueeze(0))

The pretrained word embeddings (GloVe) can be downloaded using:

::

    wget http://nlp.stanford.edu/data/glove.6B.zip -P ./data
    unzip ./data/glove.6B.zip -d ./data


Evaluation
~~~~~~~~~~~~~~

::

    metric = om.Metric()
    res = metric.get_metric(qrels, ranking_list, 'ndcg_cut_20')
    res = metric.get_mrr(qrels, ranking_list, 'mrr_cut_10')









