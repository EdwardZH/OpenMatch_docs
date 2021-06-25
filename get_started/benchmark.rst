Settings and Benchmark
=========================

Information Retrieval (IR) aims to retrieve relevant documents from a large-scale document collection to satisfy user needs, which have been used in many real-world applications, such as digital libraries, expert findings. A good IR system can benefit lots of NLP tasks, such as question answering, fact verification, and entity search. With the increasing of online documents and searching demand, achieving an effective and efficient IR system is very crucial for our society.

Ranking Scenarios in IR
-------------------------
According to different application scenarios, IR tasks can be divided into the following categories, ad hoc retrieval, and question answering. 
For example, for two related questions about Obama's family members, for document retrieval, we look forward to returning a document or paragraph related to input keywords.
For question answering, we usually input a natural language sentence, so we need to understand the underlying semantic information of the given query and return the corresponding search results according to the semantic information described in the query.


.. image:: ../images/irtasks.png

During the past decades, many ranking models (Traditional Information Retrieval Models) have been proposed to retrieve related documents, including vector space models, probabilistic models, and learning to rank (LTR) models. Such ranking techniques, especially the LTR models, have already achieved great success in many IR applications, such as commercial web search engines, Google, Bing, and Baidu. Nevertheless, the above models usually pay more attention to keyword-oriented searching with exact matches and conduct shallow comprehension for both query and document because of the vocabulary mismatch problem.


In recent years, deep neural networks have led to a new tendency in the Information Retrieval area. Existing LTR models usually derive from hand-crafted features, which are usually time-consuming to design and show less generalize ability to other ranking scenarios.
Compared to LTR models, neural models can be optimized end-to-end with relevance supervisions, and do not need these handcraft features.

In addition, the neural models also show their capability to recognize the potential matching patterns with the strong capability of the neural network, which helps ranking models comprehend the semantic information of query and candidate document. Due to these potential benefits, the neural models show their effectiveness in understanding the user intents, which thrives on searching in the question answering and fact verification tasks. The work of neural models has substantial grown in applying neural networks for ranking models in both academia and industry in recent years.






IR Benchmarks
-------------------------

The statistics of some IR datasets are listed as follows:

+----------------+--------+-----------------+
| Dataset        | Query  | Query-Doc Pair  |
+================+========+=================+
| ClueWeb09-B    | 200    | 50M             |
+----------------+--------+-----------------+
| ClueWeb12-B13  | 100    | 50M             |
+----------------+--------+-----------------+
| Robust04       | 249    | 500K            |
+----------------+--------+-----------------+
| MS MARCO       | 1.01M  | 485M            |
+----------------+--------+-----------------+
| TREC COVID     | 50     | 191k            |
+----------------+--------+-----------------+




