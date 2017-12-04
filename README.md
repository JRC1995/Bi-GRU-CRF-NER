## Bi-Directional GRU CRF for Named Entity Recognition.

Dataset Source: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
(CoNLL 2002)

(only using first 10,000 samples with sentence length<=20 for training, validation, and testing)

The paper introducing CRF (Conditional Random Field):
[Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data - by John Lafferty, Andrew McCallum, Fernando C.N. Pereira](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)

CRF in details: [An Introduction to Conditional Random Fields - by Charles Sutton and Andrew McCallum](http://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)

Notes I used to understand CRF:
http://pages.cs.wisc.edu/~jerryzhu/cs838/CRF.pdf
https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html  
https://www.youtube.com/user/hugolarochelle/search?query=crf

This model is somewhat based on:

[Bidirectional LSTM-CRF Models for Sequence Tagging - by Zhiheng Huang, Wei Xu, Kai Yu](https://arxiv.org/pdf/1508.01991.pdf)
