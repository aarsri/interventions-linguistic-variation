# Linguistic Variation Interventions

English text transformations to simulate linguistic variation.

Code for the [paper](https://arxiv.org/abs/2404.07304) "We're Calling an Intervention: Taking a Closer Look at Language Model Adaptation to Different Types of Linguistic Variation"

## Getting Started
```
pip install nltk
```

#### to use end_mod and affix_mod
```
wget https://github.com/kbatsuren/MorphyNet/blob/main/eng/eng.inflectional.v1.tsv
wget https://github.com/kbatsuren/MorphyNet/blob/main/eng/eng.derivational.v1.tsv
```
 
#### to use reg_mod
```
pip install transformers
git clone https://github.com/tatHi/maxmatch_dropout.git
```

#### to use multi-value
follow instructions from [Multi-VALUE](https://github.com/SALT-NLP/multi-value.git)
```
D = MultiDialect()`
D.convert_sae_to_dialect(sentence)`
```

#### sample execution
```
$ python
>>> import intervene
>>> intervene.ipa_mod('superstore')
>>> 'zuberzdore'
```
