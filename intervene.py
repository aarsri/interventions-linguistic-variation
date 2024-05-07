# AAROHI SRIVASTAVA
# MIT LICENSE 2024

import re
import math
import string
from nltk.corpus import wordnet
from transformers import AutoTokenizer
from maxmatch_dropout import maxMatchTokenizer

nltk.download('wordnet')

# This code contains 10 intervention functions. Most take as input a word (string). In the original paper, words are segmented using NLTK's word_tokenizer.
# Some functions also take a part of speech tag as input. The code assumes Penn Treebank labels for English part of speech tags.

# ORTHOGRAPHIC

def ipa_mod(word):
	if word.isalpha() is False:
		print('Word does not satisfy isalpha. Run at your own risk.')
	swaps = {ord('p'): ord('b'), ord('t'): ord('d'), ord('k'): ord('g'), ord('f'): ord('v'), ord('s'): ord('z')}
	caps = {}
	for k, v in swaps.items():
		caps[ord(chr(k).capitalize())] = ord(chr(v).capitalize())
	rev = {}
	for k, v in swaps.items():
		rev[v] = k
	for k, v in caps.items():
		rev[v] = k
	orthographic_map = {}
	orthographic_map.update(swaps)
	orthographic_map.update(caps)
	orthographic_map.update(rev)
	word = word.translate(orthographic_map)
	return word

def shift_mod(word):
	if word.isalpha() is False:
		print('Word does not satisfy isalpha. Run at your own risk.')
	orthographic_map = {}
	for i in range(25):
		orthographic_map[ord('a')+i] = ord('a')+i+1
	orthographic_map[ord('z')] = ord('a') # do the new Zs get messed up?
	for i in range(25):
		orthographic_map[ord('A')+i] = ord('A')+i+1
	orthographic_map[ord('Z')] = ord('A')
	word = word.translate(orthographic_map)
	return word

# SUBWORD BOUNDARIES

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
mmt = maxMatchTokenizer.MaxMatchTokenizer()
mmt.loadBertTokenizer(tokenizer, doNaivePreproc=False) #True is safer

# returns a list of tokens, assuming the default tokenizer is BERT-base (WordPiece)
def reg_mod(word):
	if word.isalpha() is False:
		print('Word does not satisfy isalpha. Run at your own risk.')
	return mmt.tokenize(word, p=0.5)

def char_mod(word):
	if word.isalpha() is False:
		print('Word does not satisfy isalpha. Run at your own risk.')
	return [c for c in word]

def pig_mod(word):
	if word.isalpha() is False:
		print('Word does not satisfy isalpha. Run at your own risk.')
	vowels = ['a','e','i','o','u']
	if word[0].lower() in vowels:
		return word+"yay"
	for i in range(1,len(word)):
		if word[i].lower() in vowels:
			new = word[i:]+word[:i]+'ay'
			if word[0].isupper():
				return new.capitalize()
			else:
				return new
	return word+'ay'

# MORPHOSYNTACTIC

inflect_words = {} # keys are inflected words, values do not have the inflection
try:
	with open('eng.inflectional.v1.tsv', 'r') as f:
		inflect_data = f.readlines()
	f.close()
	for line in inflect_data:
		line = line.split('\t')
		if line[1].strip() != line[0].strip():
			inflect_words[line[1].strip()] = line[0].strip()
except:
	print('Download inflectional morphemes from MorphyNet to use end_mod: https://github.com/kbatsuren/MorphyNet/blob/main/eng/eng.inflectional.v1.tsv')

def end_mod(word):
	if word.isalpha() is False:
		print('Word does not satisfy isalpha. Run at your own risk.')
	if word.lower() in inflect_words.keys():
		if word[0].isupper():
			word = inflect_words[word.lower()].capitalize()
		else:
			word = inflect_words[word.lower()]
	return word

# LEXICOSEMANTIC

morpho_words = {}
prefixes = []
suffixes = []
try:
	with open('eng.derivational.v1.tsv', 'r') as f:
		morpho_data = f.readlines()
	f.close()
	for line in morpho_data:
		line = line.split('\t')
		morpho_words[line[1]] = (line[4].strip(), line[5].strip(), line[3].strip())
		if line[5].strip()=='prefix':
			if line[4].strip() not in prefixes:
				prefixes.append(line[4].strip())
		else:
			if line[4].strip() not in suffixes:
				suffixes.append(line[4].strip())
except:
	print('Download derivational morphemes from MorphyNet to use affix_mod: https://github.com/kbatsuren/MorphyNet/blob/main/eng/eng.derivational.v1.tsv')

def affix_shift(affix, type_, original_word): # helper function
	if type_=='prefix':
		i = prefixes.index(affix)
		if original_word[0].isupper():
			affix.capitalize()
		return(prefixes[i+1] if i+1<len(prefixes) else type[0])
	else:
		i = suffixes.index(affix)
		if original_word[0].isupper():
			affix.capitalize()
		return(suffixes[i+1] if i+1<len(suffixes) else type[0])

def affix_mod(word, pos_tag):
	if word.isalpha() is False:
		print('Word does not satisfy isalpha. Run at your own risk.')
	original_word = word
	word = word.lower()
	if word in morpho_words.keys() and pos_tag[0]==morpho_words[word][2]:
		affix = morpho_words[word][0]
		type_ = morpho_words[word][1]
		if type_=='prefix':
			word = original_word
			word = affix_shift(affix, type_, original_word) + word[len(affix):]
		else:
			word = original_word
			word = word[:-len(affix)] + affix_shift(affix, type_, original_word)
	if original_word[0].isupper():
		word = word.capitalize()
	return word

def hyp_mod(word, pos_tag):
	if word.isalpha() is False:
		print('Word does not satisfy isalpha. Run at your own risk.')
	original_word = word
	word = word.lower()
	synsets = []
	if pos_tag in ['NN', 'NNP', 'NNPS', 'NNS']:
		synsets = wordnet.synsets(word, pos=wordnet.NOUN)
	elif pos_tag in ['RB', 'RBR', 'RBS']:
		synsets = wordnet.synsets(word, pos=wordnet.ADV)
	elif pos_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
		synsets = wordnet.synsets(word, pos=wordnet.VERB)
	elif pos_tag in ['JJ', 'JJR', 'JJS']:
		synsets = wordnet.synsets(word, pos=wordnet.ADJ)
	if len(synsets) > 0:
		hyponyms = synsets[0].hyponyms()
		if len(hyponyms) > 0:
			flag = True
			for hyponym in hyponyms:
				hyponym = hyponym.lemmas()[0].name()
				if '_' not in hyponym:
					word = hyponym
					break
	if original_word[0].isupper():
		word = word.capitalize()
	return word

def ant_mod(word, pos_tag):
	if word.isalpha() is False:
		print('Word does not satisfy isalpha. Run at your own risk.')
	original_word = word
	word = word.lower()
	synsets = []
	if pos_tag in ['NN', 'NNP', 'NNPS', 'NNS']:
		synsets = wordnet.synsets(word, pos=wordnet.NOUN)
	elif pos_tag in ['RB', 'RBR', 'RBS']:
		synsets = wordnet.synsets(word, pos=wordnet.ADV)
	elif pos_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
		synsets = wordnet.synsets(word, pos=wordnet.VERB)
	elif pos_tag in ['JJ', 'JJR', 'JJS']:
		synsets = wordnet.synsets(word, pos=wordnet.ADJ)
	if len(synsets) > 0:
		for synset in synsets:
			for lemma in synset.lemmas():
				if lemma.antonyms():
					for antonym in lemma.antonyms():
						antonym = antonym.name()
						if '_' not in antonym:
							word = antonym
							break
	if original_word[0].isupper():
		word = word.capitalize()
	return word
