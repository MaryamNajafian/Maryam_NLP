from __future__ import barry_as_FLUFL

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
This code explores semantic space analysis
using WORD2VEC and Cosine similarity measures
"""


from __future__ import barry_as_FLUFL

__version__ = '0.1'
__author__ = 'Maryam Najafian'

#%%
from scipy import spatial
import spacy
nlp = spacy.load('en_core_web_lg')

#%%
class Show_Me_My_Dict_Vocab_Info:
    def __init__(self):
        pass
    def __str__(self):
        return f'Size of dict vocabulary: {len(nlp.vocab.vectors)} and dict shape : {nlp.vocab.vectors.shape}'

#%%
class Word2Vec:
    def __init__(self, input_val):
        self.input_val = input_val
    def vectorize(self):
        return nlp(self.input_val).vector
    def find_length(self):
        return nlp(self.input_val).vector.shape
    def __str__(self):
        return f'vector = {self.vectorize()}, length = {self.find_length()}'

#%%
class Cos_Similarity_Between_Tokens():
    def __init__(self,x,y):
        self.x = nlp(x)
        self.y = nlp(y)

    def measure_distance(self):
        for token1 in self.x:
            for token2 in self.y:
                if token1.has_vector and token2.has_vector:
                    print(f'{token1.text}, with normalized representation: {token1.vector_norm}, {token2.text}, with normalized representation: {token2.vector_norm}, have Cosine similarity distance of: {token1.similarity(token2)}')
                elif token1.is_oov:
                    print(f'{token1} is oov! with normalized representation: {token1.vector_norm}')
                elif token2.is_oov:
                    print(f'{token2} is oov! with normalized representation: {token2.vector_norm}')

#%%

class Find_Corresponding_Word_In_Semantic_space:
    """
    finds word analogy
    King - Queenn ~= Prince - Princess
    Walk - Walking ~= Swim - Swimming
    Code Example: Input 3 words and output the 4th word
            distance: 1 - Cosine distance > distance: [close:0 - far:2]
            king - man + woman --> new vector: Queen
             man -> king corresponds to woman -> (?) queen
             king - man + woman = ? (queen)
             word1 - word2 + word3 = word4
    """
    def __init__(self ,word1 ,word2 ,word3):
        self.word1 = word1
        self.word2 = word2
        self.word3 = word3

    def new_word_vector_estimator(self):
        word4 = nlp.vocab[self.word1].vector - nlp.vocab[self.word2].vector + nlp.vocab[self.word3].vector
        return word4

    def cosine_similarity(self,vec1,vec2):
        cosine_similarity_value = 1 - spatial.distance.cosine(vec1, vec2)
        return cosine_similarity_value

    def find_closest_words_to_new_word_vector(self):
        new_vector = self.new_word_vector_estimator()
        computed_similarities = []

        for word in nlp.vocab:
            if word.has_vector:
                if word.is_lower:
                    if word.is_alpha:
                        similarity = self.cosine_similarity(new_vector, word.vector)
                        computed_similarities.append((word,similarity))
        # sorting (word,similarity) tuples in descending order based on their similarity (item[1] value)
        computed_similarities = sorted(computed_similarities, key=lambda item: item[1], reverse=True)
        computed_similarities_without_input_words = [item for item in computed_similarities if
                                                     item[0] not in [self.word1, self, word2, self.word3]]
        return computed_similarities_without_input_words

    def extract_top10_closest_words_to_new_word_vector(self):
        most_similar_words_list = self.find_closest_words_to_new_word_vector()
        top_10_most_similar = [w[0].text for w in most_similar_words_list[:10]]
        return top_10_most_similar

    def __str__(self):
        return str(self.extract_top10_closest_words_to_new_word_vector())

word1 ,word2 ,word3 = 'king', 'man', 'woman'
a = Find_Corresponding_Word_In_Semantic_space(word1 ,word2 ,word3)
print(a)

#%%
def main():

    print(Show_Me_My_Dict_Vocab_Info())

    input_val = u'Lion'
    v=Word2Vec(input_val)
    print(v)

    a=u'fox dog bigoo'
    b=u"pets can't live in a zoo or jungle"
    print(Cos_Similarity_Between_Tokens(a, b).measure_distance())
    print(Cos_Similarity_Between_Tokens(a, a).measure_distance())

#%%
main()
#%%
if __name__ == "__main__":
    print(__doc__)
    main()
