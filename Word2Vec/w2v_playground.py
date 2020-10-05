import warnings
from Word2Vec import embedding

warnings.filterwarnings('ignore')

path = '..\\Word2Vec\\embeddings\\word2vec.txt'

print('loading model!')
w2v_model = embedding.load_w2v_model(path, False)

print(w2v_model.wv.similar_by_word('france', topn=5))
print('')
print(w2v_model.wv.similar_by_word('car', topn=5))
print('')
# man is to king as woman is to?
print(w2v_model.wv.most_similar(topn=5, positive=['woman', 'king'], negative=['man']))
print('')
# woman is to female as man is to ?
print(w2v_model.wv.most_similar(topn=5, positive=['man', 'female'], negative=['woman']))
print('')
# paris is to france as london is to?
print(w2v_model.wv.most_similar(topn=5, positive=['london', 'france'], negative=['paris']))
print('')
# black is to criminal as mother is to?
print(w2v_model.wv.most_similar(topn=5, positive=['father', 'daughter'], negative=['mother']))

# top_10_words = w2v_model.wv.index2entity[:100]
# print(top_10_words)
