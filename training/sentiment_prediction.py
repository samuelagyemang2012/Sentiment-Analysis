import warnings
from prediction import prediction
from preprocessing import preprocessing
from models.attention_helper import Attention

warnings.filterwarnings('ignore')

"""
Arguments：
model_names: blstm_sent.h5,
             han_sent.h5
             
tokenizer_names: blstm_sent_tokenizer.pickle, 
                 han_sent_tokenizer.pickle
"""
model_name = 'han_sent.h5'
tokenizer_name = 'han_sent_tokenizer.pickle'
################################################################################

han_custom_layers = {"Attention": Attention}
model_path = '..\\models\\saved_models\\' + model_name
tokenizer_path = '..\\tokenizers\\' + tokenizer_name

# Load model
print('loading model!')
if model_name == 'han_sent.h5':
    model = prediction.load_dl_model(model_path, han_custom_layers)
else:
    model = prediction.load_dl_model(model_path, {})

model_input_shape = prediction.get_model_input_shape(model)

if model_name != 'han_sent.h5':
    model_input_length = model_input_shape[1].value
else:
    model_input_length = model_input_shape[2].value
print(model_input_length)

# Load tokenizer
print('loading tokenizer!')
tokenizer = prediction.load_tokenizer(tokenizer_path)
#
# Reviews
user_reviews = [
    "My family and I normally do not watch local movies for the simple reason that they are poorly made, "
    "they lack the depth, and just not worth our time.<br /><br />The trailer of \"Nasaan ka man\" caught my "
    "attention, my daughter in law's and daughter's so we took time out to watch it this afternoon. The movie "
    "exceeded our expectations. The cinematography was very good, the story beautiful and the acting awesome. Jericho "
    "Rosales was really very good, so's Claudine Barretto. The fact that I despised Diether Ocampo proves he was "
    "effective at his role. I have never been this touched, moved and affected by a local movie before. Imagine a "
    "cynic like me dabbing my eyes at the end of the movie? Congratulations to Star Cinema!! Way to go, Jericho and "
    "Claudine!!",
    "Believe it or not, this was at one time the worst movie I had ever seen. Since that time, I have seen many more "
    "movies that are worse (how is it possible??) Therefore, to be fair, I had to give this movie a 2 out of 10. But "
    "it was a tough call.",
    "This was the worst movie in my entire life.",
    "I recommend this movie to the entire universe.",
    "Even aside from its value as pure entertainment, this movie can serve as a primer to young adults about the "
    "tensions in the Middle East.",
    "Derivative, uneven, clumsy and absurdly sexist.",
    "I've seen it twice and it's even better the second time.",
    "This was a very good movie. I really enjoyed it.",
    "Horrible waste of time. I do not recommend this movie to anyone",
    "I would watch this movie again."]

# Clean reviews
print('cleaning reviews!')
clean_reviews = preprocessing.clean_reviews(user_reviews)

# Tokenize reviews
print('tokenizing reviews!')
tokenized_reviews = prediction.tokenize_reviews(clean_reviews, tokenizer)

# Pad sequence
print('padding_reviews!')
if model_name == 'han_sent.h5':
    padded_reviews = preprocessing.pad_reviews(tokenized_reviews, model_input_length * model_input_shape[1].value,
                                               'post')
    padded_reviews = preprocessing.reshape_inputs(padded_reviews, model_input_shape[1].value, model_input_length)
else:
    padded_reviews = preprocessing.pad_reviews(tokenized_reviews, model_input_length, 'post')

# Make predictions
print('predicting!')
prediction.predict(model, padded_reviews, user_reviews)
