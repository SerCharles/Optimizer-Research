'''
全局参数文件
'''

gpu_id = 0
seed = 1234 #复现实验，写死seed
data_dir = '../data'
result_dir = '../results/'
result_back = '.txt'

cnn_batch_size = 128
cnn_epochs = 200

imdb_epochs = 10
imdb_max_words = 10000  # imdb’s vocab_size 即词汇表大小
imdb_max_len = 200      # max length
imdb_batch_size = 256
imdb_embedding_size = 128   # embedding size
imdb_hidden_size = 128   # lstm hidden size
imdb_dropout = 0.2 
imdb_lr = 0.001

nlp_train_batch_size = 20
nlp_test_batch_size = 10
nlp_learning_rate = 0.001
nlp_epochs = 100
nlp_data_dir = '../data/ptb'

my_num_classes = 20
my_data_dir = "../data/cnn_data/"
my_input_size = 224
my_batch_size = 36
my_epochs = 100
my_lr = 0.0001
my_channel_0 = 96
my_channel_1 = 128
my_channel_2 = 256
my_channel_3 = 8
my_pooling_kernel = 2
my_pooling_stride = 2

