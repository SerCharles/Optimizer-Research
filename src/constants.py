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

imdb_epochs = 40
rnn_max_words = 10000  # imdb’s vocab_size 即词汇表大小
rnn_max_len = 200      # max length
rnn_batch_size = 256
rnn_embedding_size = 128   # embedding size
rnn_hidden_size = 128   # lstm hidden size
rnn_dropout = 0.2 