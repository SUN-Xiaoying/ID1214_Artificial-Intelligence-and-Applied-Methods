import paddle

embedding_size = 128
hidden_size = 256
num_encoder_lstm_layers = 1
en_vocab_size = 2772 #len(list(en_vocab))
cn_vocab_size = 2153 #len(list(cn_vocab))


# encoder: simply learn representation of source sentence
class Encoder(paddle.nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb = paddle.nn.Embedding(en_vocab_size, embedding_size,)
        self.lstm = paddle.nn.LSTM(input_size=embedding_size, 
                                   hidden_size=hidden_size, 
                                   num_layers=num_encoder_lstm_layers)

    def forward(self, x):
        x = self.emb(x)
        x, (_, _) = self.lstm(x)
        return x