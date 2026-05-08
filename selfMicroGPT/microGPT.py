import json
import numpy as np
class Tokenizer():

    with open('./token_2_id.json','r') as f:
        self.token_2_id = json.load(f)
    with open('./id_2_token.json','r') as f:
        self.id_2_token = json.load(f)
    
    def __init__(self,):
        pass

    def encode(self,sentence:list):
        for word in sentence:
            if word not in self.token_2_id:
                self.token_2_id.insert({word:len(self.self.token_2_id)})
                self.id_2_token.insert({str(len(self.self.token_2_id)):word})
            else:
                pass
        with open('./token_2_id.json','w') as f:
            json.dump(self.token_2_id,f)
        with open('./id_2_token.json','w') as f:
            json.dump(self.id_2_token,f)
    
    @staticmethod
    def get_vocab_token_2_id():
        return self.token_2_id

    @staticmethod
    def get_vocab_id_2_token():
        return self.id_2_token

    def get_id(self,sentence:list):
        id_list = []
        for word in sentence:
            if word not in self.token_2_id:
                id_list.append(self.token_2_id['_out_'])
            else:
                id_list.append(self.token_2_id[word])

        return id_list
    
    def get_token(self,sentence:list):
        token_list = []
        for word in sentence:
            if word not in self.id_2_token:
                token_list.append(-1)
            else:
                token_list.append(self.id_2_token[word])

        return token_list

    
class Embedding():
    def __init__(self,vocab_len,d_model=16):
        self.vocab_len = vocab_len
        self.d_model = d_model
        self.embedding_matrix = np.random.randn(self.vocab_len,self.d_model)

    def forward(self,x):
        x = np.dot(x,self.embedding_matrix)
        

