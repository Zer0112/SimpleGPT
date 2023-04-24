import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os

class FaustDataset(Dataset):
    def __init__(self, block_size,start_pos=1016,path="text/faust.txt"):
        super().__init__()
        #test if file already exists
        if not os.path.isfile(path):
            #download the text from the url and save it to a file in ./text
            import requests
            url='https://www.gutenberg.org/files/21000/21000-0.txt'
            r = requests.get(url,timeout=200)
            with open(path, 'wb') as f:
                f.write(r.content)
            print('Downloaded text from', url, 'and saved')
        
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read()
        self.data = data[start_pos:]
        self.block_size = block_size
        
        chars=sorted(list(set(self.data)))
        self.vocab_size=len(chars)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}

    def __len__(self):
        return len(self.data) - self.block_size 

    def __getitem__(self, idx):
        chunk=self.data[idx:idx+self.block_size+1]
        decoded_char=[self.char_to_idx[char] for char in chunk]
        
        x=torch.tensor(decoded_char[:-1])
        y=torch.tensor(decoded_char[1:])
        return x,y
    def to_tokens(self,test_string,device):
        return torch.tensor(
            [self.char_to_idx[char] for char in test_string]).to(device)
        
    def to_string(self,tokens):
        return ''.join([self.idx_to_char[int(token)] for token in tokens])
    
    
#coding this per hand is not the best idea for performance
#this was just for understanding
class Attention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, bias=False,use_fast=True):
        super().__init__()
        self.use_fast=use_fast
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout_value = dropout
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (n_embd // n_head) ** 0.5
        self.q = nn.Linear(n_embd, n_embd, bias=bias)
        self.k = nn.Linear(n_embd, n_embd, bias=bias)
        self.v = nn.Linear(n_embd, n_embd, bias=bias)
        self.o = nn.Linear(n_embd, n_embd, bias=bias)

    def forward(self, x):
        b, t, e, h = *x.shape, self.n_head
        
        q = self.q(x).view(b, t, h, e // h).transpose(1, 2)
        k = self.k(x).view(b, t, h, e // h).transpose(1, 2)
        v = self.v(x).view(b, t, h, e // h).transpose(1, 2)
        if self.use_fast:
            a=F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.dropout_value if self.training else 0,is_causal=True)
        else:
            a = (q @ k.transpose(-2, -1)) * self.scale
            a = a.softmax(dim=-1)
            a = self.dropout(a)
            a =(a @ v).transpose(1, 2).reshape(b, t, e) @ self.o.weight
        a=a.transpose(1, 2).contiguous().view(b, t, e)
        return a

class DecoderBlock(nn.Module):
    def __init__(self, n_embd,n_head,mlp_pdrop,resid_pdrop,attn_pdrop) -> None:
        super().__init__()
        self.ln1=nn.LayerNorm(n_embd)
        self.attention=Attention(n_embd,n_head,dropout=attn_pdrop,bias=False)
        self.ln2=nn.LayerNorm(n_embd)
        self.mlp=nn.Sequential(nn.Linear(n_embd,n_embd*4,bias=False),nn.GELU(approximate="tanh"),nn.Linear(n_embd*4,n_embd,bias=False),nn.Dropout(mlp_pdrop))

    def forward(self,x):
        x=x+self.attention(self.ln1(x))
        x=x+self.mlp(self.ln2(x))
        return x
        

class GPT(pl.LightningModule):
    """GPT model."""
    def __init__(self, vocab_size,block_size, n_embd, n_head=8, n_layer=6,
                 mlp_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,dataset=None):
        super().__init__()
        self.save_hyperparameters()
        if dataset is not None:
            self.to_text=dataset.to_string
            self.vocab_size=dataset.vocab_size
        # Step 1:
        # Embedding of the vocabulary
        self.token_emb_table=nn.Embedding(vocab_size,n_embd)
        #Step 2:
        # Positional embedding
        self.pos_emb_table=nn.Embedding(block_size,n_embd)
        #Step 3:
        # Stacked Decoder blocks
        self.blocks=nn.Sequential(*[DecoderBlock(n_embd,n_head,mlp_pdrop,resid_pdrop,attn_pdrop) for _ in range(n_layer)])
        self.l_norm=nn.LayerNorm(n_embd)
        #Step 4:
        # Final layer
        self.linear_output=nn.Linear(n_embd,vocab_size,bias=False)
        
        
        #TODO maybe later
        # self.apply(self.custom_init_weights)
        
    def custom_init_weights(self,module):
        pass

    def forward(self,idx):
        b_size,t_size=idx.shape
        device=idx.device        
        #encode tokens
        token_embedding=self.token_emb_table(idx)
        #calculate positional embedding
        pos_embedding=self.pos_emb_table(torch.arange(0,t_size,device=device)).unsqueeze(0)
        
        #add token and positional embedding
        #not using concatenation
        x=token_embedding+pos_embedding
        #decoder blocks
        x=self.blocks(x)
        #norm
        x=self.l_norm(x)
        #output
        x = self.linear_output(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-4)
        return [optimizer],[]
    
    def training_step(self, batch, _):
        if self.global_step%5000==0:
            with torch.no_grad():
                self.eval()
                device=self.device
                context = torch.zeros((1, 1), dtype=torch.long,device=device)
                print(f"\n output_step_{self.global_step}",self.to_text(self.sample(context,50)))
        self.train()
        x,y=batch
        logits=self(x)
        if y is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),y.view(-1))
            self.log('train_loss',loss.mean())
            return loss


        return None
    
    @torch.no_grad()
    def sample(self,x,max_token,temp=1.0):
        self.eval()
        for _ in range(max_token):
            #make sure we only look at tokenblock which is smaller than blocksize
            x_cond=x if x.size(1)<=self.hparams.block_size else x[:, -self.hparams.block_size:]
            #model output
            logits=self(x_cond)
            #temperature scaling for experimentation
            logits=logits[:,-1,:]/temp
            #softmax for probability
            prob=F.softmax(logits,dim=-1)
            #sample next token
            ix_next=torch.multinomial(prob,1)
            #append to input
            x = torch.cat([x, ix_next], dim=1)
        return x[0]
    
    def generate_text(self,context,max_token,temp=1.0):
        return self.to_text(self.sample(context,max_token,temp))

        
if __name__ == '__main__':
    data1=FaustDataset(128)
    decoder_block=DecoderBlock(128,8,0.1,0.1,0.1)
    vocable_size=len(data1.idx_to_char)
    gpt1=GPT(vocable_size,256,512,dataset=data1)
    gpt1=torch.compile(gpt1)
    context = torch.zeros((1, 1), dtype=torch.long)
    print(gpt1.sample(context,100))
    print(data1.to_string(gpt1.sample(context,50)))
    print(gpt1.generate_text(context,50))
    trainer=pl.Trainer(max_epochs=1,precision=16,enable_checkpointing=True,max_steps=10000)
    trainer.fit(gpt1,DataLoader(data1))
    #pickle save the model
    with open("gpt1.pkl","wb") as f:
        pickle.dump(gpt1,f)
    gpt1.load_from_checkpoint("./lightning_logs/version_9/checkpoints/epoch=0-step=217608.ckpt")
    
    print(data1.to_string(gpt1.sample(context,50)))
    print(gpt1.generate_text(context=context,max_token=150))
    # print(len(data1.idx_to_char))
    # print(data1.char_to_idx)