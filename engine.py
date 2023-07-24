import torch.optim as optim
import util
from model_test import *
class trainer():
    def __init__(self, batch_size, scaler, in_dim, seq_length, num_nodes,d_model, lrate, wdecay,
                 clip=3, lr_de_rate=0.97):

        self.model = MSHGN(in_dim, out_dim=in_dim, seq_len=seq_length, out_len=seq_length,num_nodes=num_nodes, d_model=d_model, d_layers=1, dropout=0.00,
                 device=torch.device('cuda:0'), conv_kernel=[1,2,6,12], num_edge=[30,30,30,30], isometric_kernel=[12,6,2,1])

        self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = clip
        lr_decay_rate=lr_de_rate
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()


        #print(input.shape)
        output = self.model(input)
        output = output.transpose(1,3) #(2,1,207,12)
        real = torch.unsqueeze(real_val,dim=1) #(2,1,207,12)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()

        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
