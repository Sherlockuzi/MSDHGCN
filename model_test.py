import torch
import torch.nn as nn
import torch.nn.functional as F
class MSHGN(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len, out_len,num_nodes, d_model=512, d_layers=2, dropout=0.05,if_spatial=True,if_time =True,
                 device=torch.device('cuda:0'), conv_kernel=[2, 4], num_edge=[20, 40], isometric_kernel=[6, 3]):
        super(MSHGN, self).__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.node_dim = d_model
        #self.bn_start = nn.BatchNorm2d(in_dim, affine=False)
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=d_model, kernel_size=(1, 1))
        self.prediction = Prediction(embedding_size=d_model, dropout=dropout,d_layers=d_layers,  c_out=out_dim, num_edge = num_edge,
                                     conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device)
        self.if_spatial = if_spatial
        self.if_time = if_time
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        
        if self.if_time:
            self.time_emb = nn.Parameter(
                torch.empty(self.seq_len, self.node_dim))
            nn.init.xavier_uniform_(self.time_emb)
    def forward(self, input):
        batch_size, _, num_nodes, time_len = input.shape
        #x = self.bn_start(input)
        x = input
        x = self.start_conv(x)
        #node_emb = []
        if self.if_spatial:
            # expand node embeddings
            x+=self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1).expand(
                -1, -1, -1,time_len)
        
        #time_emb=[]
        if self.if_time:
            x+=self.time_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-2).expand(
                -1, -1, num_nodes,-1)
        #x = torch.cat([x] + node_emb + time_emb, dim=1)
        output = self.prediction(x)#(B,c_out,N,T)
        output = output.permute(0,3,2,1)#(B,T,N,c_out)
        return output


class Prediction(nn.Module):
    def __init__(self, embedding_size=512, dropout=0.05, d_layers=1,  c_out=1, num_edge =[20, 40],
                conv_kernel=[2, 4], isometric_kernel=[6, 3], device='cuda'):
        super(Prediction, self).__init__()

        self.mshg = nn.ModuleList([MSHG(feature_size=embedding_size, dropout= dropout,num_edge=num_edge,
                                                  conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device)
                                      for _ in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, x):
        for mic_layer in self.mshg:
            x = mic_layer(x) #(B,C,N,T)
        return self.projection(x.permute(0,3,2,1)).permute(0,3,2,1)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class MSHG(nn.Module):
    """
    MSHG layer to extract local and global features
    """

    def __init__(self, feature_size=512,  dropout=0.05, num_edge = [20,40], conv_kernel=[2,4],  isometric_kernel=[6, 3], device='cuda'):
        super(MSHG, self).__init__()
        self.conv_kernel = conv_kernel
        self.isometric_kernel = isometric_kernel
        self.num_edge = num_edge
        self.device = device
        self.edge_embed = nn.ParameterList([nn.Parameter(torch.randn(feature_size, i).cuda(), requires_grad=True).cuda() for i in num_edge])
        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=(1,i), padding=(0,0), stride=(1,1))
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=(1,i), padding=(0,0), stride=(1,i))
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose2d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=(1,i), padding=(0,0), stride=(1,i))
                                         for i in conv_kernel])


        self.merge = torch.nn.Conv2d(in_channels=feature_size*len(self.conv_kernel), out_channels=feature_size,
                                     kernel_size=(1, 1))

        self.fnn = FeedForwardNetwork(feature_size, feature_size * 4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(feature_size)

        #self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv2d, conv2d_trans, isometric,edge_embed,i):
        batch, channel, num_nodes, seq_len = input.shape  #(B,C,N,T)
        x = input

        # downsampling convolution
        x1 = self.drop(self.act(conv2d(x))) #(B,C,N,T//i)
        x = x1

        # multi-scale  hypergraph convolution
        x = x.permute(0,3,2,1)
        H =F.relu(torch.matmul(x,edge_embed))
        #torch.save(H, str(i)+".pt")
        Ht = F.softmax(H.transpose(3,2),dim = -1)

        edge_feature = torch.matmul(Ht,x) #(B,T//i,E,C)

        Ho = F.softmax(H,dim=-1)

        x = torch.matmul(Ho,edge_feature)#(B,T//i,N,C) 节点特征


        x = x.permute(0,3,2,1)

        # isometric convolution
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2],x.shape[3] - 1), device=self.device) #shape(B,C,N,T//i-1)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))

        x = x+x1#self.norm((x + x1).permute(0, 3, 2, 1)).permute(0, 3, 2, 1) #(B,C,N,T//i)

        # upsampling convolution
        x = self.drop(self.act(conv2d_trans(x))) #(B,C,N,T)
        x = x[:, :, :, :seq_len]  # truncate

        x = x+input#self.norm(x.permute(0, 3, 2, 1) + input.permute(0, 3, 2, 1)) #(B,T,N,C)

        return x#.permute(0, 3, 2, 1)#(B,C,N,T)

    def forward(self, x):
        # multi-scale
        multi = []

        for i in range(len(self.conv_kernel)):

            x = self.conv_trans_conv(x, self.conv[i], self.conv_trans[i], self.isometric_conv[i],self.edge_embed[i],i)
            multi.append(x)

            # merge

        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i]), dim=1) #(B,self.conv_kernel*C, N,T)
        mg = self.merge(mg).permute(0,3,2,1) #(B,C,N,T)->(B,T,N,C)

        return self.fnn_norm(mg + self.fnn(mg)).permute(0,3,2,1) #(B,T,N,C)->(B,C,N,T)
