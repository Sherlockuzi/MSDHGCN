import random

import torch
import numpy as np
import argparse
import time
import os
import util
from engine import trainer
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='PEMS04',help='datasetname')
parser.add_argument('--device',type=str,default='1',help='graphics card')
parser.add_argument('--data',type=str,default='data/PEMS04',help='data path')
parser.add_argument('--seq_length',type=int,default=12,help='prediction length')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=307,help='number of nodes')
parser.add_argument('--d_model',type=int,default=64,help='')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.003,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--top_k',type=int,default=4,help='top-k sampling')
parser.add_argument('--print_every',type=int,default=100,help='')
parser.add_argument('--save',type=str,default='./garage/pems04',help='save path')
parser.add_argument('--seed',type=int,default=5,help='random seed')



args = parser.parse_args()
print(args)

def setup_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed) # Numpy module
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    #torch.cuda.manual_seed_all(seed) # multi-GPU
def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')

def main():
    setup_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device


    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']



    engine = trainer(args.batch_size, scaler, args.in_dim, args.seq_length, args.num_nodes,args.d_model,
                     args.learning_rate, args.weight_decay, args.clip, args.lr_decay_rate)

    print("start training...",flush=True)
    print_model_parameters(engine.model,only_num=False)

    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        print('***** Epoch: %03d START *****' % i)
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).cuda()
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).cuda()
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        
        engine.scheduler.step()

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).cuda()
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).cuda()
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

        print('***** Epoch: %03d END *****' %i)
        print('\n')

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).cuda()
    realy = realy.transpose(1,3)[:,0,:,:]
    y_pred = []
    y_true = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).cuda()
        testx = testx.transpose(1,3)
        y=torch.tensor(y).cuda()
        label = y[..., :1]  # 真实值
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        y_pred.append(preds.transpose(1,3))#存预测值(b,t,n,d)
        y_true.append(label)#存真实值
        outputs.append(preds.squeeze())
    y_true = torch.cat(y_true,dim=0)
    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
    np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
    np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    print("Best model epoch:", str(bestid+1))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save+"_best_"+str(round(his_loss[bestid],2))+".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
