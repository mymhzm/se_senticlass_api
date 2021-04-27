# train.py
# 這個 block 是用來訓練模型的
import numpy as np
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import evaluation

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train() # 將 emotion_model 的模式設為 train，這樣 optimizer 就可以更新 emotion_model 的參數
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵, 也就是做这三部softmax -> log -> nllloss
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr) # 將模型的參數給 optimizer，並給予適當的 learning rate
                                                        # optimizer其实的工作就是决定，如何使用已经传递到各parameters的梯度
    total_loss, total_acc, best_loss = 0, 0, 999

    epoch_ev = []
    train_loss_ev = []
    train_ac_ev = []
    test_loss_ev = []
    test_ac_ev = []

    for epoch in range(n_epoch):
        epoch_ev.append(epoch+1)
        total_loss, total_acc = 0, 0
        # training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad() # 好习惯,每次传播梯度前清零，否则会叠加
            outputs = model(inputs)
            outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
            loss = criterion(outputs, labels) # 計算此時模型的 training loss
            loss.backward() # 算 loss 的 gradient，反向传播到各个参数
            optimizer.step() # 更新训练模型的各个参数

            correct = evaluation(outputs, labels) # 计算此时模型的训练集accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	    epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
                    # end='\r'不换行输出 default:end='\n'
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))
        train_loss_ev.append(total_loss/t_batch)
        train_ac_ev.append(total_acc/t_batch*100)

        # validation
        model.eval()
        with torch.no_grad(): # 在使用pytorch时，并不是所有的操作都需要进行计算图的构建（计算过程的构建，以便梯度反向传播等操作）。
            ## 而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():，强制之后的内容不进行计算图构建。
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            test_loss_ev.append(total_loss / v_batch)
            test_ac_ev.append(total_acc / v_batch * 100)
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_loss < best_loss:
                # 如果 validation 的结果优于之前所有的结果，就把当下的模型存起来方便之后预测时使用
                best_loss = total_loss
                #torch.save(emotion_model, "{}/val_acc_{:.3f}.emotion_model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.emotion_model".format(model_dir)) # 请活用torch.save进行数据保存
                print('saving emotion_model with acc {:.3f}'.format(total_acc/v_batch*100))

        print('-----------------------------------------------')
        model.train()

    # 记录下实验数据,以便进行可视化
    ev_path = './evaluation/'
    np.save(ev_path+'epoch_ev', np.array(epoch_ev))
    np.save(ev_path+'train_loss_ev', np.array(train_loss_ev))
    np.save(ev_path+'train_ac_ev', np.array(train_ac_ev))
    np.save(ev_path+'test_loss_ev', np.array(test_loss_ev))
    np.save(ev_path+'test_ac_ev', np.array(test_ac_ev))

