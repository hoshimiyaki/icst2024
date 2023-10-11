import os
import sys
import torch
import torch.nn.functional as F
import time
import pickle
import random
from model.dataset import *
from model.model_new import *
import argparse
import tqdm
from sklearn.metrics import accuracy_score,classification_report,f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def seed_torch(seed=81402):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_batch(x1,x2, idx, bs):
    code_batch = x1[idx: idx+bs]
    matrix_batch=x2[idx: idx+bs]
    place_batch=[]
    single=[17, 59, 15, 69, 44, 47, 64] 
    for code in code_batch:
        for row in code:
            if 27 in row or row[:7]==single:
                place_batch.append(code.index(row))
                break
    return  torch.LongTensor(code_batch),torch.FloatTensor(matrix_batch),torch.LongTensor(place_batch)

def print_parameter_statistics(model):
    total_num = [p.numel() for p in model.parameters()]
    trainable_num = [p.numel() for p in model.parameters() if p.requires_grad]
    print("Total parameters: {}".format(sum(total_num)))
    print("Trainable parameters: {}".format(sum(trainable_num)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--task', type=int,required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--model',type=str,default='Conpre')
    parser.add_argument('--dataset',type=str,default='squeeze_40_20_new')
    parser.add_argument('--seed', type=int, default=81402)
    parser.add_argument('--hidden-dim', type=int, default=50)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--loss',type=str,default='hinge')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--vector-file', type=str,  default='vectors_64')
    parser.add_argument('--cnn-channel', type=int, default=2)

    args = parser.parse_args()
    print(args)

    task_id = args.task
    root = "../data/cooked/{}/".format(task_id)

    train_x_pos = root + 'train/x_pos_'+args.dataset+'.pkl'
    train_x_neg = root + 'train/x_neg_'+args.dataset+'.pkl'
    train_x=root + 'train/x_'+args.dataset+'.pkl'
    train_y=root + 'train/y.pkl'
    test_x_data = root + 'test/x_'+args.dataset+'.pkl'
    test_y_data = root + 'test/y.pkl'
    val_x_data = root + 'val/x_'+args.dataset+'.pkl'
    val_y_data = root + 'val/y.pkl'
    vector_path="../data/cooked/w2v/"+args.vector_file+".pkl"
    distinct_label="{}_{}_{}_{}_{}_{}_{}_{}".format(task_id,args.model,args.dataset,args.batch,args.epoch,args.layer_num,str(args.lr).replace('.','-'),str(args.cnn_channel))
    project_label="{}_{}_{}_{}_{}_{}_{}".format(args.model,args.dataset,args.batch,args.epoch,args.layer_num,str(args.lr).replace('.','-'),str(args.cnn_channel))
    log_path="../log/"+distinct_label+".log"

    with open(vector_path,"rb") as f:
        pretrain_vector=pickle.load(f)
    f_log=open(log_path,"w",encoding="utf8")
    f_log.write(str(args.__dict__))
    f_log.write("\n")
    LOSS=args.loss
    SEED=args.seed
    model_name=args.model
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch
    USE_GPU = True
    HIDDEN_DIM = args.hidden_dim
    LAYER_NUM=args.layer_num
    CNN_CHANNEL=args.cnn_channel
    try:
        L_S=int(args.dataset.split('_')[2])
        L_B=int(args.dataset.split('_')[1])+2
    except Exception:
        L_S=20
        L_B=40
    LABELS = 2
    VOCAB_SIZE = pretrain_vector.shape[0]
    EMBEDDING_DIM = pretrain_vector.shape[1]

    seed_torch(SEED)
    train_dataset=FusionDataset(train_x_pos,train_x_neg,args.seed,True)
    val_dataset=FusionDataset(val_x_data,val_y_data,0,False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=gen_pad_collate,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=gen_pad_collate,
        num_workers=0,
        pin_memory=True,
    )


    if model_name=='Conpre':
        model=ConprehenLSTM(EMBEDDING_DIM,HIDDEN_DIM,VOCAB_SIZE,LABELS,LAYER_NUM,CNN_CHANNEL,L_B,L_S,pretrain_vector)
    else:
        raise Exception("unknown model name")

    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    print_parameter_statistics(model)
    train_loss_ = []
    val_loss_ = []
    val_acc=[]

    best_result = 1e-3
    print('Start training...')
    for epoch in range(EPOCHS):
        start_time = time.time()

        # training phase
        model.train()
        total_loss = 0.0
        total = 0

        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}",disable=True)
        cur_iter=1
        for batch in pbar:
            model.zero_grad()

            x_pos_c,x_pos_s,x_neg_c,x_neg_s,x_p,y_p=batch
            if USE_GPU:
                x_pos_c, x_pos_s,x_neg_c,x_neg_s,x_p,y_p = x_pos_c.cuda(), x_pos_s.cuda(),x_neg_c.cuda(), x_neg_s.cuda(),x_p.cuda(),y_p.cuda()
            output_pos = model(x_pos_c,x_pos_s,x_p)
            output_neg = model(x_neg_c,x_neg_s,y_p)
            output_pos = torch.softmax(output_pos, dim=-1)[:, 0]
            output_neg = torch.softmax(output_neg, dim=-1)[:, 0]
            loss = torch.max(torch.zeros_like(output_pos), 0.1 - (output_pos - output_neg)).mean()
            total += len(x_pos_c)
            total_loss += loss.item() * len(x_pos_c) 
            
            pbar.set_description(f"epoch {epoch} loss {(total_loss / total):.4f}")
            loss.backward()
            optimizer.step()
        train_loss_.append(total_loss / total)

        # validation phase
        model.eval()
        total_loss = 0.0
        total_acc=0.0
        total = 0
        total_y=[]
        total_pre=[]
        i = 0
        for batch in val_loader:
            x_c,x_s,y,x_p=batch
            if USE_GPU:
                x_c, x_s,y,x_p= x_c.cuda(), x_s.cuda(),y.cuda(),x_p.cuda()
            output = model(x_c,x_s,x_p)
            output_s = torch.softmax(output, dim=-1)[:, 0]
            loss = F.binary_cross_entropy(output_s, y)

            pred=[]
            for i in output.cpu().detach().numpy():
                if i[1]>0.5:
                    pred.append(1)
                else:
                    pred.append(0)
                    
            total_y += y.cpu().detach().numpy().tolist()
            total_pre += pred
            total += len(x_c)
            total_loss += loss.item() * len(x_c)
    
        val_loss_.append(total_loss / total)
        val_acc.append(f1_score(total_y, total_pre,average='macro'))

        if not os.path.exists("./model_save/{}".format(project_label)):
            os.makedirs("./model_save/{}".format(project_label))
        if val_acc[-1] > best_result:
            torch.save(model.state_dict(), "./model_save/{}/model_params_{}.pkl".format(project_label,task_id))
            torch.save(model, "./model_save/{}/model_{}.pkl".format(project_label,task_id))
            best_result = val_acc[-1]
            print("Saving model: epoch_{} ".format(epoch + 1) + "=" * 20)

        end_time = time.time()
        result='[Epoch: %3d/%3d]\tTraining Loss: %.10f\tValidation Loss: %.10f\tValidation Acc: %.10f\tTime Cost: %.3f s'% (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch], val_acc[epoch],end_time - start_time)
        f_log.write(result)
        f_log.write("\n")
    # test phase
    model.load_state_dict(torch.load("./model_save/{}/model_params_{}.pkl".format(project_label,task_id)))



    #test
    with open(test_x_data,"rb") as f:
        test_x=pickle.load(f)

    model.eval()
    with open("../data/raw_data/statements.pkl", "rb") as file:
        statements = pickle.load(file)
    with open("../data/raw_data/faulty_statement_set_0.pkl", "rb") as file:
        faulty_statements = pickle.load(file)
    if not os.path.exists("./result/{}".format(project_label)):
        os.makedirs("./result/{}".format(project_label))
    with open("./result/{}/{}.txt".format(project_label,task_id), "w") as result_file:
        for version in test_x:
            result_file.write("==={}\n".format(version))
            x1,x2 = list(zip(*test_x[version]))
            predict_score = torch.empty(0)
            if USE_GPU:
                predict_score = predict_score.cuda()

            i = 0
            while i < len(x1):
                x_c,x_s,x_p=get_batch(x1,x2,i,BATCH_SIZE)
                if USE_GPU:
                    x_c, x_s,x_p= x_c.cuda(), x_s.cuda(),x_p.cuda()
                output = model(x_c,x_s,x_p)
                output_s = torch.softmax(output, dim=-1)[:, 0]
                predict_score = torch.cat((predict_score, output_s))
                
                i += BATCH_SIZE

            predict_score = predict_score.cpu().detach().numpy().tolist()
            sus_lines = statements[version]
            sus_pos_rerank_dict = {}
            for i, line in enumerate(sus_lines):
                sus_pos_rerank_dict[line] = predict_score[i]
            sorted_sus_list = sorted(sus_pos_rerank_dict.items(), key=lambda x: x[1], reverse=True)

            # output new suspicious file generated by our model
            out_suspicious_dir = "./sus_pos_rerank/{}/{}/".format(project_label,version)
            if not os.path.exists(out_suspicious_dir):
                os.makedirs(out_suspicious_dir)
            with open(os.path.join(out_suspicious_dir, "ranking.txt"), "w") as file:
                for (line, score) in sorted_sus_list:
                    file.write("{} {}\n".format(line, score))

            rerank_sus_lines = [line for (line, score) in sorted_sus_list]
            rerank_sus_scores = [float(score) for (line, score) in sorted_sus_list]
            current_faulty_statements = faulty_statements[version]
            for one_position_set in current_faulty_statements:
                current_min_index = 1e8
                for buggy_line in one_position_set:
                    if buggy_line not in rerank_sus_lines:
                        continue
                    buggy_index = len(rerank_sus_scores) - rerank_sus_scores[::-1].index(rerank_sus_scores[rerank_sus_lines.index(buggy_line)])
                    if buggy_index < current_min_index:
                        current_min_index = buggy_index
                if current_min_index == 1e8:
                    continue
                result_file.write(str(current_min_index) + "\n")
