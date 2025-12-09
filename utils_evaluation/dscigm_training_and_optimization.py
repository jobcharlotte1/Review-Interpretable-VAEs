from asyncore import write
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from utils import *
import numpy as np
import os
import pandas as pd
from customized_linear import CustomizedLinear
import pickle
import torch.nn.functional as F


def con_mask(data_path):

    with open(data_path+"/hierarchical_prior_knowledge_edge.pkl", "rb") as tf:
        data = pickle.load(tf)
    with open(data_path+"/hierarchical_prior_knowledge_layer.pkl", "rb") as tf:
        layer = pickle.load(tf)
    mask = []
    mask.append( torch.zeros(len(layer[6]), len(layer[5]))) 
    mask.append( torch.zeros(len(layer[5]), len(layer[4]))) 
    mask.append( torch.zeros(len(layer[4]), len(layer[3]))) 
    mask.append( torch.zeros(len(layer[3]), len(layer[2]))) 
    new_mask = torch.zeros(len(layer[2]), len(layer[1]))
    link = []
    link.append([])
    link.append([])
    link.append([])
    link.append([])
    for index, value in enumerate(layer[5]):
        for index_2, value_2 in enumerate(layer[6]):
            if value in data[5] and value_2 in data[5][value]:
                mask[0][index_2][index] = 1
                link[0].append((index_2,index))
    for index, value in enumerate(layer[4]):
        for index_2, value_2 in enumerate(layer[5]):
            if value in data[4] and value_2 in data[4][value]:
                mask[1][index_2][index] = 1
                link[1].append((index_2,index))
    for index, value in enumerate(layer[3]):
        for index_2, value_2 in enumerate(layer[4]):
            if value in data[3] and value_2 in data[3][value]:
                mask[2][index_2][index] = 1
                link[2].append((index_2,index))
    for index, value in enumerate(layer[2]):
        for index_2, value_2 in enumerate(layer[3]):
            if value in data[2] and value_2 in data[2][value]:
                mask[3][index_2][index] = 1
                link[3].append((index_2,index))
    for index, value in enumerate(layer[1]):
        for index_2, value_2 in enumerate(layer[2]):
            if value in data[1] and value_2 in data[1][value]:
                new_mask[index_2][index] = 1
    print(sum(sum(mask[0])))
    print(sum(sum(mask[1])))
    print(sum(sum(mask[2])))
    print(sum(sum(mask[3])))
    return mask, layer[6], link, new_mask

class GBN_model(nn.Module):
    def __init__(self, args):
        super(GBN_model, self).__init__()
        self.args = args
        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(10.0).float()
        self.wei_shape = torch.tensor(1e-1).float()

        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size

        self.topic_size = args.topic_size
        self.topic_size = [self.vocab_size] + self.topic_size
        self.layer_num = len(self.topic_size) - 1
        self.embed_size = args.embed_size
        self.bn_layer = nn.ModuleList([nn.BatchNorm1d(self.hidden_size[i]) for i in range(self.layer_num)])
        temp = pd.read_csv(args.dataset_dir+args.dataname+'',sep=',',index_col=0)
        mask, layer6, link, new_mask = con_mask(args.dataset_dir)
        gen_dict={}
        with open(args.dataset_dir+"9606.protein.links.v11.5_700.txt",'r') as r:
            for line in r:
                line = line.strip('\n').split('\t')
                gen_dict[line[0]+line[1]] = 1
                gen_dict[line[1]+line[0]] = 1

        self.columns = temp.columns
        mask.append(torch.zeros(len(list(temp.columns)), 9275))
        self.word_index = []
        for index, value in enumerate(layer6):
            for index_2, value_2 in enumerate(list(temp.columns)):
                if value == value_2 or value + value_2 in gen_dict:
                     mask[4][index_2][index] = 1
        print(sum(sum(mask[4])))
        np.random.seed(0)

        h_encoder = [CustomizedLinear(mask[4])] 
        h_encoder.append(CustomizedLinear(mask[0]))
        h_encoder.append(CustomizedLinear(mask[1]))
        h_encoder.append(CustomizedLinear(mask[2]))
        h_encoder.append(CustomizedLinear(mask[3]))
        h_encoder.append(CustomizedLinear(new_mask))
        self.h_encoder = nn.ModuleList(h_encoder)

        shape_encoder = [nn.Linear(self.topic_size[i + 1] + self.hidden_size[i], self.topic_size[i + 1] ) for i in
                         range(self.layer_num - 1)]

        shape_encoder.append(nn.Linear( self.hidden_size[self.layer_num - 1], self.topic_size[self.layer_num]))
        self.shape_encoder = nn.ModuleList(shape_encoder)

        scale_encoder = [nn.Linear( self.topic_size[i + 1] + self.hidden_size[i],self.topic_size[i + 1] ) for i in
                         range(self.layer_num - 1)]

        scale_encoder.append(nn.Linear(self.hidden_size[self.layer_num - 1], self.topic_size[self.layer_num]))
        self.scale_encoder = nn.ModuleList(scale_encoder)

        decoder = [Conv1DSoftmaxEtm(self.topic_size[i], self.topic_size[i + 1], self.embed_size) for i in
                   range(self.layer_num)]
        self.decoder = nn.ModuleList(decoder)
        self.drop = nn.ModuleList([nn.BatchNorm1d(self.hidden_size[i]) for i in range(self.layer_num)])
        print('------------------------')
        for t in range(self.layer_num - 1):  #saw
            self.decoder[t + 1].rho = self.decoder[t].alphas

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min.to(self.args.device)))

    def reparameterize(self, Wei_shape_res, Wei_scale, Sample_num = 50):
        # sample one
        eps = torch.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1).to(self.args.device)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-self.log_max(1 - eps),  torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        return torch.mean(theta, dim=0, keepdim=False)

    def compute_loss(self, x, re_x):
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1))
        return - likelihood / (x.shape[1])
    
    def compute_mse_loss(self, x, re_x):
        mse = torch.mean((x - re_x) ** 2)
        return mse

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):
        eulergamma = torch.tensor(0.5772, dtype=torch.float32)
        part1 = Gam_shape * self.log_max(Wei_scale) - eulergamma.to(self.args.device) * Gam_shape * Wei_shape_res + self.log_max(Wei_shape_res)
        part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
        part3 = eulergamma.to(self.args.device) + 1 + Gam_shape * self.log_max(Gam_scale) - torch.lgamma(Gam_shape)
        KL = part1 + part2 + part3
        return - torch.sum(KL) / (Wei_scale.shape[1])

    def _ppl(self, x, theta):
        # x: K1 * N
        X1 = self.decoder[0](theta, 0)  # V * N
        X2 = X1 / (X1.sum(0) + real_min)
        ppl = x * torch.log(X2.T + real_min) / -x.sum()
        # ppl = tf.reduce_sum(x * tf.math.log(X2 + real_min)) / tf.reduce_sum(x)
        return ppl.sum().exp()

    def test_ppl(self, x, y):
        _, theta, _, _ = self.forward(x)

        # _, theta_y, _, _ = self.forward_heart(y)
        ppl = self._ppl(y, theta[0])
        # ret_dict.update({"ppl": ppl})
        return ppl

    def forward(self, x):

        hidden_list = [0] * self.layer_num
        theta = [0] * self.layer_num
        gam_scale = [0] * self.layer_num
        k_rec = [0] * self.layer_num
        l = [0] * self.layer_num
        l_tmp = [0] * self.layer_num
        phi_theta = [0] * self.layer_num
        loss = [0] * (self.layer_num + 1)
        likelihood = [0] * (self.layer_num + 1)
        res = [0] * (self.layer_num+1)

        for t in range(self.layer_num):
            if t == 0:
                hidden = F.tanh(self.h_encoder[t](x))#
            else:
                hidden = F.tanh(self.h_encoder[t](hidden_list[t-1]))#
            hidden_list[t] = hidden

        for t in range(self.layer_num-1, -1, -1):
            if t == self.layer_num - 1:
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_list[t])),
                                       self.real_min.to(self.args.device))      # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.to(self.args.device))

                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_list[t])), self.real_min.to(self.args.device))

                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))

                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                phi_theta[t] = self.decoder[t](theta[t], t)

            else:
                temp = phi_theta[t+1].permute(1, 0)
                hidden_phitheta = torch.cat((hidden_list[t], temp), 1)
              #  hidden_phitheta = temp #nocancha
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_phitheta)),
                                       self.real_min.to(self.args.device))  # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.to(self.args.device))

                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_phitheta)), self.real_min.to(self.args.device))
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))

                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                phi_theta[t] = self.decoder[t](theta[t], t)

        for t in range(self.layer_num + 1):
            if t == 0:
                loss[t] = self.compute_loss(x.permute(1, 0), phi_theta[t])
                mse_loss = self.compute_mse_loss(x.permute(1, 0), phi_theta[t])
                likelihood[t] = loss[t]

            elif t == self.layer_num:
                loss[t] = self.KL_GamWei(torch.tensor(1.0, dtype=torch.float32).to(self.args.device), torch.tensor(1.0, dtype=torch.float32).to(self.args.device),
                                             k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))
                likelihood[t] = loss[t]

            else:
                loss[t] = self.KL_GamWei(phi_theta[t], torch.tensor(1.0, dtype=torch.float32).to(self.args.device),
                                         k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))
                likelihood[t] = self.compute_loss(theta[t - 1], phi_theta[t])
        topic_gene = torch.mm(self.decoder[0].rho.detach(), torch.transpose(self.decoder[0].alphas, 0, 1)).detach().to("cpu").numpy()#
        topic_embedding = self.decoder[0].rho.detach().to("cpu").numpy()
        gene_embedding = self.decoder[0].alphas.detach().to("cpu").numpy()
        return phi_theta, theta, loss, likelihood, topic_gene, gene_embedding, topic_embedding, mse_loss
       # return hidden_list, theta, loss, likelihood, topic_gene, gene_embedding,topic_embedding
    

class GBN_trainer:
    def __init__(self, args, voc_path='voc.txt'):
        self.args = args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)
        self.model = GBN_model(args)

        self.optimizer = torch.optim.Adam([{'params': self.model.h_encoder.parameters()},
                                           {'params': self.model.shape_encoder.parameters()},
                                           {'params': self.model.scale_encoder.parameters()}],
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(),
                                                  lr=self.lr, weight_decay=self.weight_decay)
                          
        
    def train(self, train_data_loader,test_loader):
        all_total_loss = []   # sum of losses per epoch
        all_mse_loss = []     # avg mse per epoch
        for epoch in range(self.epochs):

            for t in range(self.layer_num - 1):  #saw
                self.model.decoder[t + 1].rho = self.model.decoder[t].alphas

            self.model.to(self.args.device)

            loss_t = [0] * (self.layer_num + 1)
            likelihood_t = [0] * (self.layer_num + 1)
            num_data = len(train_data_loader)
            mse_epoch = [0] 

            for i, (train_data, _) in enumerate(train_data_loader):
                self.model.h_encoder.train()
                self.model.shape_encoder.train()
                self.model.scale_encoder.train()
                self.model.decoder.eval()

                train_data = torch.tensor(train_data, dtype=torch.float).to(self.args.device)


                re_x, theta, loss_list, likelihood, topic_gene , topic_embedding, gene_embedding, mse_loss = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        (1 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        mse_epoch[t] += mse_loss.item() / num_data

                    elif t < self.layer_num:
                        (1  * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                    else:
                      #  ((epoch+0.0/self.epochs)  * loss_list[t]).backward(retain_graph=True)
                        (self.args.rate*loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.model.h_encoder.eval()
                self.model.shape_encoder.eval()
                self.model.scale_encoder.eval()
                self.model.decoder.train()

                re_x, theta, loss_list, likelihood, topic_gene , topic_embedding, gene_embedding, mse_loss = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        (1 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                        mse_epoch[t] += mse_loss.item() / num_data

                    elif t < self.layer_num:
                        (1  * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                    else:
                        (self.args.rate * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=100, norm_type=2)
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()
            
            epoch_total_loss = sum(loss_t)    # this is already in your print
            epoch_mse_loss = sum(mse_epoch) 

            all_total_loss.append(epoch_total_loss)
            all_mse_loss.append(epoch_mse_loss)


            if epoch % 1 == 0:
                    print('epoch {}|{}, layer {}|{}, loss: {}, mse:{}'.format(epoch, self.epochs, t,self.layer_num,epoch_total_loss, epoch_mse_loss))

            if  epoch == self.epochs-1 or epoch % 100 == 0:# :#
                torch.save(self.model, self.save_path+"d-scIGM_model_"+self.args.dataname.split('.')[0])
                test_likelihood, test_ppl = self.test(train_data_loader,test_loader)
                
        self.plot_losses(all_total_loss, all_mse_loss,
                 save_path=path_dir)


    def test(self, data_loader,test_loader):
        model = torch.load(self.save_path+"d-scIGM_model_"+self.args.dataname.split('.')[0])
        model.eval()
        likelihood_t = 0
        num_data = len(test_loader)
        ppl_total = 0
        test_theta = None
        for i, (train_data, test_data) in enumerate(test_loader):
            train_data = torch.tensor(train_data, dtype = torch.float).to(self.args.device)
            test_data = torch.tensor(test_data, dtype=torch.float).to(self.args.device)

            re_x, theta, loss_list, likelihood, topic_gene , topic_embedding, gene_embedding, mse_loss = model(train_data)
            temp_theta = np.concatenate((theta[0].T.cpu().detach().numpy(), theta[1].T.cpu().detach().numpy()),axis=1)
            temp_theta = np.concatenate((temp_theta, theta[2].T.cpu().detach().numpy()),axis=1)
            temp_theta = np.concatenate((temp_theta, theta[3].T.cpu().detach().numpy()),axis=1)
            temp_theta = np.concatenate((temp_theta, theta[4].T.cpu().detach().numpy()),axis=1)
            temp_theta = np.concatenate((temp_theta, theta[5].T.cpu().detach().numpy()),axis=1)

            if test_theta is None:
                test_theta = temp_theta
            else:
                test_theta = np.concatenate((test_theta, temp_theta))

        pd.DataFrame(test_theta).to_csv(self.args.output_dir+'d-scIGM_'+ self.args.dataname.split('.')[0]+'_embedding'+'.csv') 
        temp = pd.read_csv(self.args.dataset_dir+self.args.dataname+'',sep=',',index_col=0)
        topic_gene = pd.DataFrame(topic_gene)
        topic_gene.index = temp.columns
        topic_gene.to_csv(self.args.output_dir+'d-scIGM_'+self.args.dataname.split('.')[0]+'_tg.csv') 
        topic_label = []
        for index in range(70):
            topic_label.append("topic_"+str(index+1))
        topic_embedding = pd.DataFrame(topic_embedding)   
        topic_embedding.index = topic_label
        topic_embedding.to_csv(self.args.output_dir+'d-scIGM_'+self.args.dataname.split('.')[0]+'_te.csv') 
        gene_embedding = pd.DataFrame(gene_embedding)     
        gene_embedding.index = temp.columns
        gene_embedding.to_csv(self.args.output_dir+'d-scIGM_'+self.args.dataname.split('.')[0]+'_ge.csv') 
        return likelihood_t, ppl_total

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.GBN_models.load_state_dict(checkpoint['state_dict'])

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def vision_phi(self, Phi, outpath='phi_output', top_n=50):
        if self.voc is not None:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            phi = 1
            for num, phi_layer in enumerate(Phi):
                phi = np.dot(phi, phi_layer)
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                f.close()
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
        return top_n_words
    
    def plot_losses(self, total_losses, mse_losses, save_path=path_dir):
        epochs = range(1, len(total_losses) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Total loss
        axes[0].plot(epochs, total_losses, label="Total Loss", color="blue")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Total Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()

        # MSE loss
        axes[1].plot(epochs, mse_losses, label="MSE Loss", color="red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss")
        axes[1].set_title(f"Reconstruction MSE Loss")
        axes[1].legend()
        
        fig.suptitle(f'LR: {self.lr}, Weight decay: {self.weight_decay}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path + f'plot_loss_{self.lr}_{self.weight_decay}.png')
        else:
            plt.show()


def grid_search_GBN(
    param_grid,
    args,
    train_loader,
    test_loader,
    save_path="GBN_gridsearch_results.csv"
):
    """
    param_grid: dict
        Dictionary of hyperparameters to try, e.g.,
        {'lr':[1e-3,5e-4], 'weight_decay':[0,1e-5], 'hidden_size':[[128,64],[256,128]], 'epochs':[50,100]}
    args: Namespace
        Arguments for GBN_model / GBN_trainer
    train_loader, test_loader: DataLoader
        Training and testing data loaders
    save_path: str
        CSV file to store results at each iteration
    """

    results = []

    # Resume from existing CSV if present
    if os.path.exists(save_path):
        print(f"Resuming from existing file: {save_path}")
        df_existing = pd.read_csv(save_path)
        results = df_existing.to_dict(orient="records")

    # All combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        print(f"\n=== Running config: {params} ===")

        # Update args with current hyperparameters
        args.lr = params.get('lr', args.lr)
        args.weight_decay = params.get('weight_decay', args.weight_decay)
        args.hidden_size = params.get('hidden_size', args.hidden_size)
        args.epochs = params.get('epochs', args.epochs)

        # Initialize trainer
        trainer = GBN_trainer(args)

        # Train model
        trainer.train(train_loader, test_loader)

        # Evaluate final MSE / loss on test set
        test_likelihood, _ = trainer.test(train_loader, test_loader)

        # Store results
        result = {
            **params,
            "test_likelihood": test_likelihood
        }
        results.append(result)

        # Save progress after each iteration
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"Saved progress to {save_path}")

    # Return final results DataFrame and best config (lowest test_likelihood)
    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results['test_likelihood'].idxmin()].to_dict()

    return df_results, best_row