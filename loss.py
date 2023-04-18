import torch
import torch.nn as nn
import torch.nn.functional as F


class L2_Loss(nn.Module):
    def __init__(self, gamma=3):
        super(L2_Loss, self).__init__()
        self.gamma = gamma  # margin based loss

    def dis(self, x, y):
        return torch.sum(torch.abs(x - y), dim=-1)
        # return torch.norm(x,y)

    def forward(self, x1, x2, train_set, train_batch):
    # def forward(self, x1, x2, train_set, train_batch):
        x1_train, x2_train = x1[train_set[:, 0]], x2[train_set[:, 1]]
        x1_neg1 = x1[train_batch[0].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
        x1_neg2 = x2[train_batch[1].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg1 = x2[train_batch[2].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg2 = x1[train_batch[3].view(-1)].reshape(-1, train_set.size(0), x1.size(1))

        dis_x1_x2 = self.dis(x1_train, x2_train)

        loss11 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x1_train, x1_neg1)))
        loss12 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x1_train, x1_neg2)))
        loss21 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x2_train, x2_neg1)))
        loss22 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x2_train, x2_neg2)))
        # loss1_1=torch.mean(F.relu(self.gamma - self.dis(x1_train, x1_neg1)))
        # loss1_2=torch.mean(F.relu(self.gamma - self.dis(x1_train, x1_neg2)))
        # loss2_1=torch.mean(F.relu(self.gamma - self.dis(x2_train, x2_neg1)))
        # loss2_2=torch.mean(F.relu(self.gamma - self.dis(x2_train, x2_neg2)))
        loss = (loss11 + loss12+ loss21 + loss22) / 4

        # # TransE过程损失
        # x1_negh = x1[train_batch1[0].view(-1)].reshape(-1, transindex1.size(0), x1.size(1))
        # x1_negt = x1[train_batch1[1].view(-1)].reshape(-1, transindex1.size(0), x1.size(1))
        # x2_negh = x2[train_batch2[0].view(-1)].reshape(-1, transindex2.size(0), x2.size(1))
        # x2_negt = x2[train_batch2[1].view(-1)].reshape(-1, transindex2.size(0), x2.size(1))
        # losst1= torch.mean(F.relu(self.gamma+self.dis(x1[edge_index1[0][transindex1]]+r1[transindex1], x1[edge_index1[1][transindex1]])
        #                           - self.dis(x1_negh+r1[transindex1], x1[edge_index1[1][transindex1]])))
        # losst2= torch.mean(F.relu(self.gamma+self.dis(x1[edge_index1[0][transindex1]]+r1[transindex1], x1[edge_index1[1][transindex1]])
        #                           - self.dis(x1[edge_index1[0][transindex1]]+r1[transindex1], x1_negt)))
        # loss1_1=torch.mean(F.relu(self.gamma - self.dis(x1[edge_index1[0][transindex1]]+r1[transindex1], x1_negt)))
        #
        # loss1_2 = torch.mean(F.relu(self.gamma - self.dis(x1_negh+r1[transindex1], x1[edge_index1[1][transindex1]])))
        # # print(x2_negh.size(0),x2_negh.size(1),x2_negh.size(2))
        # # print(r2[transindex2].size(0),r2[transindex2].size(1))
        # # print(x2[edge_index2[1][transindex2]].size(0),x2[edge_index2[1][transindex2]].size(1))
        # losst3 = torch.sum(F.relu(
        #     self.gamma + self.dis(x2[edge_index2[0][transindex2]] + r2[transindex2] , x2[edge_index2[1][transindex2]])
        #     - self.dis(x2_negh + r2[transindex2] , x2[edge_index2[1][transindex2]])))
        # losst4 = torch.mean(F.relu(
        #     self.gamma + self.dis(x2[edge_index2[0][transindex2]] + r2[transindex2] , x2[edge_index2[1][transindex2]])
        #     - self.dis(x2[edge_index2[0][transindex2]] + r2[transindex2] , x2_negt)))
        # loss2_1 =  torch.mean(F.relu(self.gamma - self.dis(x2_negh + r2[transindex2] , x2[edge_index2[1][transindex2]])))
        # loss2_2 =  torch.mean(F.relu(self.gamma - self.dis(x2[edge_index2[0][transindex2]] + r2[transindex2] , x2_negt)))
        # print(losst1,losst2,losst3,losst4)
        # loss=loss+losst1+losst2+losst3+losst4+loss1_1+loss2_1+loss1_2+loss2_2
        return loss


class L1_Loss(nn.Module):
    def __init__(self, gamma=3):
        super(L1_Loss, self).__init__()
        self.gamma = gamma  # margin based loss

    def dis(self, x, y):
        return torch.sum(torch.abs(x - y), dim=-1)
        # return torch.nonzero()

    def forward(self, x1, x2, train_set, train_batch):
        x1_train, x2_train = x1[train_set[:, 0]], x2[train_set[:, 1]]
        x1_neg1 = x1[train_batch[0].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
        x1_neg2 = x2[train_batch[1].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg1 = x2[train_batch[2].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg2 = x1[train_batch[3].view(-1)].reshape(-1, train_set.size(0), x1.size(1))

        dis_x1_x2 = self.dis(x1_train, x2_train)
        loss11 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x1_train, x1_neg1)))
        loss12 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x1_train, x1_neg2)))  # 是否有必要
        loss21 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x2_train, x2_neg1)))  # 是否有必要
        loss22 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x2_train, x2_neg2)))
        loss = (loss11 + loss12 + loss21 + loss22) / 4
        return loss
