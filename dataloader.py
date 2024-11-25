from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
import config
from Params import args
import pickle
import scipy.sparse as sp
import numpy as np
import torch
import os


import os
import numpy as np
import scipy.sparse as sp

def load_data():
    if args.dataset == 'IJCAI_15' or args.dataset == 'IJCAI':
        predir = 'datasets/IJCAI_15/'
        behaviors = ['click', 'fav', 'cart', 'buy']
        config.target_behavior = 'buy'
        config.user_num = 16159
        config.item_num = 35920
    if args.dataset == 'beibei':
        predir = '/root/autodl-tmp/xiangmli/DualDiffV0.5/datasets/beibei/'
        behaviors = ['pv', 'cart', 'buy']
        config.target_behavior = 'buy'
        config.user_num = 21716
        config.item_num = 7977

    if args.dataset == 'Tmall':
        predir = '/root/autodl-tmp/xiangmli/DualDiffV0.5/datasets/Tmall/'
        behaviors = ['pv', 'fav', 'cart', 'buy']
        config.target_behavior = 'buy'
        config.user_num = 18973
        config.item_num = 31231

    config.behaviors = behaviors

    trnMats = []

    user_set = set()
    item_set = set()
    behavior_data = list()

    for i in range(len(behaviors)):
        behavior = behaviors[i]
        file_path = os.path.join(predir, f'{behavior}.txt')
        data = np.loadtxt(file_path, dtype=int)
        user_ids = data[:, 0]
        item_ids = data[:, 1]
        user_set.update(user_ids)
        item_set.update(item_ids)
        behavior_data.append((user_ids, item_ids))


    user_num = config.user_num
    item_num = config.item_num
    print("User count: {}, Item count: {}".format(user_num, item_num))

    # 创建目标行为的矩阵
    target_user_ids, target_item_ids = behavior_data[-1]

    target_Mat = sp.csr_matrix(
        (np.ones(len(target_user_ids)), (target_user_ids, target_item_ids)),
        shape=(user_num, item_num)
    )
    target_Mat = (target_Mat != 0).astype(int)

    for i in range(len(behaviors)):
        user_ids, item_ids = behavior_data[i]
        mat = sp.csr_matrix(
            (np.ones(len(user_ids)), (user_ids, item_ids)),
            shape=(user_num, item_num)
        )
        mat = (mat != 0).astype(int)

        trnMats.append(matrix_to_tensor(mat))


    return target_Mat, trnMats


def matrix_to_tensor(cur_matrix):
    """
       Convert a sparse matrix (CSR or CSC) into a PyTorch sparse tensor.

       :param cur_matrix: The input sparse matrix in CSR or CSC format. If it's not in COO format,
                          it will be converted to COO format for easy processing.
       :return: A PyTorch sparse tensor representing the input matrix.
       """
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
    values = torch.from_numpy(cur_matrix.data)
    shape = torch.Size(cur_matrix.shape)

    if torch.cuda.is_available():
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(args.device)
    else:
        assert 1 == 2
        return 0



def create_auxiliary_datasets_and_loaders(trnMats, batch_size=args.aux_batch_size, shuffle=True, num_workers=0):
    auxiliary_datasets = []
    auxiliary_dataloaders = []

    # Exclude the target behavior, which is the last element in trnMats
    for aux_mat in trnMats[:-1]:
        # Create dataset for the auxiliary behavior
        auxiliary_dataset = AuxiliaryBehaviorData(aux_mat)
        auxiliary_datasets.append(auxiliary_dataset)

        # Create DataLoader for the auxiliary behavior
        auxiliary_dataloader = DataLoader(
            auxiliary_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        auxiliary_dataloaders.append(auxiliary_dataloader)

    return auxiliary_datasets, auxiliary_dataloaders

class TargetBehaviorData(Dataset):
    def __init__(self, target_mat):
        self.target_mat = target_mat
        self.num_ng = args.num_ng
        self.n_users, self.n_items = target_mat.shape

        # Extract positive interactions
        self.user_item_pairs = np.array(target_mat.nonzero()).T
        self.num_interactions = len(self.user_item_pairs)

        # Prepare for sampling
        self.user_pos_items = self.get_user_pos_items()

        # Create user-item interaction matrix
        self.user_item_matrix = csr_matrix(target_mat)
        self.Graph = None

    def getSparseGraph(self):

        print("loading adjacency matrix")
        dataset_dir = os.path.join('original_graphs', args.dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        save_path = os.path.join(dataset_dir, f"{args.dataset}_s_pre_adj_mat.npz")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(save_path)
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except FileNotFoundError:
                print("generating adjacency matrix")
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.user_item_matrix.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                print("saving norm_adj...")
                sp.save_npz(save_path, norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(args.device)  # Make sure to specify the device if needed
            print("Graph is ready")

        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_user_pos_items(self):
        user_pos_items = {}
        for user, item in self.user_item_pairs:
            if user not in user_pos_items:
                user_pos_items[user] = []
            user_pos_items[user].append(item)
        return user_pos_items

    def get_allPos(self, users):
        posItems = []
        for user in users:
            posItems.append(self.user_item_matrix[user].nonzero()[1])
        return posItems

    def get_allUsers(self, items):

        posUsers = []

        for item in items:
            posUsers.append(self.user_item_matrix.T[item].nonzero()[1])  # 物品矩阵转置查找用户
        return posUsers

    def generate_triplets(self):

        self.user = []
        self.posItem = []
        self.negItem = []

        for user in range(self.n_users):
            pos_items = self.user_pos_items.get(user, [])
            if len(pos_items) == 0:
                continue

            for _ in range(len(pos_items)):

                pos_item = np.random.choice(pos_items)

                while True:
                    neg_item = np.random.randint(0, self.n_items)
                    if neg_item not in pos_items:
                        break


                self.user.append(user)
                self.posItem.append(pos_item)
                self.negItem.append(neg_item)

    def __len__(self):
        return self.num_interactions

    def __getitem__(self, idx):
        return self.user[idx], self.posItem[idx], self.negItem[idx]

class AuxiliaryBehaviorData(Dataset):
    def __init__(self, aux_mat):
        """
        Initialize the auxiliary behavior dataset class.
        :param aux_mat: Input interaction matrix for the auxiliary behavior (scipy.sparse.csr_matrix)
        """
        self.aux_mat = aux_mat  # Store the interaction matrix for the auxiliary behavior
        self.n_users, self.n_items = aux_mat.shape

    def __getitem__(self, index):
        """
        Return the interaction record for the given user index under this auxiliary behavior.
        :param index: Index of the user
        :return: User index, list of interacted items
        """
        user_interactions = self.aux_mat[index].to_dense().flatten()
        return index, user_interactions

    def __len__(self):
        """
        Return the total number of users in the dataset.
        """
        return self.n_users
