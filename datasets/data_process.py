import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# 定义数据集信息
datasets = {
    'IJCAI': {
        'predir': 'IJCAI_15/',
        'behaviors': ['click', 'fav', 'cart', 'buy']
    },
    'beibei': {
        'predir': 'beibei/',
        'behaviors': ['pv', 'cart', 'buy']
    },
    'Tmall': {
        'predir': 'Tmall/',
        'behaviors': ['pv', 'fav', 'cart', 'buy']
    }
}


def process_dataset(dataset_name):
    dataset_info = datasets[dataset_name]
    predir = dataset_info['predir']
    behaviors = dataset_info['behaviors']
    target_behavior = 'buy'  # 目标行为是 'buy'

    # 定义训练和测试文件前缀
    tr_file = os.path.join(predir, 'trn_')
    tst_file = os.path.join(predir, 'tst_int')  # 测试文件名为 'tst_int'

    # 用于存储每个行为的数据
    behavior_data = {}
    users_to_keep_set = set()

    # 定义用户和物品ID集合用于统计
    user_ids = set()
    item_ids = set()

    # 加载目标行为的数据
    trn_path = tr_file + target_behavior
    with open(trn_path, 'rb') as f:
        trn_data = pickle.load(f)
        print(f"Loaded {target_behavior} training data of type: {type(trn_data)}")
        trn_interactions = data_to_interactions(trn_data)

    tst_path = tst_file
    with open(tst_path, 'rb') as f:
        tst_data = pickle.load(f)
        print(f"Loaded {target_behavior} testing data of type: {type(tst_data)}")
        tst_interactions = data_to_interactions(tst_data)

    # 合并训练和测试交互
    total_interactions = trn_interactions + tst_interactions
    total_interactions.sort(key=lambda x: (x[0], x[2]))  # 按用户ID和时间戳排序

    # 转换为DataFrame
    df = pd.DataFrame(total_interactions, columns=['user_id', 'item_id', 'timestamp'])

    # 筛选出目标行为交互次数大于等于5的用户
    user_interaction_counts = df.groupby('user_id').size()
    users_to_keep = user_interaction_counts[user_interaction_counts >= 5].index
    users_to_keep_set = set(users_to_keep)

    # 过滤掉不满足条件的用户
    df = df[df['user_id'].isin(users_to_keep_set)]
    behavior_data[target_behavior] = df

    # 将用户和物品ID添加到集合
    user_ids.update(df['user_id'].unique())
    item_ids.update(df['item_id'].unique())

    # 加载并处理辅助行为的数据
    for behavior in behaviors:
        if behavior == target_behavior:
            continue  # 已经处理过目标行为，跳过

        trn_path = tr_file + behavior
        with open(trn_path, 'rb') as f:
            trn_data = pickle.load(f)
            print(f"Loaded {behavior} training data of type: {type(trn_data)}")
            trn_interactions = data_to_interactions(trn_data)

        df_aux = pd.DataFrame(trn_interactions, columns=['user_id', 'item_id', 'timestamp'])
        df_aux = df_aux[df_aux['user_id'].isin(users_to_keep_set)]

        # 将用户和物品ID添加到集合
        user_ids.update(df_aux['user_id'].unique())
        item_ids.update(df_aux['item_id'].unique())

        behavior_data[behavior] = df_aux

    # 重新组织用户ID
    all_users = sorted(users_to_keep_set)
    user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_users)}

    # 更新所有行为的数据中的用户ID
    for behavior in behaviors:
        df = behavior_data[behavior]
        df['user_id'] = df['user_id'].map(user_id_mapping)
        behavior_data[behavior] = df

    # 重新编码后统计用户数和物品数
    num_users = len(user_id_mapping)  # 重新编码后的用户数
    num_items = len(item_ids)  # 重新编码后的物品数

    print(f"处理后的用户数: {num_users}, 物品数: {num_items}")

    max_user_id = max(user_id_mapping.values())  # 处理后用户ID的最大值
    max_item_id = max(item_ids)  # 物品ID的最大值

    print(f"处理后的用户ID最大值: {max_user_id}, 物品ID最大值: {max_item_id}")

    # 分割目标行为的数据为训练、验证和测试集
    target_df = behavior_data[target_behavior]
    target_df = target_df.sort_values(['user_id', 'timestamp'])

    train_list = []
    val_list = []
    test_list = []

    for user_id, user_df in target_df.groupby('user_id'):
        interactions = user_df[['user_id', 'item_id']].values
        if len(interactions) >= 2:
            train_interactions = interactions[:-2]
            val_interaction = interactions[-2]
            test_interaction = interactions[-1]
        else:
            train_interactions = interactions[:-1]
            val_interaction = interactions[-1]
            test_interaction = None

        train_list.extend(train_interactions)
        val_list.append(val_interaction)
        if test_interaction is not None:
            test_list.append(test_interaction)

    train_df = pd.DataFrame(train_list, columns=['user_id', 'item_id'])
    val_df = pd.DataFrame(val_list, columns=['user_id', 'item_id'])
    test_df = pd.DataFrame(test_list, columns=['user_id', 'item_id'])

    # 保存辅助行为的数据
    for behavior in behaviors:
        if behavior != target_behavior:
            aux_df = behavior_data[behavior]
            aux_df[['user_id', 'item_id']].to_csv(os.path.join(predir, f'{behavior}.txt'), index=False, header=False,
                                                  sep=' ')

    # 保存目标行为的训练、验证和测试集
    train_df.to_csv(os.path.join(predir, f'{target_behavior}.txt'), index=False, header=False, sep=' ')
    val_df.to_csv(os.path.join(predir, 'validation.txt'), index=False, header=False, sep=' ')
    test_df.to_csv(os.path.join(predir, 'test.txt'), index=False, header=False, sep=' ')


def data_to_interactions(data):
    interactions = []
    if isinstance(data, csr_matrix):
        for user_id in range(data.shape[0]):
            row = data.getrow(user_id)
            item_indices = row.indices
            for idx, item_id in enumerate(item_indices):
                timestamp = idx
                interactions.append((user_id, item_id, timestamp))
    elif isinstance(data, list):
        for user_id, items in enumerate(data):
            if items is None:
                continue
            elif isinstance(items, (list, tuple, np.ndarray)):
                for idx, item in enumerate(items):
                    if isinstance(item, tuple) and len(item) == 2:
                        item_id, timestamp = item
                    else:
                        item_id = item
                        timestamp = idx
                    interactions.append((user_id, item_id, timestamp))
            else:
                item_id = items
                timestamp = 0
                interactions.append((user_id, item_id, timestamp))
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    return interactions


# 处理特定的数据集，例如 'IJCAI'
process_dataset('Tmall')
