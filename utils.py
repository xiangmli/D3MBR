import torch
from torch import nn, optim
import config
import numpy as np
from Params import args
from torch import log
from time import time
from model import LightGCN
import random
import os
import scipy.sparse as sp
import pickle
from dataloader import TargetBehaviorData
class BPRLoss:
    def __init__(self,
                 recmodel):
        self.model = recmodel
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.diff_lr = args.diff_lr
        self.alpha = args.alpha
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def call_bpr(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        loss = loss.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def perform_inference_and_save(target_dataset, Recmodel, item_reverse_model, diffusion, auxiliary_dataloaders,
                               auxiliary_datasets, aux_dnn_models, denoised_user_embs, denoised_item_embs):
    """
    Perform inference to get denoised user embeddings and denoised graphs, then save them.

    :param target_dataset: The dataset containing user-item interactions
    :param Recmodel: LightGCN model
    :param item_reverse_model: DNN model for item denoising
    :param diffusion: Diffusion model
    :param auxiliary_dataloaders: List of DataLoader objects for each auxiliary behavior
    :param aux_dnn_models: List of trained DNN models for each auxiliary behavior
    """

    Recmodel.eval()
    item_reverse_model.eval()
    for aux_dnn_model in aux_dnn_models:
        aux_dnn_model.eval()

    # Step 2: Perform inference to obtain denoised graphs for all auxiliary behaviors
    denoised_graphs = inference_denoised_graphs(auxiliary_dataloaders, aux_dnn_models, diffusion, denoised_user_embs)
    print("Denoised graphs obtained for all auxiliary behaviors.")

    processed_graphs = []
    for denoised_graph, aux_dataset in zip(denoised_graphs, auxiliary_datasets):
        processed_graph = process_denoised_graph(denoised_graph, aux_dataset.aux_mat)
        processed_graphs.append(processed_graph)
    denoised_graphs = processed_graphs
    # user_0_interactions_with_values = get_user_0_interactions_with_values(denoised_graphs)

    hyperparam_combo = f"{args.dataset}ld{args.latdim}_nl{args.n_layer}_ep{args.epoch}_lr{args.lr}_dlr{args.diff_lr}_alr{args.aux_lr}_wd{args.weight_decay}_d{args.dims}_ad{args.aux_dims}_" \
                       f"st{args.steps}_ns{args.noise_scale}_nm{args.noise_min}_nx{args.noise_max}_ss{args.sampling_steps}_" \
                       f"rw{args.reweight}_al{args.alpha}_be{args.beta}_rs{args.restrict}_kr{args.keep_rate}_ktr{args.keep_target_rate}_es{args.early_stop}_se{args.stop_epochs}"

    denoised_embeddings_path = os.path.join("results", hyperparam_combo, "denoised_embeddings")
    denoised_graphs_path = os.path.join("results", hyperparam_combo, "denoised_graphs")

    os.makedirs(denoised_embeddings_path, exist_ok=True)
    os.makedirs(denoised_graphs_path, exist_ok=True)

    # Step 3: Save the denoised user embeddings and denoised graphs
    save_denoised_data(denoised_user_embs, denoised_item_embs, denoised_graphs, denoised_embeddings_path, denoised_graphs_path)
    print("Denoised data has been saved.")



def process_denoised_graph(denoised_graph, original_interactions, keep_rate=args.keep_rate, restrict_to_original=args.restrict):
    """
    Process the denoised graph by retaining a fraction of the user interactions based on their scores.
    If `restrict_to_original` is True, the retained items are selected only from the original interactions.
    Otherwise, items are selected from the entire denoised graph.

    :param denoised_graph: The denoised graph with scores for each user-item pair
    :param original_interactions: The original user-item interaction matrix (sparse matrix format)
    :param keep_rate: The fraction of original interactions to keep in the processed graph
    :param restrict_to_original: If True, restrict selection to original interactions only
    :return: A sparse matrix for the denoised graph
    """
    if isinstance(denoised_graph, torch.Tensor):
        denoised_graph = denoised_graph.cpu().numpy()

    sparse_graphs = []

    # Convert original_interactions to CPU sparse matrix if it's a PyTorch sparse tensor
    if isinstance(original_interactions, torch.Tensor):
        original_interactions = original_interactions.to("cpu").coalesce()
        original_interactions = sp.csr_matrix(
            (original_interactions.values().numpy(), original_interactions.indices().numpy()),
            shape=original_interactions.shape
        )

    for user_id, user_ratings in enumerate(denoised_graph):
        if restrict_to_original:
            original_items = original_interactions[user_id].nonzero()[1]  # Use scipy.sparse method to get nonzero indices

            # If there are no original interactions for this user, continue
            if len(original_items) == 0:
                sparse_graphs.append(np.zeros_like(user_ratings))
                continue

            # Filter denoised scores to only include those for items in original interactions
            filtered_scores = user_ratings[original_items]

            # Determine how many items to keep based on the keep_rate
            num_to_keep = int(len(original_items) * keep_rate)

            if num_to_keep > 0:
                # Get the indices of the top scores among the original interactions
                top_indices = np.argsort(-filtered_scores)[:num_to_keep]

                # Map these top indices back to the original item indices
                top_item_indices = original_items[top_indices]
            else:
                top_item_indices = []
        else:
            original_items = original_interactions[user_id].nonzero()[1]  # Use scipy.sparse method to get nonzero indices
            num_to_keep = int(len(original_items) * keep_rate)
            top_item_indices = np.argsort(-user_ratings)[:num_to_keep]

        # Create a binary row indicating which items are retained
        processed_row = np.zeros_like(user_ratings)
        processed_row[top_item_indices] = 1

        sparse_graphs.append(processed_row)

    # Convert the processed dense matrix to a sparse matrix
    final_sparse_graph = sp.csr_matrix(np.vstack(sparse_graphs))

    return final_sparse_graph


def save_denoised_data(denoised_user_embs, item_embs, denoised_graphs, user_emb_path, graph_path_prefix):
    """
    Save the denoised user embeddings, item embeddings, and graphs to files.

    :param denoised_user_embs: Tensor containing the denoised user embeddings
    :param item_embs: Numpy array containing the LightGCN item embeddings
    :param denoised_graphs: List of sparse matrices representing the denoised graphs for each auxiliary behavior
    :param user_emb_path: Path to save the denoised user embeddings
    :param graph_path_prefix: Prefix for the file paths to save the denoised graphs
    """

    denoised_user_embs_np = denoised_user_embs.cpu().numpy()
    user_emb_file_path = os.path.join(user_emb_path, "denoised_user_embs.npz")
    np.savez_compressed(user_emb_file_path, denoised_user_embs=denoised_user_embs_np)
    print(f"Denoised user embeddings saved as .npz at {user_emb_file_path}")

    item_embs = item_embs.cpu().numpy()
    item_emb_file_path = os.path.join(user_emb_path, "item_embs.npz")
    np.savez_compressed(item_emb_file_path, item_embs=item_embs)
    print(f"Item embeddings saved as .npz at {item_emb_file_path}")

    auxiliary_behaviors = [behavior for behavior in config.behaviors if behavior != config.target_behavior]


    for idx, (graph, behavior_name) in enumerate(zip(denoised_graphs, auxiliary_behaviors)):


        if not isinstance(graph, sp.csr_matrix):
            graph = sp.csr_matrix(graph)


        user_ids, item_ids = graph.nonzero()
        data = np.vstack((user_ids, item_ids)).T

        graph_path = os.path.join(graph_path_prefix, f'{behavior_name}.txt')
        np.savetxt(graph_path, data, fmt='%d', delimiter=' ')
        print(f"Denoised graph for behavior '{behavior_name}' saved as txt file at {graph_path}")


def inference_denoised_graphs(auxiliary_dataloaders, aux_dnn_models, diffusion, denoised_user_embs, device=args.device):
    """
    Perform inference to obtain the denoised representations for all auxiliary behaviors.

    :param auxiliary_dataloaders: List of DataLoader objects for each auxiliary behavior
    :param aux_dnn_models: List of trained DNN models for each auxiliary behavior
    :param diffusion: The diffusion model used for sampling
    :param device: The device used for computation (e.g., 'cuda' or 'cpu')
    :return: A list containing denoised representations (graphs) for each auxiliary behavior
    """
    denoised_graphs = []

    for behavior_idx, (aux_loader, aux_dnn_model) in enumerate(zip(auxiliary_dataloaders, aux_dnn_models)):
        aux_dnn_model.eval()  # Set the model to evaluation mode
        denoised_behavior_representation = []

        with torch.no_grad():
            for batch_idx, (user_idx, batch) in enumerate(aux_loader):
                # Move the batch to the specified device
                batch = batch.to(device)
                con_emb = denoised_user_embs[user_idx]
                # Perform the diffusion sampling to get the denoised representation using the trained model
                denoised_batch = diffusion.p_sample(aux_dnn_model, batch, con_emb, args.sampling_steps, args.sampling_noise)

                # Store the denoised representations for this batch
                denoised_behavior_representation.append(denoised_batch.cpu())

        # Combine all batches to form the complete denoised graph representation for this behavior
        denoised_behavior_representation = torch.cat(denoised_behavior_representation, dim=0)
        denoised_graphs.append(denoised_behavior_representation)
        print(f"Completed inference for auxiliary behavior {behavior_idx + 1}")

    return denoised_graphs


def get_denoised_embeddings(target_dataset, Recmodel, user_denoise_model, item_denoise_model, diffusion):

    denoised_user_embs = torch.zeros((config.user_num, args.latdim)).to(args.device)
    denoised_item_embs = torch.zeros((config.item_num, args.latdim)).to(args.device)

    users = list(range(config.user_num))
    for batch_users in minibatch(users, batch_size=args.aux_batch_size):
        allPos = target_dataset.get_allPos(batch_users)
        denoised_batch_users = denoise_user_interactions(batch_users, allPos, Recmodel, user_denoise_model, diffusion)
        for idx, user in enumerate(batch_users):
            denoised_user_embs[user] = denoised_batch_users[idx]

    items = list(range(config.item_num))
    for batch_items in minibatch(items, batch_size=args.aux_batch_size):
        allUsers = target_dataset.get_allUsers(batch_items)
        denoised_batch_items = denoise_item_interactions(batch_items,allUsers, Recmodel, item_denoise_model, diffusion)
        for idx, item in enumerate(batch_items):
            denoised_item_embs[item] = denoised_batch_items[idx]

    return denoised_user_embs, denoised_item_embs


def denoise_user_interactions(batch_users, allPos, Recmodel, user_denoise_model, diffusion):

    users_emb = Recmodel.embedding_user.weight[batch_users]  # (batch_size, emb_size)


    all_items_emb = [torch.mean(Recmodel.embedding_item.weight[allPos[i]], dim=0) for i in range(len(allPos))]
    all_items_emb = torch.stack(all_items_emb)  # (batch_size, emb_size)


    if args.sampling_steps == 0:
        noise_user_emb = users_emb
    else:
        t = torch.tensor([args.sampling_steps - 1] * users_emb.shape[0]).to(users_emb.device)
        noise_user_emb = diffusion.q_sample(users_emb, t)

    indices = list(range(args.steps))[::-1]
    for i in indices:
        t = torch.tensor([i] * noise_user_emb.shape[0]).to(noise_user_emb.device)
        out = diffusion.p_mean_variance(user_denoise_model, noise_user_emb, all_items_emb, t)
        if args.sampling_noise:
            noise = torch.randn_like(noise_user_emb)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(noise_user_emb.shape) - 1)))
            noise_user_emb = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        else:
            noise_user_emb = out["mean"]

    return noise_user_emb


def denoise_item_interactions(batch_items, allUsers, Recmodel, item_denoise_model, diffusion):

    items_emb = Recmodel.embedding_item.weight[batch_items]

    noise_item_emb = items_emb.clone()

    mask = torch.tensor([len(users) > 0 for users in allUsers]).to(items_emb.device)

    if mask.sum() == 0:
        return noise_item_emb

    valid_indices = mask.nonzero(as_tuple=True)[0]

    valid_items_emb = items_emb[valid_indices]
    valid_all_users_emb = [torch.mean(Recmodel.embedding_user.weight[allUsers[i]], dim=0) for i in valid_indices]
    valid_all_users_emb = torch.stack(valid_all_users_emb)  # (valid_batch_size, emb_size)

    if args.sampling_steps != 0:
        t = torch.tensor([args.sampling_steps - 1] * valid_items_emb.shape[0]).to(valid_items_emb.device)
        valid_noise_item_emb = diffusion.q_sample(valid_items_emb, t)
    else:
        valid_noise_item_emb = valid_items_emb

    indices = list(range(args.steps))[::-1]
    for step in indices:
        t = torch.tensor([step] * valid_noise_item_emb.shape[0]).to(valid_noise_item_emb.device)
        out = diffusion.p_mean_variance(item_denoise_model, valid_noise_item_emb, valid_all_users_emb, t)

        if args.sampling_noise:
            noise = torch.randn_like(valid_noise_item_emb)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(valid_noise_item_emb.shape) - 1)))
            valid_noise_item_emb = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        else:
            valid_noise_item_emb = out["mean"]


    noise_item_emb[valid_indices] = valid_noise_item_emb

    return noise_item_emb



def apply_T_noise(cat_emb, diff_model):
    t = torch.tensor([args.steps - 1] * cat_emb.shape[0]).to(cat_emb.device)
    noise = torch.randn_like(cat_emb)
    noise_emb = diff_model.q_sample(cat_emb, t, noise)
    return noise_emb

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', args.batch_size)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(Recmodel, user_reverse_model, item_reverse_model, aux_dnn_models):

    recmodel_params = count_parameters(Recmodel)
    user_reverse_model_params = count_parameters(user_reverse_model)
    item_reverse_model_params = count_parameters(item_reverse_model)
    aux_dnn_models_params = sum(count_parameters(aux_model) for aux_model in aux_dnn_models)

    total_params = recmodel_params + user_reverse_model_params + item_reverse_model_params + aux_dnn_models_params

    print(f"Recmodel Parameters: {recmodel_params:,}")
    print(f"User Reverse Model Parameters: {user_reverse_model_params:,}")
    print(f"Item Reverse Model Parameters: {item_reverse_model_params:,}")
    print(f"Auxiliary DNN Models Parameters (total for all auxiliary behaviors): {aux_dnn_models_params:,}")
    print(f"Total Parameters: {total_params:,}")

def generate_filtered_target_data(target_dataset, user_embs, item_embs, keep_top_x=args.keep_target_rate):

    user_embs_np = user_embs.cpu().detach().numpy()
    item_embs_np = item_embs.cpu().detach().numpy()
    num_users, num_items = user_embs_np.shape[0], item_embs_np.shape[0]


    similarity_matrix = np.dot(user_embs_np, item_embs_np.T)  # 形状为 (num_users, num_items)


    filtered_matrix = sp.lil_matrix((num_users, num_items), dtype=np.int32)


    for user_id in range(num_users):

        original_items = target_dataset.user_item_matrix[user_id].nonzero()[1]

        if len(original_items) == 0:
            continue


        user_similarity = similarity_matrix[user_id][original_items]


        num_to_keep = int(len(original_items) * keep_top_x)


        top_indices = np.argsort(-user_similarity)[:num_to_keep]
        top_items = original_items[top_indices]


        filtered_matrix[user_id, top_items] = 1


    filtered_matrix = filtered_matrix.tocsr()


    new_target_dataset = TargetBehaviorData(filtered_matrix)

    return new_target_dataset
