import dataloader
import torch.utils.data as data
import torch
from torch import nn, optim
import utils
from Params import args
from model import LightGCN, DNN
import diffusion as gd
import config
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
import os

seed = args.seed
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
print("seed set!")
print(torch.cuda.is_available())


n = args.stop_epochs

def main():
    device = torch.device(args.device if args.cuda else "cpu")
    target_Mat, trnMats = dataloader.load_data()
    target_dataset = dataloader.TargetBehaviorData(target_Mat)
    target_loader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    auxiliary_datasets, auxiliary_dataloaders = dataloader.create_auxiliary_datasets_and_loaders(trnMats)
    print("dataset ready!")

    # Initialize tensorboard SummaryWriter
    log_dir_name = f"{args.dataset}ld{args.latdim}_nl{args.n_layer}_ep{args.epoch}_lr{args.lr}_dlr{args.diff_lr}_d{args.dims}_" \
                   f"ad{args.aux_dims}_st{args.steps}_ns{args.noise_scale}_nm{args.noise_min}_nx{args.noise_max}_ss{args.sampling_steps}_" \
                   f"rw{args.reweight}_al{args.alpha}_be{args.beta}"
    writer = SummaryWriter(log_dir=f"runs/{log_dir_name}")

    Recmodel = LightGCN(target_dataset)
    Recmodel = Recmodel.to(device)

    # DNN for target behavior
    out_dims = eval(args.dims) + [args.latdim]
    in_dims = out_dims[::-1]
    in_dims[0] = in_dims[0] * 2

    user_reverse_model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm)
    user_reverse_model = user_reverse_model.to(device)
    item_reverse_model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm)
    item_reverse_model = item_reverse_model.to(device)

    user_reverse_optimizer = optim.Adam(user_reverse_model.parameters(), lr=args.diff_lr, weight_decay=args.weight_decay)
    item_reverse_optimizer = optim.Adam(item_reverse_model.parameters(), lr=args.diff_lr, weight_decay=args.weight_decay)

    # DNN for auxiliary behaviors
    aux_out_dims = eval(args.aux_dims) + [config.item_num]
    aux_in_dims = aux_out_dims[::-1]
    aux_in_dims[0] = aux_in_dims[0] + args.latdim

    aux_dnn_models = []
    auxiliary_optimizers = []
    for i in range(len(auxiliary_datasets)):
        aux_dnn_model = DNN(aux_in_dims, aux_out_dims, args.emb_size, time_type="cat", norm=args.norm)
        aux_dnn_model = aux_dnn_model.to(device)
        aux_dnn_models.append(aux_dnn_model)

        aux_optimizer = optim.AdamW(aux_dnn_model.parameters(), lr=args.aux_lr, weight_decay=args.weight_decay)
        auxiliary_optimizers.append(aux_optimizer)

    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)
    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
                                     args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

    bpr = utils.BPRLoss(Recmodel)

    # count num of all parameters
    utils.count_all_parameters(Recmodel, user_reverse_model, item_reverse_model, aux_dnn_models)

    target_no_improve_epochs = 0
    target_best_user_loss = float('inf')
    target_stop = False

    aux_no_improve_epochs = [0] * len(auxiliary_datasets)
    aux_best_losses = [float('inf')] * len(auxiliary_datasets)
    aux_stop_flags = [False] * len(auxiliary_datasets)

    iter = 0

    print("Start Training!")
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        Recmodel.train()
        user_reverse_model.train()
        item_reverse_model.train()
        target_loader.dataset.generate_triplets()
        aver_loss = 0.
        idx = 0
        for batch_users, batch_pos, batch_neg in target_loader:
            batch_users = batch_users.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)
            loss = bpr.call_bpr(batch_users, batch_pos, batch_neg)
            aver_loss += loss
            idx += 1
            iter += 1
        aver_loss = aver_loss / idx
        epoch_time = time.time() - epoch_start_time
        print(f'EPOCH[{epoch + 1}/{args.epoch}] loss:{aver_loss}, Time taken: {epoch_time:.2f}s')
        # save in tensorboard
        writer.add_scalar('Training/BPR Loss', aver_loss, epoch + 1)

        if (not args.early_stop) or (not target_stop):
            epoch_start_time = time.time()
            with torch.no_grad():
                filtered_target_dataset = utils.generate_filtered_target_data(target_dataset, Recmodel.embedding_user.weight,
                                                                        Recmodel.embedding_item.weight, keep_top_x=args.keep_target_rate)
                filtered_target_loader = data.DataLoader(filtered_target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
                filtered_target_loader.dataset.generate_triplets()
            filter_time = time.time() - epoch_start_time
            print("Filter Time:", filter_time)


            total_user_loss = 0.0
            total_item_loss = 0.0
            batch_count = 0
            epoch_start_time = time.time()
            for batch_users, batch_pos, batch_neg in filtered_target_loader:
                batch_users = batch_users.to(device)
                batch_pos = batch_pos.to(device)
                users_emb = Recmodel.embedding_user(batch_users)
                pos_items_emb = Recmodel.embedding_item(batch_pos)

                user_reverse_optimizer.zero_grad()
                user_loss_dict = diffusion.training_losses(user_reverse_model, users_emb.detach(), pos_items_emb.detach(),
                                                           reweight=args.reweight)
                user_loss = user_loss_dict["loss"].mean()

                user_loss.backward()
                user_reverse_optimizer.step()

                total_user_loss += user_loss.item()

                item_reverse_optimizer.zero_grad()
                item_loss_dict = diffusion.training_losses(item_reverse_model, pos_items_emb.detach(), users_emb.detach(),
                                                           reweight=args.reweight)
                item_loss = item_loss_dict["loss"].mean()
                item_loss.backward()
                item_reverse_optimizer.step()
                total_item_loss += item_loss.item()
                batch_count += 1

            avg_user_loss = total_user_loss / batch_count
            avg_item_loss = total_item_loss / batch_count
            epoch_time = time.time() - epoch_start_time

            print(
                f'EPOCH[{epoch + 1}] User Denoise Loss: {avg_user_loss:.4f}, Item Denoise Loss: {avg_item_loss:.4f}, Time: {epoch_time:.2f}s')


            writer.add_scalar('Denoising/User Loss', avg_user_loss, epoch + 1)
            writer.add_scalar('Denoising/Item Loss', avg_item_loss, epoch + 1)

            if args.early_stop:
                if avg_user_loss < target_best_user_loss:
                    target_best_user_loss = avg_user_loss
                    target_no_improve_epochs = 0
                else:
                    target_no_improve_epochs += 1

                if target_no_improve_epochs >= n:
                    target_stop = True
                    print("Target Behavior Stop Early!")

        with torch.no_grad():
            denoised_user_embs, denoised_item_embs = utils.get_denoised_embeddings(target_dataset, Recmodel, user_reverse_model, item_reverse_model, diffusion)


        for aux_idx, (aux_dnn_model, aux_optimizer, aux_loader) in enumerate(
                zip(aux_dnn_models, auxiliary_optimizers, auxiliary_dataloaders)):
            if (not args.early_stop) or (not aux_stop_flags[aux_idx]):
                aux_start_time = time.time()
                aux_dnn_model.train()
                batch_count = 0
                aux_total_loss = 0.0

                for batch_idx, (user_idx, batch) in enumerate(aux_loader):
                    batch = batch.to(device)
                    batch_count += 1

                    aux_optimizer.zero_grad()
                    losses = diffusion.training_losses(aux_dnn_model, batch, denoised_user_embs[user_idx], args.reweight)
                    loss = losses["loss"].mean()

                    aux_total_loss += loss.item()
                    loss.backward()
                    aux_optimizer.step()

                aux_avg_loss = aux_total_loss / batch_count
                aux_time = time.time() - aux_start_time

                print(f'EPOCH[{epoch + 1}/{args.epoch}] Auxiliary Behavior {aux_idx + 1} Diffusion loss: {aux_avg_loss:.4f}, Time taken: {aux_time:.2f}s')
                # save in tensorboard
                writer.add_scalar(f'Training/Auxiliary Behavior {aux_idx + 1} Diffusion Loss', aux_total_loss, epoch + 1)


                if args.early_stop:
                    if aux_avg_loss < aux_best_losses[aux_idx]:
                        aux_best_losses[aux_idx] = aux_avg_loss
                        aux_no_improve_epochs[aux_idx] = 0
                    else:
                        aux_no_improve_epochs[aux_idx] += 1

                    if aux_no_improve_epochs[aux_idx] >= n:
                        aux_stop_flags[aux_idx] = True
                        print(f"auxiliary behavior {aux_idx + 1}  stop early!")

        if args.early_stop and target_stop and all(aux_stop_flags):
            print("Stop training !")
            break

        print('---' * 18)
    print("End Training !")

    utils.perform_inference_and_save(target_dataset, Recmodel, item_reverse_model, diffusion, auxiliary_dataloaders,
                                     auxiliary_datasets, aux_dnn_models, denoised_user_embs, denoised_item_embs)

if __name__ == '__main__':
    main()
