import logging
from tqdm import tqdm
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from gamc.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from gamc.datasets.data_util import load_fake_news_graph_dataset
from gamc.models import build_model
import setproctitle
setproctitle.setproctitle('yinshu')


def graph_classification_evaluation(model, pooler, dataloader, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                    mute=False):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, batch_g in enumerate(dataloader):
            batch_g = batch_g.to(device)
            feat = batch_g.x
            labels = batch_g.y.cpu()
            out = model.embed(feat, batch_g.edge_index)
            if pooler == "mean":
                out = global_mean_pool(out, batch_g.batch)
            elif pooler == "max":
                out = global_max_pool(out, batch_g.batch)
            elif pooler == "sum":
                out = global_add_pool(out, batch_g.batch)
            else:
                raise NotImplementedError

            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    return test_f1


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob=True, logger=None):
    train_loader, eval_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g = batch
            batch_g = batch_g.to(device)

            feat = batch_g.x
            model.train()
            loss, loss_dict = model(feat, batch_g.edge_index, batch_g.news_node)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)
        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    feature = args.feature
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    pooler = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size

    # graphs, (num_features, num_classes) = load_graph_classification_dataset(dataset_name, deg4feat=deg4feat)
    dataset, (num_features, num_classes) = load_fake_news_graph_dataset(dataset_name, feature, deg4feat=deg4feat)
    args.num_features = num_features

    # train_idx = torch.arange(len(dataset))

    train_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        if not load_model:
            model = pretrain(model, pooler, (train_loader, eval_loader), optimizer, max_epoch, device, scheduler,
                             num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()
        test_f1 = graph_classification_evaluation(model, pooler, eval_loader, num_classes, lr_f, weight_decay_f,
                                                  max_epoch_f, device, mute=False)
        acc_list.append(test_f1)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(acc_list)


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)