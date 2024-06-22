import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os


def train(
        model,
        config,
        train_loader,
        valid_loader=None,
        valid_epoch_interval=300,
        foldername="",
):
    # Control random seed in the current script.
    torch.manual_seed(0)
    np.random.seed(0)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p0 = int(0.25 * config["epochs"])
    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    p3 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )
    history = {'train_loss': [], 'val_rmse': []}
    best_valid_loss = 1e10
    train_loss = []
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                # The forward method returns loss.
                loss = model(train_batch)
                train_loss.append(loss.detach().cpu().numpy())
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def evaluate(model, test_loader, nsample=1000, scaler=1, mean_scaler=0, foldername=""):
    torch.manual_seed(0)
    np.random.seed(0)

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                                      ((samples_median.values - c_target) * eval_points) ** 2
                              ) * (scaler ** 2)
                mae_current = (
                                  torch.abs((samples_median.values - c_target) * eval_points)
                              ) * scaler
                mse_total += torch.sum(mse_current, dim=0)
                mae_total += torch.sum(mae_current, dim=0)
                evalpoints_total += torch.sum(eval_points, dim=0)
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": torch.mean(
                            torch.sqrt(torch.div(mse_total, evalpoints_total))
                        ).item(),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

    print("RMSE:", torch.mean(torch.sqrt(torch.div(mse_total, evalpoints_total))).item(), )


def genera(model, genera_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", max_arr=None, attention_mech=''):
    if max_arr is None:
        max_arr = []
    torch.manual_seed(0)
    np.random.seed(0)

    imputed_samples = []
    observed_data_all = []
    cond_mask_all = []
    observed_mask_all = []

    with torch.no_grad():
        model.eval()

        with tqdm(genera_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, gen_batch in enumerate(it, start=1):
                output = model.genera(gen_batch, nsample)
                samples, observed_data, target_mask, observed_mask, _ = output
                observed_data_all.append(observed_data.squeeze())
                cond_mask_all.append(target_mask.squeeze())
                observed_mask_all.append(observed_mask.squeeze())
                samples = samples.permute(0, 1, 3, 2)
                samples_median = samples.median(dim=1)
                imputed_samples.append(samples_median.values.squeeze())

        imputed_samples_ndarray = torch.cat(imputed_samples, dim=0).cpu().numpy()
        observed_data_all_ndarray = torch.cat(observed_data_all, dim=0).cpu().numpy()
        cond_mask_all_ndarray = torch.cat(cond_mask_all, dim=0).cpu().numpy()
        indices = np.where(cond_mask_all_ndarray == 1)
        observed_data_all_ndarray[indices] = imputed_samples_ndarray[indices]
        pd.DataFrame(imputed_samples_ndarray).to_csv(f'./imputed.csv', header=False, index=False)

