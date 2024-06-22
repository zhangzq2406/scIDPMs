import argparse
import torch
import datetime
import json
import yaml
import os
import glob

from src.main_model_table import scIDPMs
from src.utils_table import train, genera
from dataset import get_dataloader
import pandas as pd

parser = argparse.ArgumentParser(description="scIDM")
parser.add_argument("--config", type=str, default="default.yaml")
parser.add_argument("--device", default="cpu", help="Device")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--file_path", type=str, default="")
parser.add_argument("--label_path", type=str, default=None)
parser.add_argument('--att', type=str, default='MHA')
parser.add_argument('--n_genes', type=int, default=0)

args = parser.parse_args()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_dir = os.getcwd()
file_list = glob.glob(os.path.join(current_dir, '*.pk'))
for file_path in file_list:
    os.remove(file_path)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/test" + "_" + current_time + "/"
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

file_path = args.file_path
file_label = args.label_path

data = pd.read_csv(file_path, index_col=0)

args.n_genes = data.shape[1]

train_loader, valid_loader, test_loader, genera_loader, dataset, max_arr = get_dataloader(
    file_path=args.file_path,
    label_path=args.label_path,
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    eva_length=args.n_genes
)

model = scIDPMs(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

genera(
    model,
    genera_loader,
    nsample=args.nsample,
    scaler=1,
    foldername=foldername,
    max_arr=max_arr,
    attention_mech=args.att
)

current_dir = os.getcwd()
file_list = glob.glob(os.path.join(current_dir, '*.pk'))
for file_path in file_list:
    os.remove(file_path)

print('Imputation finish.')
