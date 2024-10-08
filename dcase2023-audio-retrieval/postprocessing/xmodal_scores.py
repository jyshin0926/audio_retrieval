import dbm
import json
import os
import shelve
from dbm import dumb

import nltk
import torch

from utils import criterion_utils, data_utils, model_utils

dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}

stopwords = nltk.corpus.stopwords.words("english")

# Trial info
trial_base = "~"
trial_series = "~"
trial_name = "data/Clotho"
ckp_dir = "pretrained_models"

# Model checkpoint directory
# ckp_fpath = os.path.join(trial_base, trial_series, trial_name, ckp_dir)
ckp_fdir= '/content/drive/MyDrive/2024dcase/task8_retrieval/dcase2023-audio-retrieval/data/Clotho/pretrained_models'
ckp_fpath = os.path.join(ckp_fdir, 'baseline_ckpt.pth')
# ckp_fpath = '/content/drive/MyDrive/2024dcase/task8_retrieval/dcase2023-audio-retrieval/baseline_output/tmp_output/2024-05-24/checkpoint' # 추후에 리팩토링 필요

os.makedirs(ckp_fdir, exist_ok=True)


# Load trial parameters
# conf_fpath = os.path.join(trial_base, trial_series, trial_name, "params.json")
conf_fpath = "/content/drive/MyDrive/2024dcase/task8_retrieval/dcase2023-audio-retrieval/baseline_output/tmp_output/2024-05-24_ef230_00000/params.json"
with open(conf_fpath, "rb") as store:
    conf = json.load(store)
print("Load", conf_fpath)

# Load data
data_conf = conf["data_conf"]
train_ds = data_utils.load_data(data_conf["train_data"], train=False)
val_ds = data_utils.load_data(data_conf["val_data"], train=False)
eval_ds = data_utils.load_data(data_conf["eval_data"], train=False)

# Restore model checkpoint
param_conf = conf["param_conf"]
model_params = conf[param_conf["model"]]
obj_params = conf["criteria"][param_conf["criterion"]]
model = model_utils.init_model(model_params, train_ds.text_vocab)
model = model_utils.restore(model, ckp_fpath)
print(model)

model.eval()

for name, ds in zip(["train", "val", "eval"], [train_ds, val_ds, eval_ds]):
    text2vec = {}
    for idx in ds.text_data.index:
        item = ds.text_data.loc[idx]

        if ds.text_level == "word":
            text_vec = torch.as_tensor([ds.text_vocab(key) for key in item["tokens"] if key not in stopwords])
            text2vec[item["tid"]] = torch.unsqueeze(text_vec, dim=0)

        elif ds.text_level == "sentence":
            text_vec = torch.as_tensor([ds.text_vocab(item["tid"])])
            text2vec[item["tid"]] = torch.unsqueeze(text_vec, dim=0)

    # Compute pairwise cross-modal scores
    score_fpath = os.path.join(ckp_fdir, f"{name}_xmodal_scores.db")
    with shelve.open(filename=score_fpath, flag="n", protocol=2) as stream:
        for fid in ds.text_data["fid"].unique():
            group_scores = {}

            # Encode audio data
            audio_vec = torch.as_tensor(ds.audio_data[fid][()])
            audio_vec = torch.unsqueeze(audio_vec, dim=0)
            audio_embed = model.audio_branch(audio_vec)[0]

            for tid in text2vec:
                text_embed = model.text_branch(text2vec[tid])[0]  # Encode text data
                xmodal_S = criterion_utils.score(audio_embed, text_embed, obj_params["args"].get("dist", "dot_product"))
                group_scores[tid] = xmodal_S.item()
            stream[fid] = group_scores
    print("Save", score_fpath)
