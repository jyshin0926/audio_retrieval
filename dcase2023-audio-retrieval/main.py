import os
import random
import time

import numpy
import ray
from ray.air import session
import torch
import torch.optim as optim
import yaml
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.tune.stopper import TrialPlateauStopper
from torch.utils.data import DataLoader

from utils import criterion_utils, data_utils, model_utils

torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)

ckp_directory = '/content/drive/MyDrive/2024dcase/task8_retrieval/dcase2023-audio-retrieval/baseline_output/tmp_output/2024-05-24' # 추후에 리팩토링 필요


def exec_trial(conf, ckp_dir=ckp_directory):
    data_conf = conf["data_conf"]
    param_conf = conf["param_conf"]

    train_ds = data_utils.load_data(data_conf["train_data"])
    train_dl = DataLoader(dataset=train_ds, batch_size=param_conf["batch_size"],
                          shuffle=True, collate_fn=data_utils.collate_fn)

    val_ds = data_utils.load_data(data_conf["val_data"])
    val_dl = DataLoader(dataset=val_ds, batch_size=param_conf["batch_size"],
                        shuffle=True, collate_fn=data_utils.collate_fn)

    model_params = conf[param_conf["model"]]
    model = model_utils.init_model(model_params, train_ds.text_vocab)
    print(model)

    obj_params = conf["criteria"][param_conf["criterion"]]
    objective = getattr(criterion_utils, obj_params["name"], None)(**obj_params["args"])

    optim_params = conf[param_conf["optimizer"]]
    optimizer = getattr(optim, optim_params["name"], None)(model.parameters(), **optim_params["args"])

    lr_params = conf[param_conf["lr_scheduler"]]
    lr_scheduler = getattr(optim.lr_scheduler, lr_params["name"], "ReduceLROnPlateau")(optimizer, **lr_params["args"])

    if ckp_dir is not None:
        # model_state, optimizer_state = torch.load(os.path.join(ckp_dir, "checkpoint"))
        model_state, optimizer_state = torch.load(os.path.join(ckp_dir, "checkpoint/checkpoint_ep_78.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    max_epoch = param_conf["num_epoch"] + 1

    for epoch in numpy.arange(0, max_epoch):

        if epoch > 0:
            model_utils.train(model, train_dl, objective, optimizer)

        epoch_results = {}
        epoch_results["train_obj"] = model_utils.eval(model, train_dl, objective)
        epoch_results["val_obj"] = model_utils.eval(model, val_dl, objective)
        epoch_results["stop_metric"] = epoch_results["val_obj"]

        # Reduce learning rate w.r.t validation loss
        lr_scheduler.step(epoch_results["stop_metric"])

        # Save the model to the trial directory: local_dir/exp_name/trial_name/checkpoint_<step>
        # with tune.Experiment.checkpoint_dir(step=epoch) as ckp_dir:
        path = os.path.join(ckp_dir, "checkpoint/checkpoint_ep_%d.pt"%(epoch+78))
        torch.save((model.state_dict(), optimizer.state_dict()), path)

        # Create a checkpoint dictionary
        # checkpoint = {
        #     "model_path": model_path,
        #     "optimizer_path": optimizer_path,
        #     "epoch": epoch
        # }

        # Send the current statistics back to the Ray cluster
        # session.report(**epoch_results)
        # session.report(epoch_results, checkpoint=checkpoint)
        session.report(epoch_results)


        # Reload data (Pair Bootstrapping)
        train_ds = data_utils.load_data(data_conf["train_data"])
        train_dl = DataLoader(dataset=train_ds, batch_size=param_conf["batch_size"],
                              shuffle=True, collate_fn=data_utils.collate_fn)

        val_ds = data_utils.load_data(data_conf["val_data"])
        val_dl = DataLoader(dataset=val_ds, batch_size=param_conf["batch_size"],
                            shuffle=True, collate_fn=data_utils.collate_fn)

# Function to replace placeholders with actual values
def replace_variables(conf, variables):
    conf_str = yaml.dump(conf)
    for key, value in variables.items():
        conf_str = conf_str.replace(f"${{{key}}}", value)
    return yaml.safe_load(conf_str)


# Main
if __name__ == "__main__":
    # Load configuration
    with open("conf.yaml", "rb") as stream:
        conf = yaml.full_load(stream)

    # Extract variables
    variables = conf.pop("variables")

    # Replace placeholders with actual environment variable values
    conf = replace_variables(conf, variables)

    # Print configuration to verify replacement
    print(yaml.dump(conf, default_flow_style=False))

    # Configure ray-tune clusters
    ray_conf = conf["ray_conf"]

    # Set the RAY_TMPDIR environment variable to a valid absolute path
    os.environ["RAY_TMPDIR"] = ray_conf["init_args"]["_temp_dir"]

    ray.init(**ray_conf["init_args"])
    trial_stopper = getattr(tune.stopper, ray_conf["trial_stopper"], TrialPlateauStopper)(**ray_conf["stopper_args"])
    trial_reporter = CLIReporter(max_report_frequency=60, print_intermediate_tables=True)

    for metric in ["train_obj", "val_obj", "stop_metric"]:
        trial_reporter.add_metric_column(metric=metric)


    def trial_name_creator(trial):
        trial_name = "_".join([conf["param_conf"]["model"], trial.trial_id])
        return trial_name


    def trial_dirname_creator(trial):
        trial_dirname = "_".join([time.strftime("%Y-%m-%d"), trial.trial_id])
        return trial_dirname


    # Execute trials - local_dir/exp_name/trial_name
    analysis = tune.run(
        run_or_experiment=exec_trial,
        name=conf["trial_series"],
        stop=trial_stopper,
        config=conf,
        resources_per_trial={"cpu": 1, "gpu": 1},
        num_samples=1,
        local_dir=conf["trial_base"],
        keep_checkpoints_num=3,
        checkpoint_score_attr="min-stop_metric",
        verbose=1,
        progress_reporter=trial_reporter,
        log_to_file=True,
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_dirname_creator,
        fail_fast=False,
        reuse_actors=True,
        raise_on_failed_trial=True
    )

    # Check best trial
    best_trial = analysis.get_best_trial(metric=ray_conf["stopper_args"]["metric"],
                                         mode=ray_conf["stopper_args"]["mode"], scope="all")
    print("Best trial:", best_trial.trial_id)

    # Check best checkpoint
    best_ckp = analysis.get_best_checkpoint(trial=best_trial, metric=ray_conf["stopper_args"]["metric"],
                                            mode=ray_conf["stopper_args"]["mode"])
    print("Best checkpoint:", best_ckp)
