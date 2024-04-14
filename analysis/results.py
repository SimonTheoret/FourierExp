import fire
import matplotlib
import torch
import pandas as pd

# TODO: Compute the average over the 49 batches

# with open("results/allcnngaussiansgd_epoch100", "r") as f:
experiments = [
    "allcnnvanillaadamw",
    "allcnngaussianadamw",
    # "allcnnadvadamw",
    "allcnnvanillasgd",
    "allcnngaussiansgd",
    # "allcnnadvsgd",
    "mobilevitadamwvanill",  # do not correct the typo
    "mobilevitadamwgaussian",
    # "mobilevitadamwadv",
    "mobilevitsgdvanilla",
    "mobilevitsgdgaussian",
    # "mobilevitsgdadv",
]
print("Looking for the file")
exp_dict = {}
for exp in experiments:
    exp_dict[exp] = torch.load("models/" + exp + "_epoch100")
# fourier_accuracies = {
#     key: data["accuracies"][key]
#     for key in ["fourier_high_pass_accuracy", "fourier_low_pass_accuracy"]
# }
# test_accuracies = data["accuracies"]["test_accuracy"]

# df = pd.DataFrame.from_dict(accuracies)
