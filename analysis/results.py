import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns

sns.set()

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
expe_dict = {}
fourier_acc_dict = {}
for exp in experiments:
    expe_dict[exp] = torch.load("models/" + exp + "_epoch100")
    accs = expe_dict[exp]["accuracies"]
    fourier_acc_dict[exp + "low"] = accs["fourier_low_pass_accuracy"]
    fourier_acc_dict[exp + "high"] = accs["fourier_high_pass_accuracy"]

fourier_df = pd.DataFrame.from_dict(fourier_acc_dict)
mapper = {
    "allcnnvanillaadamwlow": "ALL-CNN Vanilla Adamw Low pass" , "allcnnvanillaadamwhigh":"ALL-CNN Vanilla Adam High pass",
    "allcnngaussianadamwlow": "ALL-CNN Gaussian Adamw Low pass" , "allcnngaussianadamwhigh":"ALL-CNN Gaussian Adam High pass",
    "allcnnvanillasgdlow": "ALL-CNN Vanilla Adamw Low pass" , "allcnnvanillaadamwhigh":"ALL-CNN Vanilla Adam High pass",
    "allcnnvanillaadamwlow": "ALL-CNN Vanilla Adamw Low pass" , "allcnnvanillaadamwhigh":"ALL-CNN Vanilla Adam High pass",
    "allcnnvanillaadamwlow": "ALL-CNN Vanilla Adamw Low pass" , "allcnnvanillaadamwhigh":"ALL-CNN Vanilla Adam High pass",
    "allcnnvanillaadamwlow": "ALL-CNN Vanilla Adamw Low pass" , "allcnnvanillaadamwhigh":"ALL-CNN Vanilla Adam High pass",
    "allcnnvanillaadamwlow": "ALL-CNN Vanilla Adamw Low pass" , "allcnnvanillaadamwhigh":"ALL-CNN Vanilla Adam High pass",
    "allcnnvanillaadamwlow": "ALL-CNN Vanilla Adamw Low pass" , "allcnnvanillaadamwhigh":"ALL-CNN Vanilla Adam High pass",
          }
print(fourier_df)
fourier_df.plot()
plt.show()
# fourier_accuracies = {
#     key: data["accuracies"][key]
#     for key in ["fourier_high_pass_accuracy", "fourier_low_pass_accuracy"]
# }
# test_accuracies = data["accuracies"]["test_accuracy"]

# df = pd.DataFrame.from_dict(accuracies)
