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
fourier_acc_low_dict = {}
fourier_acc_high_dict = {}
test_acc_dict = {}
for exp in experiments:
    expe_dict[exp] = torch.load("models/" + exp + "_epoch300")
    accs = expe_dict[exp]["accuracies"]
    fourier_acc_low_dict[exp + "low"] = accs["fourier_low_pass_accuracy"]
    fourier_acc_high_dict[exp + "high"] = accs["fourier_high_pass_accuracy"]
    test_acc_dict[exp] = accs["test_accuracy"]

test_acc_df = pd.DataFrame.from_dict(test_acc_dict)
fourier_low_df = pd.DataFrame.from_dict(fourier_acc_low_dict)
fourier_high_df = pd.DataFrame.from_dict(fourier_acc_high_dict)
mapper_acc = {
    "allcnnvanillaadamw": "CNN Vanilla Adamw",
    "allcnngaussianadamw": "CNN Gaussian Adamw",
    "allcnnvanillasgd": "CNN Vanilla SGD",
    "allcnngaussiansgd": "CNN Gaussian SGD",
    "mobilevitadamwvanill": "ViT Vanilla Adamw",
    "mobilevitadamwgaussian": "ViT Gaussian Adamw",
    "mobilevitsgdvanilla": "ViT Vanilla SGD",
    "mobilevitsgdgaussian": "ViT Gaussian SGD",
}
mapper_fourier = {  # TODO: update for the ADV results
    "allcnnvanillaadamwlow": "CNN Vanilla Adamw Low pass",
    "allcnnvanillaadamwhigh": "CNN Vanilla Adamw High pass",
    "allcnngaussianadamwlow": "CNN Gaussian Adamw Low pass",
    "allcnngaussianadamwhigh": "CNN Gaussian Adamw High pass",
    "allcnnvanillasgdlow": "CNN Vanilla SGD Low pass",
    "allcnnvanillasgdhigh": "CNN Vanilla SGD High pass",
    "allcnngaussiansgdlow": "CNN Gaussian SGD Low pass",
    "allcnngaussiansgdhigh": "CNN Gaussian SGD High pass",
    "mobilevitadamwvanilllow": "ViT Vanilla Adamw Low pass",
    "mobilevitadamwvanillhigh": "ViT Vanilla Adamw High pass",
    "mobilevitadamwgaussianlow": "ViT Gaussian Adamw Low pass",
    "mobilevitadamwgaussianhigh": "ViT Gaussian Adamw High pass",
    "mobilevitsgdvanillalow": "ViT Vanilla SGD Low pass",
    "mobilevitsgdvanillahigh": "ViT Vanilla SGD High pass",
    "mobilevitsgdgaussianlow": "ViT Gaussian SGD Low pass",
    "mobilevitsgdgaussianhigh": "ViT Gaussian SGD High pass",
}


index = list(range(0, 301, 15))
fourier_low_df["index"] = index
fourier_low_df.set_index("index")
fourier_high_df["index"] = index
fourier_high_df.set_index("index")
fourier_high_df.rename(columns=mapper_fourier, inplace=True)
fourier_low_df.rename(columns=mapper_fourier, inplace=True)
test_acc_df.rename(columns=mapper_acc, inplace=True)

# styles = [
#     "bx--",
#     "bP--",
#     "bx-",
#     "bP-",
#     "rx--",
#     "rP--",
#     "rx-",
#     "rP-",
# ]  # TODO update for the ADV resutls

high_list = fourier_high_df.iloc[-1].to_list()[:-1]  # ignore the index
low_list = fourier_low_df.iloc[-1].to_list()[:-1]  # ignore the index
test_acc_list = test_acc_df.iloc[-1].to_list()
exp_list = list(mapper_acc.values())

big_dict = {
    "High Pass acc": high_list,
    "Low Pass Acc": low_list,
    "Test Acc": test_acc_list,
    "Experiments": exp_list,
}
df = pd.DataFrame.from_dict(big_dict)
ax = df.plot(
    kind="bar",
    x="Experiments",
    y=["High Pass acc", "Low Pass Acc", "Test Acc"],
    stacked=True,
    title="Accuracy with High, low, and no frequency filter",
    rot=30,
    # color=["red", "green", "blue"],
)

plt.savefig("assets/stacked_acc", dpi=1000)
plt.show()


fourier_high_df.plot(
    x="index",
    title="Accuracy Without High Frequency Features",
    xlabel="Epoch",
    ylabel="Test accuracy",
)
plt.savefig("assets/high_pass_fourier_test_acc")
plt.show()
fourier_low_df.plot(
    x="index",
    title="Accuracy Without Low Frequency Features",
    xlabel="Epoch",
    ylabel="Test accuracy",
)
plt.savefig("assets/low_pass_fourier_test_acc")
plt.show()

test_acc_df.plot(
    # style=styles,
    title="Models Test Accuracy",
    xlabel="Epoch",
    ylabel="Test accuracy",
)
plt.savefig("assets/test_accuracy")
plt.show()


# for i, col in enumerate(fourier_df.columns):
#     fourier_df[col].plot()
#     if i == len(mapper) - 1:
#         plt.show()

# fourier_accuracies = {
#     key: data["accuracies"][key]
#     for key in ["fourier_high_pass_accuracy", "fourier_low_pass_accuracy"]
# }
# test_accuracies = data["accuracies"]["test_accuracy"]

# df = pd.DataFrame.from_dict(accuracies)
