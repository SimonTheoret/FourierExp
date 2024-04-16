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
    expe_dict[exp] = torch.load("models/" + exp + "_epoch100")
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
    "mobilevitadamwgaussian": "Vit Gaussian Adamw",
    "mobilevitsgdvanilla": "Vit Vanilla SGD",
    "mobilevitsgdgaussian": "Vit Gaussian SGD",
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
    "mobilevitadamwgaussianlow": "Vit Gaussian Adamw Low pass",
    "mobilevitadamwgaussianhigh": "Vit Gaussian Adamw High pass",
    "mobilevitsgdvanillalow": "Vit Vanilla SGD Low pass",
    "mobilevitsgdvanillahigh": "Vit Vanilla SGD High pass",
    "mobilevitsgdgaussianlow": "Vit Gaussian SGD Low pass",
    "mobilevitsgdgaussianhigh": "Vit Gaussian SGD High pass",
}


index = list(range(0, 91, 15))
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
