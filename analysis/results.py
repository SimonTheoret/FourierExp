import matplotlib.pyplot as plt
import re
import torch
import pandas as pd
import seaborn as sns

sns.set()

# TODO: Compute the average over the 49 batches
MAX_N_SEEDS = 6
FINAL_EPOCH = "_epoch105"
FINAL_EPOCH_N = 105
SEEDS_NAMES = ["_seed_" + str(i) for i in range(MAX_N_SEEDS)]
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
expe_dict = {}
fourier_acc_low_dict = {}
fourier_acc_high_dict = {}
test_acc_dict = {}
for exp in experiments:
    for seed_num in SEEDS_NAMES:
        exp_seed = exp + seed_num
        expe_dict[exp_seed] = torch.load("models/" + exp + seed_num + FINAL_EPOCH)
        accs = expe_dict[exp_seed]["accuracies"]
        fourier_acc_low_dict[exp_seed + "low"] = accs["fourier_low_pass_accuracy"]
        fourier_acc_high_dict[exp_seed + "high"] = accs["fourier_high_pass_accuracy"]
        test_acc_dict[exp_seed] = accs["test_accuracy"]

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


def mapper_func(name: str):
    if name == "index":
        return name
    else:
        new_name: str = ""
        for key in mapper_acc.keys():
            if "high" in name or "low" in name:
                if key in name:
                    new_name += (
                        mapper_fourier[key + "high"]
                        if "high" in name
                        else mapper_fourier[key + "low"]
                    )  # new name starts with the value
                    seed = re.findall(r"\d+", name)[
                        0
                    ]  # should contain a single integer
                    assert isinstance(seed, str)
                    new_name = new_name + " seed " + str(seed)
                    return new_name

        for key in mapper_acc.keys():
            if key in name:
                new_name += mapper_acc[key]  # new name starts with the value
                seed = re.findall(r"\d+", name)[0]  # should contain a single integer
                assert isinstance(seed, str)
                new_name = new_name + " seed " + str(seed)
                return new_name


index = list(range(0, FINAL_EPOCH_N + 1, 15))
assert len(index) == 7
assert index[-1] == 105

fourier_low_df["index"] = index
fourier_low_df.set_index("index")
fourier_high_df["index"] = index
fourier_high_df.set_index("index")
fourier_high_df.rename(columns=mapper_func, inplace=True)
fourier_low_df.rename(columns=mapper_func, inplace=True)
test_acc_df.rename(columns=mapper_func, inplace=True)

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
# def avg_col(df: pd.DataFrame, start:int, end:int):
#     df = df.drop(columns=["index"])
#     return df.iloc[:, start:end].mean(axis = 1)


def avg_col(df: pd.DataFrame, contains: str):
    names = [name for name in df.columns if contains in name]
    df = df[names]
    if df.empty:
        return None
    return df.mean(axis=1)


avg_high_fourier_df = [
    (key, avg_col(fourier_high_df, key)) for key in mapper_fourier.values()
]
avg_high_fourier_df = {
    i[0]: i[1].to_numpy() for i in avg_high_fourier_df if i[1] is not None
}
avg_high_fourier_df = pd.DataFrame.from_dict(avg_high_fourier_df)
avg_high_fourier_df["index"] = index
avg_high_fourier_df.set_index("index")


avg_low_fourier_df = [
    (key, avg_col(fourier_low_df, key)) for key in mapper_fourier.values()
]
avg_low_fourier_df = {
    i[0]: i[1].to_numpy() for i in avg_low_fourier_df if i[1] is not None
}
avg_low_fourier_df = pd.DataFrame.from_dict(avg_low_fourier_df)
avg_low_fourier_df["index"] = index
avg_low_fourier_df.set_index("index")

print(test_acc_df)
avg_acc_df = [(key, avg_col(test_acc_df, key)) for key in mapper_acc.values()]
print(avg_acc_df)
avg_acc_df = {i[0]: i[1].to_numpy() for i in avg_acc_df if i[1] is not None}
avg_acc_df = pd.DataFrame.from_dict(avg_acc_df)


avg_high_fourier_df.plot(
    title="Accuracy Without High Frequency Features",
    x="index",
    xlabel="Epoch",
    ylabel="Test accuracy",
)

plt.savefig("assets/avg_high_pass_fourier_test_acc")
plt.show()


avg_low_fourier_df.plot(
    title="Accuracy Without Low Frequency Features",
    x="index",
    xlabel="Epoch",
    ylabel="Test accuracy",
)

plt.savefig("assets/avg_low_pass_fourier_test_acc")
plt.show()

avg_acc_df.plot(
    title="Models Test Accuracy",
    xlabel="Epoch",
    ylabel="Test accuracy",
)

plt.savefig("assets/avg_test_accuracy")
plt.show()
# ax = df.plot(
#     kind="bar",
#     x="Experiments",
#     y=["High Pass acc", "Low Pass Acc", "Test Acc"],
#     stacked=True,
#     title="Accuracy with High, low, and no frequency filter",
#     rot=30,
#     # color=["red", "green", "blue"],
# )

# plt.savefig("assets/stacked_acc", dpi=1000)
# plt.show()


# fourier_high_df.plot(
#     x="index",
#     title="Accuracy Without High Frequency Features",
#     xlabel="Epoch",
#     ylabel="Test accuracy",
# )
# plt.savefig("assets/high_pass_fourier_test_acc")
# plt.show()
# fourier_low_df.plot(
#     x="index",
#     title="Accuracy Without Low Frequency Features",
#     xlabel="Epoch",
#     ylabel="Test accuracy",
# )
# plt.savefig("assets/low_pass_fourier_test_acc")
# plt.show()

# test_acc_df.plot(
#     # style=styles,
#     title="Models Test Accuracy",
#     xlabel="Epoch",
#     ylabel="Test accuracy",
# )
# plt.savefig("assets/test_accuracy")
# plt.show()


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
