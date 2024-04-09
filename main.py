from fire import Fire

from exp_datasets import VanillaCifar10

def main_generic():
    dataset = VanillaCifar10(batch_size=16)
