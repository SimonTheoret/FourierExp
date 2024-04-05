from cifar10 import Cifar10


class VanillaCifar10(Cifar10):
    def __init__(self, batch_size: int, dataset_name: str = "vanillacifar10"):
        super().__init__(batch_size=batch_size, dataset_name=dataset_name, transformations = None)

class GaussianCifar10(Cifar10):
    def __init__(self, batch_size: int, dataset_name: str = "gaussiancifar10"):
        super().__init__(batch_size=batch_size, dataset_name=dataset_name, transformations = [self.apply_gaussian_tensor])

class AdversarialCifar10(Cifar10):
    def __init__(self, batch_size: int, dataset_name: str = "adversarialcifar10"):
        super().__init__(batch_size=batch_size, dataset_name=dataset_name, transformations = [self.apply_gaussian_tensor])
