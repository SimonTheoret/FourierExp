from architecture import Trainer

class TrainerVanilla(Trainer):

    def train(self) -> None: #TODO: Compute accuracy at each epoch and add it to the train_accuracy list
        self.model.train()
        assert self.train_dataset.train_dataset is not None
        assert self.train_dataset.train_dataloader is not None
        for batch_idx, (data, target) in enumerate(self.train_dataset.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.train_dataset.train_dataset),
                    100. * batch_idx / len(self.train_dataset.train_dataset), loss.item()))

    def test(self)-> None: #TODO: Compute accuracy at each epoch and add it to the test_accuracy list
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
