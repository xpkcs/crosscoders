from abc import ABC, abstractmethod
from typing import Protocol


class FitLoop(ABC):

    def __init__(self, model_factory=None, optimizer_factory=None, loss_fn=None, train_dl=None, val_dl=None, num_epochs=1):
        # self.model = model_factory
        # self.optimizer = optimizer_factory
        # self.loss_fn = loss_fn
        # self.train_dl = train_dl
        # self.val_dl = val_dl
        # self.num_epochs = num_epochs

        self._initialize()

    def _initialize(self): pass

    # def __call__(self, model): ...
    def __call__(self):

        print(self.__class__.__name__)

        self._on_start()

        for epoch_idx in range(1):

            self._on_epoch_start()

            # train
            self.model.train()
            for batch_idx in range(1):

                self._on_batch_start()

                self.training_step()

                # report
                self._on_batch_end()

            # eval
            self.model.eval()


            self._on_epoch_end()

        # report
        # save checkpoint
        self._on_end()

    # def __call__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
    #             loss_fn: Callable, train_dataloader: Any,
    #             val_dataloader: Optional[Any] = None,
    #             scheduler: Optional[Any] = None,
    #             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...

    def _on_start(self): pass
    def _on_epoch_start(self): pass
    def _on_batch_start(self):
        self.optimizer.zero_grad()

    @abstractmethod
    def training_step(self): pass
    def _on_batch_end(self): pass
    def _on_epoch_end(self): pass
    def _on_end(self): pass

class DefaultFitLoop(FitLoop):
    def training_step(self):
        pass






# class DistributedFitLoop(FitLoop):

    # def __call__(self, model):

    #     print("distributed fit loop")
