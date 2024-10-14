import torch


class DataPrefetcher:
    def __init__(self, loader, device):
        self.origin_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device=self.device)
        self.use_gpu = torch.cuda.is_available()
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.loader = iter(self.origin_loader)
            self.batch = next(self.loader)
        if self.use_gpu:
            with torch.cuda.stream(self.stream):
                self.batch = self.batch.to(device=self.device, non_blocking=True)

    def next(self):
        if self.use_gpu:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
