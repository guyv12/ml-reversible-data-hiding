import torch


class Ridge:

    def __init__(self) -> None:
        self.weights = torch.zeros(512, 512)
        
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        return
    

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.weights @ X
        
    