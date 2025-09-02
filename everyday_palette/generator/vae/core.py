import os
import torch
import joblib
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from ..base import BaseGenerator

MODEL_FILENAME = 'model.pt'

def loss_function(recon_x, x, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

class VAEGenerator(BaseGenerator):
    
    def __init__(self, model_cls : type[torch.nn.Module], model_params : dict | None = None):
        super().__init__(model_cls, model_params or {})

    @property
    def device(self) -> str:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train(self, model : torch.nn.Module, data_loader : DataLoader, optimizer_params : dict, epochs : int):
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for (batch,) in data_loader:
                optimizer.zero_grad()
                recon, mu, logvar = model(batch)
                loss = loss_function(recon, batch, mu, logvar)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f}")

    def train(self, x : np.ndarray, epochs : int = 100, data_loader_params : dict | None = {}, optimizer_params : dict | None = {}, resume : bool = False):
        if resume:
            raise NotImplementedError('Resuming train is not available yet')
        else:
            input_size = x.shape[1]
            model_params = {**self.model_params, 'input_size':input_size}
            model = self.model_cls(**model_params)
            self.model_params = model_params
        data_loader_params = data_loader_params or {'batch_size':32, 'shuffle': True}
        optimizer_params = optimizer_params or {'lr':1e-3}
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, **data_loader_params)
        self._train(model, data_loader, optimizer_params, epochs)
        self.model = model

    def sample(self, n : int = 1) -> np.ndarray:
        model = self.model.to(self.device)
        model.eval()
        latent_dim = self.model_params['latent_dim']
        with torch.no_grad():
            z = torch.randn(n, latent_dim, device=self.device)
            samples = model.decode(z).cpu().numpy()  # shape (n, 12)
        return samples
    
    def _save_model(self, path: str):
        state_dict = self.model.to("cpu").state_dict()
        torch.save(state_dict, os.path.join(path, MODEL_FILENAME))

    def _load_model(self, path: str):
        model_path = os.path.join(path, MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        state_dict = torch.load(model_path)
        self.model = self.model_cls(**self.model_params)
        self.model.load_state_dict(state_dict)
