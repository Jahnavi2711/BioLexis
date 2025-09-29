# src/autoencoder.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

class SparseKmerDataset(Dataset):
    def __init__(self, X_sparse):
        self.X = X_sparse
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        row = self.X.getrow(idx).toarray().squeeze().astype('float32')
        return row

class AE(nn.Module):
    # This class is unchanged
    def __init__(self, input_dim, hidden_dims=[1024,512], latent_dim=128):
        super().__init__()
        dims = [input_dim] + hidden_dims + [latent_dim]
        enc_layers = []
        for i in range(len(dims)-1):
            enc_layers.append(nn.Linear(dims[i], dims[i+1]))
            enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        
        d_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        dec_layers = []
        for i in range(len(d_dims)-1):
            dec_layers.append(nn.Linear(d_dims[i], d_dims[i+1]))
            if i < len(d_dims)-2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# MODIFIED: This function is updated with progress bars and parallel loading control
def train_autoencoder(X_sparse, latent_dim=128, hidden_dims=[1024,512], batch_size=512, epochs=20, lr=1e-3,
                      weight_decay=1e-6, device='cpu', checkpoint_path=None, early_stopping=5, num_workers=2):
    # Ensure parameters are the correct types
    early_stopping = int(early_stopping)
    epochs = int(epochs)
    batch_size = int(batch_size)
    num_workers = int(num_workers)
    device = torch.device(device)
    dataset = SparseKmerDataset(X_sparse)
    # MODIFIED: Use the num_workers parameter for parallel data loading
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    input_dim = X_sparse.shape[1]
    model = AE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience = 0
    
    print("Training Autoencoder...")
    # NEW: Outer progress bar for epochs
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training AE")
    
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        
        # NEW: Inner progress bar for batches, disappears after completion
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch in batch_pbar:
            batch = batch.to(device)
            recon, z = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        
        epoch_loss = epoch_loss / len(dataset)
        
        # NEW: Update the epoch progress bar with the latest loss instead of printing
        epoch_pbar.set_postfix(loss=f"{epoch_loss:.6f}")
        
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            patience = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience += 1
            if patience >= early_stopping:
                print("\n[AE] Early stopping triggered")
                epoch_pbar.close() # Close the progress bar cleanly
                break
                
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    print("Generating final embeddings...")
    model.eval()
    # MODIFIED: Use the num_workers parameter here as well
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    zs = []
    with torch.no_grad():
        # NEW: Progress bar for the final embedding generation step
        for batch in tqdm(full_loader, desc="Generating Embeddings"):
            batch = batch.to(device)
            _, z = model(batch)
            zs.append(z.cpu().numpy())
            
    embeddings = np.vstack(zs)
    return model, embeddings