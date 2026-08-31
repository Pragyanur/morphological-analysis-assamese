import math

import torch
import torch.nn as nn

# defaults
MODEL_DIM = 16
HEADS = 4
LAYERS = 3
LINEAR = 32
DIM_FF = 64
WINDOW_SIZE = 40

class MorphoTransformer(nn.Module):
    def __init__(self, feature_dim=64, window_size=WINDOW_SIZE, nhead=HEADS, num_layers=LAYERS, model_dim=MODEL_DIM, linear=LINEAR, feedforward_dim=DIM_FF):
        super(MorphoTransformer, self).__init__()
        self.model_dim = model_dim
        self.input_projection = nn.Linear(feature_dim, self.model_dim)
                
        # --- Fixed Sinusoidal Positional Embeddings ---
        pe = torch.zeros(window_size, self.model_dim)
        position = torch.arange(0, window_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.model_dim, 2).float() * (-math.log(10000.0) / self.model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # register_buffer ensures it moves with the model to GPU

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=nhead, dim_feedforward=feedforward_dim, 
            dropout=0.2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.model_dim, linear),
            nn.ReLU(),
            nn.Linear(linear, 1) # Outputting a single "Grammar Score"
        )
        
    def forward(self, x, mask=None):
        # x: (Batch, 30, 64)
        x = self.input_projection(x)
        
        # Add the fixed positional embeddings
        x = x + self.pe[:, :x.size(1), :]

        # x = self.input_projection(x) + self.pos_embedding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
                
        # Mean + Max Pooling
        if mask is not None:
            active = mask.logical_not().unsqueeze(-1).float()

            # Masked mean pooling
            mean_pooled = torch.sum(x * active, dim=1) / torch.clamp(
                torch.sum(active, dim=1),
                min=1e-9
            )

            # Masked max pooling
            x_masked = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
            max_pooled = torch.max(x_masked, dim=1).values

        else:
            mean_pooled = x.mean(dim=1)
            max_pooled = x.max(dim=1).values

        # Concatenate mean and max representations
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)

        return self.classifier(pooled)