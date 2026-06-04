"""
LSTM model for exchange rate forecasting.
Architecture: LSTM → Dropout → LSTM → Dropout → Dense → Dense(1)
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Stacked LSTM for univariate/multivariate time-series forecasting.

    Args:
        input_size:   Number of input features per timestep
        hidden_size:  Number of LSTM hidden units (first layer)
        num_layers:   Number of stacked LSTM layers
        dropout:      Dropout probability between LSTM layers
        output_size:  Number of output values (1 = next-step prediction)
    """

    def __init__(
        self,
        input_size:  int   = 7,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
        output_size: int   = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch, output_size)
        """
        # Initialise hidden and cell states on same device as input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))   # (batch, seq_len, hidden)
        last_step   = lstm_out[:, -1, :]        # take final timestep
        out         = self.dropout(last_step)
        return self.fc(out)
