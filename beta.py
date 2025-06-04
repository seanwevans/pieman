import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import beta
import matplotlib.pyplot as plt
import time


class BetaNet(nn.Module):
    def __init__(self, input_dim):
        super(BetaNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Outputs: alpha and beta
        self.softplus = nn.Softplus()  # To ensure positive outputs

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softplus(x)  # Ensure positivity for alpha and beta
        return x


def get_ground_truth(space, time):
    x, y, z = space
    t = time

    a = torch.randn(4)
    b = torch.randn(4)

    # simulated
    return (
        torch.abs(
            2
            + a[0] * torch.sin(np.pi * x)
            + a[1] * torch.cos(np.pi * y)
            + a[2] * torch.exp(-z * t)
            + a[3] * (x * y * z)
        )
        + 1e-3,
        torch.abs(
            5
            + b[0] * torch.log(1 + x + 0.1)
            + b[1] * torch.sqrt(torch.abs(y) + 0.1)
            + b[2] * torch.cos(2 * np.pi * z)
            + b[3] * (x + y + z) * t
        )
        + 1e-3,
    )


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    num_samples = 2 ** 10
    input_dim = 4  # x, y, z, t
    x_range = [0, 1]  # Normalized range for inputs

    batch_size = 2 ** 8
    max_epochs = 1000
    loss_history = []
    epoch_loss = 1
    epoch = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic data
    inputs = np.random.uniform(x_range[0], x_range[1], size=(num_samples, input_dim))
    x, y, z, t = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]

    # Convert numpy arrays to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    z_tensor = torch.tensor(z, dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)

    # Compute ground truth alpha and beta
    alpha_true, beta_true = get_ground_truth((x_tensor, y_tensor, z_tensor), t_tensor)

    # Convert back to numpy for CDF computation
    cdf_true = beta.cdf(x, alpha_true.numpy(), beta_true.numpy())  # precompute cdf

    # Convert to PyTorch tensors - first conversion from numpy still needs torch.tensor()
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    alpha_true_tensor = alpha_true.clone().detach().unsqueeze(1).to(device)
    beta_true_tensor = beta_true.clone().detach().unsqueeze(1).to(device)
    cdf_true_tensor = (
        torch.tensor(cdf_true, dtype=torch.float32).unsqueeze(1).to(device)
    )

    # Initialize the model, loss function, and optimizer
    model = BetaNet(input_dim=input_dim).to(device)
    criterion_alpha = nn.MSELoss()
    criterion_beta = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    dataset = torch.utils.data.TensorDataset(
        inputs_tensor, alpha_true_tensor, beta_true_tensor
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    prev_loss = 1
    prev_time = time.time()

    while epoch_loss > 1e-9 and epoch < max_epochs:
        model.train()
        epoch_loss = 0
        for batch_inputs, batch_alpha, batch_beta in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_alpha = batch_alpha.to(device)
            batch_beta = batch_beta.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)

            alpha_pred, beta_pred = outputs[:, 0].unsqueeze(1), outputs[:, 1].unsqueeze(
                1
            )
            loss_alpha = criterion_alpha(alpha_pred, batch_alpha)
            loss_beta = criterion_beta(beta_pred, batch_beta)
            loss = loss_alpha + loss_beta
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        loss_history.append(epoch_loss)
        epoch += 1

        print(f"{epoch}: {prev_loss-epoch_loss:.10f} in {time.time()-prev_time:.3f}s")

    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid()
    plt.show()
