import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset

from geometry_utils.collision_detection import ObjectCollisionDetection


class ObjectCollisionDetectionNN(ObjectCollisionDetection):
    def __init__(self, obstacles, robot, offset=0.05, num_points=100000):
        super().__init__(obstacles, robot, offset)
        self.start_time = time.time()
        for obstacle_idx in range(len(obstacles)):
            points, sdf_values = self.generate_sdf_dataset(obstacle_idx, num_points)
            self.sdf_net = SDFNetwork()
            self.train(points, sdf_values)
            self.sdf_net.eval()
        # print("finished setting up")

    def generate_sdf_dataset(self, obstacle_idx, num_points):
        mesh = self.obstacles[obstacle_idx].poly
        points = np.random.uniform(low=mesh.bounds[0], high=mesh.bounds[1], size=(num_points, 3))
        sdf_values = []
        # point_idx = 0
        for point in points:
            """if point_idx % 1000 == 0:
                  print("finished " + str(point_idx))"""
            sdf_values.append(self.sdf_for_sample_point(mesh, point))
            # point_idx += 1
        return points, np.array(sdf_values)

    def train(self, points, sdf_values):
        # Prepare the dataset for PyTorch
        dataset = TensorDataset(torch.Tensor(points), torch.Tensor(sdf_values).unsqueeze(1))
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.sdf_net.parameters(), lr=0.001)

        # Training loop
        num_epochs = 100
        self.sdf_net.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            if time.time() - self.start_time > 1200:
                return False
            for batch_points, batch_sdf_values in dataloader:
                optimizer.zero_grad()
                predictions = self.sdf_net(batch_points)
                batch_sdf_values = batch_sdf_values.squeeze(-1)
                loss = criterion(predictions, batch_sdf_values)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

    def sdf(self, obstacle_idx, point):
        return self.sdf_net(torch.from_numpy(point).float())


class SDFNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=1, num_layers=4):
        super(SDFNetwork, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
