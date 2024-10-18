import numpy as np
import matplotlib.pyplot as plt
from SDF_compute import Naive_SDF as Coarse_SDF
from SDF_compute import generate_sphere_point_cloud, generate_plane_point_cloud
import random


class Interpolated_SDF:
    def __init__(self, point_cloud, method: str = 'Shepard'):
        self.point_cloud = point_cloud
        self.method = method
        # get bounding box
        self.min_x, self.min_y, self.min_z = np.min(point_cloud, axis=0)
        self.max_x, self.max_y, self.max_z = np.max(point_cloud, axis=0)
        # get the coarse SDF
        self.coarse_sdf = Coarse_SDF(point_cloud)
        # sample points
        self.sample_points = list(self.point_cloud)
        self.sample_sdf = list(np.zeros(len(self.sample_points)))
        # random sample more points
        self.x_bias = (self.max_x - self.min_x) / 4
        self.y_bias = (self.max_y - self.min_y) / 4
        self.z_bias = (self.max_z - self.min_z) / 4
        for i in range(200):
            self.sample_points.append([random.uniform(self.min_x - self.x_bias, self.max_x + self.x_bias),
                                       random.uniform(
                                      self.min_y - self.y_bias, self.max_y + self.y_bias),
                random.uniform(self.min_z - self.z_bias, self.max_z + self.z_bias)])
            self.sample_sdf.append(
                self.coarse_sdf._get_point_sdf(self.sample_points[-1]))

        if self.method == 'Shepard':
            pass

        if self.method == 'RBF':
            self.RBF_h = 0.6
            self.RBF_weights = self._RBF_weights()

    def plot_sdf(self, resolution=20):
        # Generate grid points
        x = np.linspace(self.min_x, self.max_x, resolution)
        y = np.linspace(self.min_y, self.max_y, resolution)
        z = np.linspace(self.min_z, self.max_z, resolution)

        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        # Compute SDF for each grid point
        sdf_values = np.array([self._get_point_sdf(point)
                              for point in grid_points])

        # Reshape SDF values to match grid shape
        sdf_values = sdf_values.reshape(xx.shape)
        # Compute the color for each grid point based on SDF values
        colors = sdf_values.flatten()
        # Normalize the colors to range from 0 to 1
        colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

        # Plot SDF as a scatter plot with varying colors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(grid_points[:, 0], grid_points[:, 1],
                   grid_points[:, 2], c=colors, cmap='coolwarm', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def _get_point_sdf(self, point):
        if self.method == 'Shepard':
            return self._shepard_interpolation(point, 2)

        if self.method == 'RBF':
            return self._RBF_interpolation(point)

        return self.coarse_sdf._get_point_sdf(point)

    # * Shepard interpolation

    def _shepard_interpolation(self, point, p):
        # * Implement the shepard interpolation here
        weight = []
        for i in range(len(self.sample_points)):
            # compute weight
            weight.append(1 / np.linalg.norm(
                np.array(self.sample_points[i]) - np.array(point))**(-p))
        # normalize weight
        weight = weight / np.sum(weight)
        # compute the sdf
        sdf = 0
        for i in range(len(self.sample_points)):
            sdf += weight[i] * self.sample_sdf[i]

        return sdf

    # * RBF interpolation
    def _radial_basis_function(self, x, c, h):
        # x, c are vectors
        return np.exp(-np.linalg.norm(x - c)**2 / h**2)

    def _RBF_weights(self):
        # * Implement the RBF weights computation here
        radius_matrix = np.zeros(
            (len(self.sample_points), len(self.sample_points)))
        for i in range(len(self.sample_points)):
            for j in range(0, i+1):
                if i == j:
                    radius_matrix[i, j] = 1
                else:
                    radius_matrix[i, j] = radius_matrix[j, i] = self._radial_basis_function(
                        np.array(self.sample_points[i]), np.array(self.sample_points[j]), self.RBF_h)
        # radius_matrix * weights = sample_sdf
        # weights = radius_matrix^-1 * sample_sdf
        radius_matrix_inv = np.linalg.inv(radius_matrix)
        weights = np.dot(radius_matrix_inv, self.sample_sdf)

        return weights

    def _RBF_interpolation(self, point):
        # * Implement the RBF interpolation here
        sdf = 0
        for i in range(len(self.sample_points)):
            sdf += self.RBF_weights[i] * self._radial_basis_function(
                np.array(point), np.array(self.sample_points[i]), self.RBF_h)

        return sdf


radius = 1.0
num_points = 200
point_cloud = generate_sphere_point_cloud(radius, num_points)
sdf = Interpolated_SDF(point_cloud, method='RBF')
sdf.plot_sdf(20)
