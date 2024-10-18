import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import random


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        elif self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


class Naive_SDF:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.kdtree = KDTree(point_cloud)
        self.normals = []
        self.MST = None
        self._compute_normals(point_cloud, 10)

    def render_figure(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1],
                   self.point_cloud[:, 2], c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Plotting the normals
        for i in range(len(self.point_cloud)):
            start = self.point_cloud[i]
            end = self.point_cloud[i] + self.normals[i]
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    [start[2], end[2]], c='r')
        # Plotting the minimum spanning tree edges
        # for edge in self.MST:
        #     start = self.point_cloud[edge[0]]
        #     end = self.point_cloud[edge[1]]
        #     ax.plot([start[0], end[0]], [start[1], end[1]],
        #             [start[2], end[2]], c='g')
        # Labeling the points
        # for i in range(len(self.point_cloud)):
        #     ax.text(self.point_cloud[i, 0], self.point_cloud[i, 1],
        #             self.point_cloud[i, 2], str(i), color='black')

        plt.show()

    def plot_sdf(self, resolution=10):
        # Generate grid points
        x = np.linspace(np.min(self.point_cloud[:, 0]), np.max(
            self.point_cloud[:, 0]), resolution+1)
        y = np.linspace(np.min(self.point_cloud[:, 1]), np.max(
            self.point_cloud[:, 1]), resolution+1)
        z = np.linspace(np.min(self.point_cloud[:, 2]), np.max(
            self.point_cloud[:, 2]), resolution+1)

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
        # get the nearest point and its normal
        nearest_point_index = self.kdtree.query(point)[1]
        nearest_point = self.point_cloud[nearest_point_index]
        normal = self.normals[nearest_point_index]
        # compute the signed distance
        sdf = np.dot(point - nearest_point, normal)

        return sdf

    def _compute_normals(self, point_cloud, k: int):
        # * fit local plane and get normal
        for i in range(len(point_cloud)):
            # get k nearest neighbors
            neighbor_index = self.kdtree.query(point_cloud[i], k)[1]
            neighbor_points = point_cloud[neighbor_index]
            # find centroid
            centroid = np.mean(neighbor_points, axis=0)
            # compute covariance matrix
            covariance_matrix = np.cov(neighbor_points, rowvar=False)
            # get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            # get normal, which is the minimum eigenvector
            normal = eigenvectors[:, np.argmin(eigenvalues)]/np.linalg.norm(
                eigenvectors[:, np.argmin(eigenvalues)]) * random.choice([1, -1])
            self.normals.append(normal)

        # * orient normals
        # build edge graph
        # set the nearest 4 neighbors as the edge
        edge_graph = {}
        for i in range(len(point_cloud)):
            neighbor_index = self.kdtree.query(point_cloud[i], 5)[1][1:]
            if i not in edge_graph:
                edge_graph[i] = list(neighbor_index)
            else:
                edge_graph[i] += list(neighbor_index)
            for j in neighbor_index:
                if j not in edge_graph:
                    edge_graph[j] = []
                edge_graph[j].append(i)
        # define edge weight
        edge_weight = {}
        for i in range(len(point_cloud)):
            for j in edge_graph[i]:
                if (j, i) in edge_weight:
                    continue
                edge_weight[(i, j)] = 1 - \
                    abs(np.dot(self.normals[i], self.normals[j]))
        # normal propagation
        mst = self._minimum_spanning_tree(edge_graph, edge_weight)
        self.MST = mst
        stack = []
        stack.append(0)
        while stack:
            u = stack.pop()
            for v in edge_graph[u]:
                if (u, v) in mst or (v, u) in mst:
                    stack.append(v)
                    if (u, v) in mst:
                        mst.remove((u, v))
                    else:
                        mst.remove((v, u))
                    if np.dot(self.normals[u], self.normals[v]) < 0:
                        self.normals[v] = - self.normals[v]

    def _minimum_spanning_tree(self, edge_graph: dict, edge_weight: dict) -> list:
        # * Kruskal's algorithm
        edges = sorted(edge_weight.keys(), key=lambda x: edge_weight[x])
        mst = []
        uf = UnionFind()
        for edge in edges:
            u, v = edge
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                mst.append(edge)

        return mst

# test


def generate_sphere_point_cloud(radius, num_points):
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    point_cloud = np.column_stack((x, y, z))
    return point_cloud


def generate_plane_point_cloud(width, height, num_points):
    x = np.random.uniform(0, width, num_points)
    y = np.random.uniform(0, height, num_points)
    z = np.zeros(num_points)
    point_cloud = np.column_stack((x, y, z))
    return point_cloud


if __name__ == '__main__':

    radius = 1.0
    num_points = 1000
    point_cloud = generate_sphere_point_cloud(radius, num_points)
    # point_cloud = generate_plane_point_cloud(2, 2, num_points)
    sdf = Naive_SDF(point_cloud)
    # sdf.render_figure()
    sdf.plot_sdf()
