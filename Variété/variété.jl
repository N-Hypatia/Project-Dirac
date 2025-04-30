using Pkg
Pkg.add(["Plots", "Manifolds", "DimensionalityReduction", "Makie", "LinearAlgebra"])
# DimensionalityReduction for t-SNE, PCA, UMAP.
using Plots, Manifolds, DimensionalityReduction, Makie, LinearAlgebra

# Simple Manifold, Swiss Roll
n_samples = 1000 # Number of points
# Defines the number of points to sample on the manifold.
# More points improve visualization but increase computation time.

# Creates a vector of n_samples evenly spaced values between 0 and 3π.
# This vector will be used as a parameter for the spiral.
t = range(0, stop=3π, length=n_samples) # Parameter for spiral

# Creates a vector of n_samples random values between 0 and 10.
# h simulates the height of the points on the manifold.
h = rand(n_samples) * 10 # Height parameter

# Generate points on the Swiss Roll
function swiss_roll(t, h)
    # Computes x-coordinate using the formula t * cos(t)
    # Creating the spiral in the x-z plane
    x = t .* cos.(t)
    # y-coordinate is simply the height parameter h
    y = h 
    # Computes z-coordinate using the formula t * sin(t)
    # Creating the spiral in the x-z plane
    z = t .*sint.(t)
    # hcat concatenates the vectors x, y, and z horizontally
    # Combines x, y, z into a single matrix
    # Each row represents a point in 3D space
    return hcat(x, y, z) # Returns n_samples x 3 matrix
end

data = swiss_roll(t, h)

# Visualization

# Let's Create a 3D scatter plot of the points on the Swiss Roll
scatter(
    # Extracting the x, y, and z coordinates from the data matrix
    # data[:, 1] selects the first column (x-coordinates), all rows
    data[:, 1], data[:, 2], data[:, 3],
    title="Swiss Roll Manifold", 
    xlabel="X", ylabel="Y", zlabel="Z",
    markersize=2, color=:blue
    # markersize sets the size of the points in the scatter plot
    # color sets the color of the points
)

# Dimensionality Reduction (PCA)

# Let's perform PCA on the data to reduce it to 2 dimensions
# Projecting the 3D data onto a 2D plane

# Center the data
data_centered = data .- mean(data, dims=1)
# PCA requires zero-mean data to focus on variance.

# Compute the covariance matrix
# The covariance matrix captures the variance and correlation between dimensions,
# resulting in a 3 x 3 matrix for 3D data
cov_matrix = cov(data_centered)

# Compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = eigen(cov_matrix)

principal_components = eigenvectors[:, end:-1:end-1] # Selecting the first two eigenvectors
projected_data = data_centered * principal_components

# Visualization of PCA
# 2D scatter plot of the projected data
scatter(
    projected_data[:, 1], projected_data[:, 2],
    title="PCA of Swiss Roll", 
    xlabel="Principal Component 1", ylabel="Principal Component 2",
    markersize=2, color=:red
)