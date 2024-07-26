import numpy as np
import trimesh
from sklearn.decomposition import PCA


def load_stl(file_path):
    """
    Loads an STL file and extracts the mesh vertices.

    Parameters:
    file_path (Path or str): The path to the STL file.

    Returns:
    tuple: A tuple containing the points (vertices) of the mesh and the mesh object itself.

    Explanation:
    - Uses the `trimesh` library to load the STL file.
    - Extracts the vertices (points) from the mesh.
    """
    mesh = trimesh.load(file_path)
    points = mesh.vertices
    return points, mesh


def apply_pca(points):
    """
    Applies Principal Component Analysis (PCA) to the points.

    Parameters:
    points (ndarray): The vertices of the mesh.

    Returns:
    PCA: The fitted PCA object.

    Explanation:
    - PCA is used to identify the main axes of variation in the data.
    - The `sklearn.decomposition.PCA` class is used to perform PCA.
    - The number of components is set to 3 to match the 3D nature of the points.
    """
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca


def align_with_axis(points, pca):
    """
    Aligns the points with a specified axis using the PCA components.

    Parameters:
    points (ndarray): The vertices of the mesh.
    pca (PCA): The fitted PCA object.

    Returns:
    ndarray: The aligned points.

    Explanation:
    - The first principal component (main axis) is aligned with the Y-axis.
    - A rotation matrix is computed to align the main axis with the target axis.
    - The points are rotated using this rotation matrix.
    """
    components = pca.components_
    main_axis = components[0]
    target_axis = np.array([0, 1, 0])  # Align with Y-axis
    rot_matrix = rotation_matrix_from_vectors(main_axis, target_axis)
    aligned_points = np.dot(points - pca.mean_, rot_matrix.T) + pca.mean_
    return aligned_points


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Computes the rotation matrix that aligns vec1 to vec2.

    Parameters:
    vec1 (ndarray): The source vector.
    vec2 (ndarray): The target vector.

    Returns:
    ndarray: The rotation matrix.

    Explanation:
    - Uses the cross product to find the axis of rotation.
    - Uses the dot product to find the cosine of the angle of rotation.
    - Constructs the rotation matrix using the Rodrigues' rotation formula.
    """
    """
    Rodrigues' rotation formula is a method for rotating a vector in three-dimensional space. It provides a way to compute the rotation matrix given an axis of rotation and an angle. The formula is particularly useful for converting between axis-angle representation and rotation matrices.  Given a unit vector k (the axis of rotation) and an angle θ (the angle of rotation), the rotation matrix R can be computed as:  [ R = I + \sin(\theta) \cdot K + (1 - \cos(\theta)) \cdot K^2 ]  where:  
    I is the identity matrix.
    K is the skew-symmetric matrix of k: [ K = \begin{bmatrix} 0 & -k_z & k_y \ k_z & 0 & -k_x \ -k_y & k_x & 0 \end{bmatrix} ]
    In this context:  
    k_x, k_y, and k_z are the components of the unit vector k.
    K^2 is the matrix multiplication of K with itself.
    The formula effectively combines the identity matrix, the skew-symmetric matrix, and the squared skew-symmetric matrix to produce the rotation matrix. This matrix can then be used to rotate any vector around the axis k by the angle θ.
    """

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def move_center_to_origin(points):
    """
    Moves the center of the points to the origin.

    Parameters:
    points (ndarray): The vertices of the mesh.

    Returns:
    ndarray: The centered points.

    Explanation:
    - Computes the bounding box of the points.
    - Finds the center of the bounding box.
    - Translates the points so that the center is at the origin.
    """
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_center = (bbox_max + bbox_min) / 2.0
    centered_points = points - bbox_center
    return centered_points


def save_aligned_stl(points, mesh, file_path):
    """
    Saves the aligned points back to an STL file.

    Parameters:
    points (ndarray): The aligned vertices of the mesh.
    mesh (trimesh.Trimesh): The mesh object.
    file_path (Path or str): The path to save the STL file.

    Explanation:
    - Updates the vertices of the mesh with the aligned points.
    - Uses the `trimesh` library to export the mesh to an STL file.
    """
    mesh.vertices = points
    mesh.export(file_path)


def process_file(input_path, output_path):
    """
    Processes an STL file by loading, aligning, centering, and saving it.

    Parameters:
    input_path (Path or str): The path to the input STL file.
    output_path (Path or str): The path to save the processed STL file.

    Explanation:
    - Loads the STL file and extracts the points.
    - Applies PCA to find the main axes of variation.
    - Aligns the points with the Y-axis.
    - Moves the center of the points to the origin.
    - Saves the processed points back to an STL file.
    """
    points, mesh = load_stl(input_path)
    pca = apply_pca(points)
    aligned_points = align_with_axis(points, pca)
    centered_points = move_center_to_origin(aligned_points)
    save_aligned_stl(centered_points, mesh, output_path)
