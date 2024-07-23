import numpy as np
import trimesh
from sklearn.decomposition import PCA


def load_stl(file_path):
    mesh = trimesh.load(file_path)
    points = mesh.vertices
    return points, mesh


def apply_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca


def align_with_axis(points, pca):
    # The PCA components
    components = pca.components_
    # Assuming the first component is the longest axis
    main_axis = components[0]

    # Determine the rotation to align with the Y axis
    target_axis = np.array([0, 1, 0])

    # Compute rotation matrix
    rot_matrix = rotation_matrix_from_vectors(main_axis, target_axis)

    # Apply rotation
    aligned_points = np.dot(points - pca.mean_, rot_matrix.T) + pca.mean_

    return aligned_points


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3)
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def move_center_to_origin(points):
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_center = (bbox_max + bbox_min) / 2.0
    centered_points = points - bbox_center
    return centered_points


def save_aligned_stl(points, mesh, file_path):
    mesh.vertices = points
    mesh.export(file_path)


def process_file(file_path, output_path):
    points, mesh = load_stl(file_path)
    pca = apply_pca(points)
    aligned_points = align_with_axis(points, pca)
    centered_points = move_center_to_origin(aligned_points)
    save_aligned_stl(centered_points, mesh, output_path)


# Example usage
input_file = "C:/Users/Danya/Downloads/KI_Test/12413.12.22.stl"
output_file = "C:/Users/Danya/Downloads/KI_Test/12413.12.22.0036.stl"
process_file(input_file, output_file)
