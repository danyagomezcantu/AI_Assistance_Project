import dlib
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import pymeshlab


# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def load_stl(file_path):
    """
    Loads an STL file and extracts the points and mesh.

    Parameters:
    file_path (Path or str): The path to the STL file.

    Returns:
    tuple: A tuple containing the points (ndarray) and the mesh (trimesh.Trimesh).
    """
    mesh = trimesh.load(file_path)
    points = mesh.vertices
    return points, mesh


def apply_pca(points):
    """
    Applies PCA to the points to find the main axes of variation.

    Parameters:
    points (ndarray): The points of the mesh.

    Returns:
    PCA: The PCA object after fitting the points.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca


def align_with_axis(points, pca):
    """
    Aligns the points with the Y-axis based on PCA.

    Parameters:
    points (ndarray): The points of the mesh.
    pca (PCA): The PCA object after fitting the points.

    Returns:
    ndarray: The aligned points.
    """
    aligned_points = np.dot(points - pca.mean_, pca.components_.T)
    return aligned_points


def move_center_to_origin(points):
    """
    Moves the center of the points to the origin.

    Parameters:
    points (ndarray): The points of the mesh.

    Returns:
    ndarray: The centered points.
    """
    center = np.mean(points, axis=0)
    centered_points = points - center
    return centered_points


def detect_landmarks(image):
    """
    Detects facial landmarks in the given image.

    Parameters:
    image (ndarray): The input image.

    Returns:
    list: A list of tuples containing the coordinates of the facial landmarks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None

    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        for i in range(0, 68):
            landmarks.append((shape.part(i).x, shape.part(i).y))
    return landmarks


def is_upside_down(landmarks):
    """
    Determines if the face is upside down based on the landmarks.

    Parameters:
    landmarks (list): A list of tuples containing the coordinates of the facial landmarks.

    Returns:
    bool: True if the face is upside down, False otherwise.
    """
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    mouth = np.mean(landmarks[48:68], axis=0)

    return mouth[1] < (left_eye[1] + right_eye[1]) / 2


def rotate_mesh(points, angle):
    """
    Rotates the mesh points by the given angle.

    Parameters:
    points (ndarray): The mesh points.
    angle (float): The angle to rotate the mesh.

    Returns:
    ndarray: The rotated mesh points.
    """
    rotation_matrix = R.from_euler('z', angle, degrees=True).as_matrix()
    return np.dot(points, rotation_matrix.T)


def mesh_to_image(mesh):
    """
    Converts a 3D mesh to a 2D image for landmark detection.

    Parameters:
    mesh (trimesh.Trimesh): The mesh object.

    Returns:
    ndarray: The 2D image of the mesh.
    """
    # This function needs to be implemented based on your specific requirements
    pass


def correct_orientation(points, image):
    """
    Corrects the orientation of the face in the mesh if it is upside down.

    Parameters:
    points (ndarray): The mesh points.
    image (ndarray): The 2D image of the mesh.

    Returns:
    ndarray: The corrected mesh points.
    """
    landmarks = detect_landmarks(image)
    if landmarks and is_upside_down(landmarks):
        points = rotate_mesh(points, 180)
    return points


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


def remesh(mesh, target_edge_length):
    """
    Remesh the mesh to have a more uniform resolution.

    Parameters:
    mesh (trimesh.Trimesh): The mesh object.
    target_edge_length (float): The target edge length for remeshing.

    Returns:
    trimesh.Trimesh: The remeshed object.
    """
    # Create a MeshSet object
    ms = pymeshlab.MeshSet()
    # Add the mesh to the MeshSet
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    # Apply remeshing with a target edge length
    ms.meshing_isotropic_explicit_remeshing(targetlen=target_edge_length)
    # Retrieve the remeshed mesh
    remeshed_mesh = ms.current_mesh()
    return trimesh.Trimesh(vertices=remeshed_mesh.vertex_matrix(), faces=remeshed_mesh.face_matrix())


def process_file(input_path, output_path, target_edge_length=1.0):
    """
    Processes an STL file by loading, aligning, centering, remeshing, and saving it.

    Parameters:
    input_path (Path or str): The path to the input STL file.
    output_path (Path or str): The path to save the processed STL file.
    target_edge_length (float): The target edge length for remeshing.

    Explanation:
    - Loads the STL file and extracts the points.
    - Applies PCA to find the main axes of variation.
    - Aligns the points with the Y-axis.
    - Moves the center of the points to the origin.
    - Corrects the orientation of the face.
    - Remeshes the mesh to have a more uniform resolution.
    - Saves the processed points back to an STL file.
    """
    points, mesh = load_stl(input_path)
    pca = apply_pca(points)
    aligned_points = align_with_axis(points, pca)
    centered_points = move_center_to_origin(aligned_points)

    # Convert mesh to 2D image for landmark detection
    image = mesh_to_image(mesh)
    corrected_points = correct_orientation(centered_points, image)

    # Remesh the mesh
    remeshed_mesh = remesh(mesh, target_edge_length)

    save_aligned_stl(remeshed_mesh.vertices, remeshed_mesh, output_path)