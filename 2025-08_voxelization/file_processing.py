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

# =========================
# Voxelization pipeline
# =========================

def estimate_voxel_pitch(points, target_cells_longest=200, min_pitch=0.5): # OBS

    # use scaling to not estimate and follow same pitch for every face, using a regular grid

    """
    Choose a uniform voxel size ('pitch') so faces and masks are directly comparable.

    Strategy:
    - Use the longest bounding-box edge / target_cells_longest (≈ target voxels along the longest side).
    - Clamp to a sensible minimum pitch to avoid tiny voxels on small meshes.

    Returns:
        float: voxel edge length in the same units as the STL (often millimeters).
    """
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    extent = bbox_max - bbox_min
    longest = float(np.max(extent))
    pitch = max(longest / float(target_cells_longest), float(min_pitch))
    return pitch


def voxelize_to_mesh(mesh, pitch, fill=True, dilate_iters=0):
    """
    Voxelize a (possibly thin/open) surface and convert the occupancy grid back to a closed mesh.

    Args:
        mesh (trimesh.Trimesh): Input mesh (already aligned & centered is best).
        pitch (float): Edge length of each cubic voxel, identical across faces & masks.
        fill (bool): If True, fills internal cavities so you get a solid shell (helps with open scans).
        dilate_iters (int): Optional morphological dilation (integer voxel steps) if you need a slightly thicker shell.

    Returns:
        trimesh.Trimesh: A watertight, low-poly-ish mesh created from voxels.
    """
    # Create a voxel grid from the surface
    v = mesh.voxelized(pitch=pitch)

    # Optionally fill inner gaps (good for open shells)
    if fill:
        v = v.fill()

    # Optional “thicken” step in voxel space if your shells are too thin
    if dilate_iters and dilate_iters > 0:
        v = v.dilate(dilate_iters)

    # Convert the voxel grid back to a triangle mesh (marching-cubes under the hood)
    voxel_mesh = v.to_mesh()
    voxel_mesh.remove_degenerate_faces()
    voxel_mesh.remove_unreferenced_vertices()
    voxel_mesh.merge_vertices()  # small cleanup
    return voxel_mesh


def process_and_voxelize_file(
    input_path,
    output_path,
    pitch=None,
    target_cells_longest=200,
    min_pitch=0.5,
    fill=True,
    dilate_iters=0
):
    """
    Full pipeline for one STL:
      1) load
      2) PCA-align main axis to Y
      3) center to origin (common frame for face & mask)
      4) choose voxel pitch (if not given) so both classes can be matched 1:1
      5) voxelize & remesh
      6) save as STL

    Args:
        input_path (str/Path): source STL
        output_path (str/Path): destination STL with "_processed.stl"
        pitch (float|None): set a fixed voxel size; if None we estimate from bbox (same code for both face & mask to keep them comparable)
        target_cells_longest (int): if pitch is None, aim for ~this many voxels along the longest extent
        min_pitch (float): floor on voxel size to avoid exploding grids on tiny parts
        fill (bool): fill internal voids in voxel grid
        dilate_iters (int): voxel dilation steps (optional)

    Returns:
        dict: summary with chosen pitch and triangle counts
    """
    # Load + orient + center (your existing steps)
    points, mesh = load_stl(input_path)
    pca = apply_pca(points)
    aligned_points = align_with_axis(points, pca)
    centered_points = move_center_to_origin(aligned_points)
    mesh.vertices = centered_points

    # Decide voxel size (use SAME pitch for face & mask to make them "matchable")
    chosen_pitch = pitch if pitch is not None else estimate_voxel_pitch(
        centered_points,
        target_cells_longest=target_cells_longest,
        min_pitch=min_pitch
    )

    # Voxelize and remesh
    voxel_mesh = voxelize_to_mesh(
        mesh,
        pitch=chosen_pitch,
        fill=fill,
        dilate_iters=dilate_iters
    )

    # Final tidy-up (optional smooth step could be added later if desired)
    voxel_mesh.export(output_path)

    return {
        "pitch": chosen_pitch,
        "input_faces": int(len(mesh.faces)),
        "output_faces": int(len(voxel_mesh.faces))
    }
