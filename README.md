# ZIM - KI Assistenz Project

## AI-Based Assistance System for Manufacturing Custom Ventilation Masks

This repository contains the code and documentation for the ZIM - KI Assistenz project, a collaborative initiative between the University of Applied Sciences Cologne (TH Köln) and local SMEs in North Rhine-Westphalia (NRW). The project aims to modernize and optimize the manufacturing process of custom-made ventilation masks using artificial intelligence (AI) and advanced technologies. The primary goal is to reduce production time and costs while maintaining the unique customization required for each mask.

## Key Objectives
- Develop an AI-based assistance and feedback system for the production of custom ventilation masks.
- Integrate AI and Big Data to automate feature extraction from 3D scans.
- Improve efficiency and reduce costs in the manufacturing process without disrupting existing workflows.

## Methodology

### Data Preparation and Normalization

#### Tool Familiarization
- **Cloud Compare**: Download and use Cloud Compare for managing point cloud files.
- **OpenCV**: Utilize facial recognition libraries in OpenCV, focusing on detecting facial profiles and specific features like the nose.

#### Facial Region Delimitation
- Delimit the facial region from the nose to the chin (the anatomical boundary) and use this to scale facial masks appropriately.

### Handling Point Clouds

#### Point Cloud Management
- Work with both dense (high-resolution) and light (low-resolution) point clouds.
- Use the octree-based method to divide the 3D space into small cubes and normalize the scan coordinates to determine occupied cubes.

#### Scale Adjustment and Normalization
- Manipulate scales and adjust dimensions using real-world data, applying PCA to identify the longest axis for alignment.

### Algorithm Development

#### Feature Detection and Delimitation
- Develop algorithms to detect and delimit facial features.
- Implement techniques to handle and normalize point clouds using octrees.

#### Scan Alignment
- Align facial scans based on PCA and the overall height of the face.
- Propose aligning specifically the regions covered by the mask (nasal, perioral, and chin areas) using the nasolabial angle and mapping points of nasal wings and mouth corners to create a subnasal-perioral trapezoid.

## Technical Concepts and Definitions

- **ZIM (Zentrales Innovationsprogramm Mittelstand)**: A funding program by the German government to support SMEs in innovation.
- **Artificial Intelligence (AI)**: The simulation of human intelligence in machines that are programmed to think and learn.
- **Neural Networks**: A set of algorithms modeled after the human brain, designed to recognize patterns and interpret data.
- **Data Normalization**: The process of organizing data to meet certain standards and formats, making it suitable for analysis by neural networks.
- **Point Cloud**: A collection of data points in space, often produced by 3D scanners, representing the external surface of an object.
- **Cloud Compare**: An open-source software for 3D point cloud and mesh processing.
- **OpenCV**: An open-source computer vision and machine learning software library.
- **Octree**: A tree structure used to partition a three-dimensional space by recursively subdividing it into eight octants.
- **Principal Component Analysis (PCA)**: A statistical technique used to emphasize variation and bring out strong patterns in a dataset, often used to reduce the dimensionality of data.
- **Nasolabial Angle**: The angle formed between the nose and the upper lip, significant in facial feature alignment.
- **Industry 4.0**: The ongoing automation of traditional manufacturing and industrial practices using modern smart technology.

## References
- TH Köln. (n.d.). KI-Assistenz – Assistenz- und Feedbacksystem auf Basis von Big Data und künstlicher Intelligenz zur Vorbereitung des Fertigungsprozesses von individuellen Beatmungsmasken auf Fertigung 4.0. Retrieved from: https://www.th-koeln.de/anlagen-energie-und-maschinensysteme/ki-assistenz_110308.php
- AirTec Beatmungshilfen GmbH & Co. KG. (n.d.). Willkommen bei AirTec Beatmungshilfen GmbH & Co. KG. Retrieved from: https://airtec-beatmungshilfen.de/