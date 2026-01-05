"""
This module converts RID2 roof segment outlines to oriented bounding boxes (OBBs) for segment edges.

For each segment, we extract the individual edges (line segments) that form the segment boundary, 
deduplicate them within each roof, and compute an oriented bounding box for each unique edge.

This is useful for edge detection tasks where we want to detect the boundaries between roof segments
rather than the filled segment areas.

Data is stored in .txt files in OBB format:
class_index x1 y1 x2 y2 x3 y3 x4 y4

where (xi, yi) are the coordinates of the four corners of the OBB in clockwise order starting from top-left corner.
The class_index is set to 0 for all edges as we only have one class (for now!).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Polygon
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# Module constants for image dimensions
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

dataset_root = Path("~/datasets/roof_information_dataset_2").expanduser()
# RID2_SOURCE_IMAGE_DIR = dataset_root / "case_study_roof_centered" / "images_roof_centered"
RID2_SOURCE_IMAGE_DIR = dataset_root / "images"
geometry_dir = dataset_root / "geometries"
preprocessing_output = dataset_root / "preprocessing_output"
RID2_SOURCE_LABEL_DIR = preprocessing_output / "edge_labels"
mapping_file = preprocessing_output / "image_geometry_mapping_grid.json"

RID2_SOURCE_LABEL_DIR.mkdir(parents=True, exist_ok=True)


def load_mapping_file() -> Dict:
    """Load the image to geometry mapping file."""
    with open(mapping_file, 'r') as f:
        return json.load(f)


def load_segments_geometry() -> Dict:
    """Load the segments geometry data from the GeoJSON file."""
    segments_file = geometry_dir / "gdf_all_segments.json"
    with open(segments_file, 'r') as f:
        data = json.load(f)
    
    return data["features"]


def extract_polygon_coordinates(geometry: Dict) -> List[List[Tuple[float, float]]]:
    """
    Extract coordinates from a GeoJSON polygon or multipolygon geometry.
    
    Returns:
        List of coordinate lists, where each coordinate list represents one polygon.
        For Polygon: returns a list with one coordinate list.
        For MultiPolygon: returns a list with multiple coordinate lists.
    """
    if geometry['type'] == 'Polygon':
        # Get the exterior ring coordinates (first array in coordinates)
        exterior_coords = geometry['coordinates'][0]
        
        # Remove the last coordinate if it's the same as the first (closed polygon)
        if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
            exterior_coords = exterior_coords[:-1]
        
        coords = [(coord[0], coord[1]) for coord in exterior_coords]
        return [coords]  # Return as list of coordinate lists
        
    elif geometry['type'] == 'MultiPolygon':
        all_polygons = []
        # MultiPolygon coordinates are an array of polygon coordinate arrays
        for polygon_coords in geometry['coordinates']:
            # Get the exterior ring (first array) of each polygon
            exterior_coords = polygon_coords[0]
            
            # Remove the last coordinate if it's the same as the first (closed polygon)
            if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            
            coords = [(coord[0], coord[1]) for coord in exterior_coords]
            all_polygons.append(coords)
        
        return all_polygons
    
    else:
        raise ValueError(f"Expected Polygon or MultiPolygon geometry, got {geometry['type']}")


def extract_segment_edges(coords: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
    """
    Extract individual edges from a polygon outline.
    
    Args:
        coords: List of polygon vertices
        
    Returns:
        List of edges, where each edge is a list of two points
    """
    if len(coords) < 3:
        return []
    
    edges = []
    for i in range(len(coords)):
        # Get current point and next point (wrapping around)
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]
        edges.append([p1, p2])
    
    return edges


def deduplicate_edges(edges: List[List[Tuple[float, float]]], tolerance: float = 1e-3) -> List[List[Tuple[float, float]]]:
    """
    Deduplicate edges by checking if they are the same line segment (allowing for reverse direction).
    
    Args:
        edges: List of edges, where each edge is [point1, point2]
        tolerance: Distance tolerance for considering edges the same
        
    Returns:
        List of unique edges
    """
    if not edges:
        return []
    
    unique_edges = []
    
    for edge in edges:
        if len(edge) != 2:
            continue
            
        p1, p2 = edge
        is_duplicate = False
        
        for unique_edge in unique_edges:
            if len(unique_edge) != 2:
                continue
                
            u1, u2 = unique_edge
            
            # Check if edges are the same (in either direction)
            # Edge A->B is the same as edge B->A
            dist1 = ((p1[0] - u1[0])**2 + (p1[1] - u1[1])**2)**0.5 + ((p2[0] - u2[0])**2 + (p2[1] - u2[1])**2)**0.5
            dist2 = ((p1[0] - u2[0])**2 + (p1[1] - u2[1])**2)**0.5 + ((p2[0] - u1[0])**2 + (p2[1] - u1[1])**2)**0.5
            
            if min(dist1, dist2) < tolerance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_edges.append(edge)
    
    return unique_edges


def convert_world_to_image_coords(world_coords: List[Tuple[float, float]], 
                                 image_bounds: List[List[float]], 
                                 image_size: int = IMAGE_WIDTH) -> List[Tuple[float, float]]:
    """
    Convert world coordinates to image coordinates using the correct image bounds.
    
    This uses the same logic as the mask_generation.py module to ensure consistency.
    
    Args:
        world_coords: List of (x, y) world coordinates
        image_bounds: Image bounds polygon from the mapping file
        image_size: Size of the image (assumed square)
    
    Returns:
        List of (x, y) image coordinates
    """
    # Extract bounds from polygon
    x_coords = [point[0] for point in image_bounds]
    y_coords = [point[1] for point in image_bounds]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    image_coords = []
    for world_x, world_y in world_coords:
        # Transform to pixel coordinates
        x_pixel = ((world_x - min_x) / (max_x - min_x)) * image_size
        y_pixel = ((max_y - world_y) / (max_y - min_y)) * image_size
        
        # Keep as float for OBB coordinates (don't clamp to integer bounds)
        image_coords.append((x_pixel, y_pixel))
    
    return image_coords



def save_edge_file(image_name: str, edge_list: List[List[Tuple[float, float]]]) -> None:
    """Save edge data to a text file in the required format with normalized coordinates."""
    output_file = RID2_SOURCE_LABEL_DIR / f"{image_name.replace('.png', '.npz')}"
    
    valid_edges = 0
    clipped_edges = 0
    filtered_edges = 0
    
    np.savez_compressed(output_file, edges=np.array(edge_list))
        
        
    if clipped_edges > 0:
        print(f"  Clipped {clipped_edges} edges that extended beyond image boundaries")
    if filtered_edges > 0:
        print(f"  Filtered {filtered_edges} edges that were completely outside image bounds")


def process_single_image(image_name: str, segment_ids: List[int], segments_geometry: list, image_bounds: List[List[float]]) -> None:
    """Process a single image and generate OBB labels for its segment edges."""
    
    # Extract all edges from all segments in this image
    all_edges = []
    for segment_id in segment_ids:
        segment_data = segments_geometry[segment_id]
        try:
            # Extract coordinates (returns list of coordinate lists for MultiPolygon support)
            polygon_coords_list = extract_polygon_coordinates(segment_data['geometry'])
            
            # Process each polygon in the segment (handles both Polygon and MultiPolygon)
            for coords in polygon_coords_list:
                edges = extract_segment_edges(coords)
                all_edges.extend(edges)
                
        except Exception as e:
            print(f"Warning: Error processing segment {segment_id}: {e}")

    if not all_edges:
        print(f"Warning: No valid edges found for image {image_name}")
        return
    
    # Deduplicate edges within this roof
    unique_edges = deduplicate_edges(all_edges)
    converted_edges = [convert_world_to_image_coords(e, image_bounds) for e in unique_edges]
    
    if converted_edges:
        save_edge_file(image_name, converted_edges)
    else:
        print(f"Warning: No valid edge OBBs generated for image {image_name}")


def main():
    """Main function to process all roofs in the dataset."""
    print("Loading mapping file...")
    mapping_data = load_mapping_file()
    
    print("Loading segments geometry...")
    segments_geometry = load_segments_geometry()
    
    print(f"Found {len(mapping_data)} images to process")
    print(f"Loaded geometry data for {len(segments_geometry)} segments")
    
    # Process each image
    processed_count = 0
    for image_name, image_data in tqdm(mapping_data.items()):
        # Add .png extension if not present
        if not image_name.endswith('.png'):
            image_name_with_ext = f"{image_name}.png"
        else:
            image_name_with_ext = image_name
        
        # Check if image file exists
        image_path = RID2_SOURCE_IMAGE_DIR / image_name_with_ext
        if not image_path.exists():
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        # Get segment IDs and image bounds for this image
        segment_ids = image_data.get('segments', [])
        image_bounds = image_data.get('image_bounds', [])
        
        if not segment_ids:
            print(f"No segments found for image {image_name_with_ext}")
            continue
            
        if not image_bounds:
            print(f"No image bounds found for image {image_name_with_ext}")
            continue
        
        try:
            process_single_image(image_name_with_ext, segment_ids, segments_geometry, image_bounds)
            processed_count += 1
        except Exception as e:
            print(f"Error processing image {image_name_with_ext}: {e}")

    
    print(f"\nProcessing complete. Processed {processed_count} images.")
    print(f"OBB labels saved to: {RID2_SOURCE_LABEL_DIR}")
    
   


if __name__ == "__main__":
    main()