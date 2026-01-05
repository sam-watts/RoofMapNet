import numpy as np
from pathlib import Path
from skimage.draw import line_aa

def preprocess_roof_lines(lines, output_filename, image_size=512, heatmap_size=128):
    """
    Convert labelled lines to RoofMapNet format.
    
    Args:
        lines: List of line segments, each as [(y1,x1), (y2,x2)]
        image_size: Original image dimensions
        heatmap_size: Size of the output heatmaps
    """
    scale = heatmap_size / image_size
    
    # 1. Extract junctions (unique endpoints)
    junctions = []
    junction_map = {}
    
    for line in lines:
        for point in line:
            y, x = point
            y_scaled = y * scale
            x_scaled = x * scale
            
            # Snap to integer pixel grid and check for duplicates
            key = (round(y_scaled), round(x_scaled))
            if key not in junction_map:
                junction_map[key] = len(junctions)
                # Add junction type (0 for now)
                junctions.append([y_scaled, x_scaled, 0])
    
    junc = np.array(junctions)
    
    # 2. Create junction heat maps
    jmap = np.zeros((1, heatmap_size, heatmap_size))  # Single type for now
    joff = np.zeros((1, 2, heatmap_size, heatmap_size))
    
    for y, x, t in junctions:
        yi, xi = int(y), int(x)
        if 0 <= yi < heatmap_size and 0 <= xi < heatmap_size:
            # Heat map with Gaussian
            jmap[int(t), yi, xi] = 1.0
            # Sub-pixel offset
            joff[int(t), 0, yi, xi] = y - yi
            joff[int(t), 1, yi, xi] = x - xi
    
    
    # 3. Create line heat map
    lmap = np.zeros((heatmap_size, heatmap_size))
    
    for line in lines:
        y1, x1 = line[0][0] * scale, line[0][1] * scale
        y2, x2 = line[1][0] * scale, line[1][1] * scale
        
        # Draw anti-aliased line
        rr, cc, val = line_aa(int(y1), int(x1), int(y2), int(x2))
        valid = (rr >= 0) & (rr < heatmap_size) & (cc >= 0) & (cc < heatmap_size)
        lmap[rr[valid], cc[valid]] = np.maximum(lmap[rr[valid], cc[valid]], val[valid])
    
    # 4. Create line connectivity
    lpos = []
    Lpos = []
    
    for line in lines:
        # Get junction indices for this line's endpoints
        p1 = (round(line[0][0] * scale), round(line[0][1] * scale))
        p2 = (round(line[1][0] * scale), round(line[1][1] * scale))
        
        if p1 in junction_map and p2 in junction_map:
            idx1, idx2 = junction_map[p1], junction_map[p2]
            Lpos.append([idx1, idx2])
            lpos.append([junctions[idx1], junctions[idx2]])
        else:
            print("Warning: Line endpoints not found in junctions.")
    
    # 5. Generate all possible negative lines (non-connections)
    lneg, Lneg = generate_all_negative_lines(junctions, Lpos)
    
    # Save as NPZ
    np.savez(
        output_filename,
        jmap=jmap.astype(np.float32),
        joff=joff.astype(np.float32),
        lmap=lmap.astype(np.float32),
        junc=np.array(junc).astype(np.float32),
        Lpos=np.array(Lpos).astype(np.int32),
        Lneg=np.array(Lneg).astype(np.int32),
        lpos=np.array(lpos).astype(np.float32),
        lneg=np.array(lneg).astype(np.float32)
    )

def generate_all_negative_lines(junctions, positive_pairs):
    """Generate all possible junction pairs that aren't actual lines."""
    n_junc = len(junctions)
    positive_set = set(map(tuple, positive_pairs))
    
    lneg, Lneg = [], []
    
    # Generate all possible pairs (i, j) where i < j
    for i in range(n_junc):
        for j in range(i + 1, n_junc):
            # Check if this pair is not in the positive set
            if (i, j) not in positive_set and (j, i) not in positive_set:
                Lneg.append([i, j])
                lneg.append([junctions[i], junctions[j]])
    
    return np.array(lneg), np.array(Lneg)