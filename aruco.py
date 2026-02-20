#!/usr/bin/env python3
"""
ArUco Marker Generator

This script generates ArUco markers using OpenCV.
ArUco markers are binary square fiducial markers commonly used in computer vision
for camera calibration, pose estimation, and object tracking.
"""

import cv2
import numpy as np
import argparse
import os


def generate_aruco_marker(marker_id, dictionary_name="DICT_5X5_50", marker_size=50, border_bits=1):
    """
    Generate a single ArUco marker.
    
    Args:
        marker_id: ID of the marker to generate (must be valid for the chosen dictionary)
        dictionary_name: Name of the ArUco dictionary to use
        marker_size: Size of the marker in pixels
        border_bits: Number of border bits (default is 1)
    
    Returns:
        numpy array containing the marker image
    """
    # Dictionary of available ArUco dictionaries
    aruco_dicts = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    
    if dictionary_name not in aruco_dicts:
        raise ValueError(f"Invalid dictionary name. Choose from: {list(aruco_dicts.keys())}")
    
    # Get the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dicts[dictionary_name])
    
    # Generate the marker
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, borderBits=border_bits)
    
    return marker_image


def generate_multiple_markers(start_id, count, dictionary_name="DICT_5X5_50", 
                              marker_size=50, border_bits=1, output_dir="aruco_markers"):
    """
    Generate multiple ArUco markers and save them to files.
    
    Args:
        start_id: Starting marker ID
        count: Number of markers to generate
        dictionary_name: Name of the ArUco dictionary to use
        marker_size: Size of each marker in pixels
        border_bits: Number of border bits
        output_dir: Directory to save the markers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {count} ArUco markers...")
    print(f"Dictionary: {dictionary_name}")
    print(f"Marker size: {marker_size}x{marker_size} pixels")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    for i in range(count):
        marker_id = start_id + i
        try:
            marker = generate_aruco_marker(marker_id, dictionary_name, marker_size, border_bits)
            
            # Save the marker
            filename = os.path.join(output_dir, f"aruco_{dictionary_name}_id_{marker_id}.png")
            cv2.imwrite(filename, marker)
            print(f"Generated marker ID {marker_id}: {filename}")
            
        except Exception as e:
            print(f"Error generating marker ID {marker_id}: {e}")
    
    print("-" * 50)
    print(f"Done! Generated {count} markers in '{output_dir}/'")


def create_marker_sheet(marker_ids, dictionary_name="DICT_5X5_50", marker_size=50,
                        cols=4, spacing=50, border_bits=1, output_file="marker_sheet.png"):
    """
    Create a sheet with multiple ArUco markers arranged in a grid.
    
    Args:
        marker_ids: List of marker IDs to include
        dictionary_name: Name of the ArUco dictionary to use
        marker_size: Size of each marker in pixels
        cols: Number of columns in the grid
        spacing: Spacing between markers in pixels
        border_bits: Number of border bits
        output_file: Output filename
    """
    rows = (len(marker_ids) + cols - 1) // cols
    
    # Calculate sheet dimensions
    sheet_width = cols * marker_size + (cols + 1) * spacing
    sheet_height = rows * marker_size + (rows + 1) * spacing
    
    # Create white canvas
    sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
    
    print(f"Creating marker sheet with {len(marker_ids)} markers...")
    print(f"Grid: {rows}x{cols}")
    print(f"Sheet size: {sheet_width}x{sheet_height} pixels")
    
    # Place markers on the sheet
    for idx, marker_id in enumerate(marker_ids):
        row = idx // cols
        col = idx % cols
        
        # Generate marker
        marker = generate_aruco_marker(marker_id, dictionary_name, marker_size, border_bits)
        
        # Calculate position
        y_pos = spacing + row * (marker_size + spacing)
        x_pos = spacing + col * (marker_size + spacing)
        
        # Place marker on sheet
        sheet[y_pos:y_pos + marker_size, x_pos:x_pos + marker_size] = marker
        
        # Add marker ID text below the marker
        text = f"ID: {marker_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        text_x = x_pos + (marker_size - text_size[0]) // 2
        text_y = y_pos + marker_size + 20
        
        if text_y < sheet_height:
            cv2.putText(sheet, text, (text_x, text_y), font, 0.5, 0, 1, cv2.LINE_AA)
    
    # Save the sheet
    cv2.imwrite(output_file, sheet)
    print(f"Marker sheet saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco markers for computer vision applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a single marker with ID 0
  python generate_aruco.py --id 0
  
  # Generate markers with IDs 0-9
  python generate_aruco.py --start 0 --count 10
  
  # Generate a 4x4 marker sheet with IDs 0-15
  python generate_aruco.py --sheet --ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  
  # Use a different dictionary
  python generate_aruco.py --id 0 --dict DICT_6X6_250 --size 300
        """
    )
    
    parser.add_argument("--id", type=int, help="Generate a single marker with this ID")
    parser.add_argument("--start", type=int, default=0, help="Starting marker ID for batch generation")
    parser.add_argument("--count", type=int, help="Number of markers to generate")
    parser.add_argument("--dict", default="DICT_4X4_50", 
                       help="ArUco dictionary (e.g., DICT_4X4_50, DICT_6X6_250)")
    parser.add_argument("--size", type=int, default=200, help="Marker size in pixels")
    parser.add_argument("--border", type=int, default=1, help="Border width in bits")
    parser.add_argument("--output-dir", default="aruco_markers", help="Output directory")
    parser.add_argument("--sheet", action="store_true", help="Generate a marker sheet")
    parser.add_argument("--ids", type=int, nargs="+", help="Marker IDs for sheet generation")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in sheet")
    parser.add_argument("--spacing", type=int, default=50, help="Spacing between markers in sheet")
    
    args = parser.parse_args()
    
    if args.sheet:
        # Generate marker sheet
        if not args.ids:
            print("Error: --ids required for sheet generation")
            return
        
        output_file = os.path.join(args.output_dir, f"aruco_sheet_{args.dict}.png")
        os.makedirs(args.output_dir, exist_ok=True)
        create_marker_sheet(args.ids, args.dict, args.size, args.cols, 
                          args.spacing, args.border, output_file)
    
    elif args.id is not None:
        # Generate single marker
        os.makedirs(args.output_dir, exist_ok=True)
        marker = generate_aruco_marker(args.id, args.dict, args.size, args.border)
        filename = os.path.join(args.output_dir, f"aruco_{args.dict}_id_{args.id}.png")
        cv2.imwrite(filename, marker)
        print(f"Generated marker ID {args.id}: {filename}")
    
    elif args.count:
        # Generate multiple markers
        generate_multiple_markers(args.start, args.count, args.dict, 
                                 args.size, args.border, args.output_dir)
    
    else:
        # Default: generate marker with ID 0
        os.makedirs(args.output_dir, exist_ok=True)
        marker = generate_aruco_marker(0, args.dict, args.size, args.border)
        filename = os.path.join(args.output_dir, f"aruco_{args.dict}_id_0.png")
        cv2.imwrite(filename, marker)
        print(f"Generated default marker ID 0: {filename}")
        print("\nTip: Use --help to see all options")


if __name__ == "__main__":
    main()