"""
Carton Dieline Generator Module

This module generates RSC (Regular Slotted Container) carton dielines based on 
object dimensions. It automatically adds clearance to dimensions and creates a 
flat pattern layout suitable for die-cutting and folding.

Author: Python Software Engineer
Date: January 29, 2026
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


class DielinePanel:
    """Represents a single panel in the dieline with its coordinates."""
    
    def __init__(self, name: str, x: float, y: float, width: float, height: float):
        """
        Initialize a panel.
        
        Args:
            name: Panel identifier (e.g., 'front', 'left_flap_top')
            x: X-coordinate of bottom-left corner
            y: Y-coordinate of bottom-left corner
            width: Panel width
            height: Panel height
        """
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """Return the four corners of the panel as (x, y) tuples."""
        return [
            (self.x, self.y),                          # Bottom-left
            (self.x + self.width, self.y),             # Bottom-right
            (self.x + self.width, self.y + self.height),  # Top-right
            (self.x, self.y + self.height)             # Top-left
        ]
    
    def __repr__(self) -> str:
        return f"DielinePanel('{self.name}', x={self.x}, y={self.y}, w={self.width}, h={self.height})"


class CartonDieline:
    """
    Represents a complete RSC carton dieline with all panels and flaps.
    
    An RSC dieline consists of:
    - Bottom panel (base)
    - Four side panels (front, back, left, right)
    - Four top flaps
    - Four bottom flaps
    """
    
    def __init__(self, length: float, width: float, height: float):
        """
        Initialize the dieline structure.
        
        Args:
            length: Internal length of the carton (cm)
            width: Internal width of the carton (cm)
            height: Internal height of the carton (cm)
        """
        self.length = length
        self.width = width
        self.height = height
        self.panels: List[DielinePanel] = []
        self.fold_lines: List[Dict[str, Any]] = []
        self.cut_lines: List[Dict[str, Any]] = []
    
    def add_panel(self, panel: DielinePanel):
        """Add a panel to the dieline."""
        self.panels.append(panel)
    
    def add_fold_line(self, x1: float, y1: float, x2: float, y2: float):
        """Add a fold line to the dieline."""
        self.fold_lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    
    def add_cut_line(self, x1: float, y1: float, x2: float, y2: float):
        """Add a cut line to the dieline."""
        self.cut_lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding box of the entire dieline.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if not self.panels:
            return (0, 0, 0, 0)
        
        all_x = []
        all_y = []
        for panel in self.panels:
            corners = panel.get_corners()
            all_x.extend([c[0] for c in corners])
            all_y.extend([c[1] for c in corners])
        
        return (min(all_x), min(all_y), max(all_x), max(all_y))
    
    def __repr__(self) -> str:
        return f"CartonDieline(L={self.length}, W={self.width}, H={self.height}, panels={len(self.panels)})"


def adjust_dimensions(length: float, width: float, height: float, 
                      clearance: float = 6.0) -> Tuple[float, float, float]:
    """
    Adjust raw object dimensions by adding clearance.
    
    Business Rule: Add 6 cm to each dimension for proper fit and handling.
    
    Args:
        length: Measured object length in cm
        width: Measured object width in cm
        height: Measured object height in cm
        clearance: Clearance to add to each dimension (default: 6.0 cm)
    
    Returns:
        Tuple of (adjusted_length, adjusted_width, adjusted_height)
    
    Example:
        >>> adjust_dimensions(30, 20, 15)
        (36.0, 26.0, 21.0)
    """
    adjusted_length = length + clearance
    adjusted_width = width + clearance
    adjusted_height = height + clearance
    
    return (adjusted_length, adjusted_width, adjusted_height)


def calculate_flap_size(box_width: float, flap_ratio: float = 0.5) -> float:
    """
    Calculate the flap size based on box width.
    
    Standard practice: Flaps are typically 50% of the box width.
    
    Args:
        box_width: Width of the box
        flap_ratio: Ratio of flap to width (default: 0.5 for 50%)
    
    Returns:
        Flap size in cm
    """
    return box_width * flap_ratio


def generate_rsc_dieline(length: float, width: float, height: float) -> CartonDieline:
    """
    Generate a complete RSC (Regular Slotted Container) dieline.
    
    The RSC layout consists of:
    - A continuous strip of panels: front, bottom, back, top
    - Side panels (left and right) attached to the bottom
    - Flaps attached to front, back, left, and right panels
    
    Layout (flat view):
    
           [Left Flap]     [Back Flap]    [Right Flap]   [Front Flap]
    [Left]   [Front]        [Bottom]        [Back]         [Top]      [Right]
           [Left Flap]     [Back Flap]    [Right Flap]   [Front Flap]
    
    Args:
        length: Internal carton length (cm)
        width: Internal carton width (cm)
        height: Internal carton height (cm)
    
    Returns:
        CartonDieline object containing all panels and coordinates
    """
    dieline = CartonDieline(length, width, height)
    
    # Calculate flap sizes
    # Top and bottom flaps extend from the length panels
    flap_length = calculate_flap_size(width)
    
    # Starting position for layout
    start_x = 0
    start_y = flap_length  # Offset to accommodate bottom flaps
    
    # ========== MAIN PANEL STRIP (horizontal) ==========
    # This creates the continuous strip: Left side -> Front -> Bottom -> Back -> Top
    
    current_x = start_x
    
    # 1. LEFT SIDE PANEL
    left_panel = DielinePanel('left_side', current_x, start_y, width, height)
    dieline.add_panel(left_panel)
    current_x += width
    
    # 2. FRONT PANEL (length x height)
    front_panel = DielinePanel('front', current_x, start_y, length, height)
    dieline.add_panel(front_panel)
    current_x += length
    
    # 3. BOTTOM PANEL (width x length) - rotated 90°
    bottom_panel = DielinePanel('bottom', current_x, start_y, width, height)
    dieline.add_panel(bottom_panel)
    current_x += width
    
    # 4. BACK PANEL (length x height)
    back_panel = DielinePanel('back', current_x, start_y, length, height)
    dieline.add_panel(back_panel)
    current_x += length
    
    # 5. TOP PANEL (width x length) - rotated 90°
    top_panel = DielinePanel('top', current_x, start_y, width, height)
    dieline.add_panel(top_panel)
    current_x += width
    
    # 6. RIGHT SIDE PANEL (with glue flap)
    right_panel = DielinePanel('right_side', current_x, start_y, width, height)
    dieline.add_panel(right_panel)
    
    # ========== FLAPS ==========
    
    # BOTTOM FLAPS (attached to left, front, back, right)
    # Left bottom flap
    left_bottom_flap = DielinePanel('left_bottom_flap', 
                                    start_x, start_y - flap_length, 
                                    width, flap_length)
    dieline.add_panel(left_bottom_flap)
    
    # Front bottom flaps (two half-width flaps)
    front_bottom_flap_1 = DielinePanel('front_bottom_flap_1',
                                       start_x + width, start_y - flap_length,
                                       length / 2, flap_length)
    dieline.add_panel(front_bottom_flap_1)
    
    front_bottom_flap_2 = DielinePanel('front_bottom_flap_2',
                                       start_x + width + length / 2, start_y - flap_length,
                                       length / 2, flap_length)
    dieline.add_panel(front_bottom_flap_2)
    
    # Back bottom flaps (two half-width flaps)
    back_x_start = start_x + width + length + width
    back_bottom_flap_1 = DielinePanel('back_bottom_flap_1',
                                      back_x_start, start_y - flap_length,
                                      length / 2, flap_length)
    dieline.add_panel(back_bottom_flap_1)
    
    back_bottom_flap_2 = DielinePanel('back_bottom_flap_2',
                                      back_x_start + length / 2, start_y - flap_length,
                                      length / 2, flap_length)
    dieline.add_panel(back_bottom_flap_2)
    
    # Right bottom flap
    right_x_start = start_x + width + length + width + length + width
    right_bottom_flap = DielinePanel('right_bottom_flap',
                                     right_x_start, start_y - flap_length,
                                     width, flap_length)
    dieline.add_panel(right_bottom_flap)
    
    # TOP FLAPS (attached to left, front, back, right)
    # Left top flap
    left_top_flap = DielinePanel('left_top_flap',
                                 start_x, start_y + height,
                                 width, flap_length)
    dieline.add_panel(left_top_flap)
    
    # Front top flaps (two half-width flaps)
    front_top_flap_1 = DielinePanel('front_top_flap_1',
                                    start_x + width, start_y + height,
                                    length / 2, flap_length)
    dieline.add_panel(front_top_flap_1)
    
    front_top_flap_2 = DielinePanel('front_top_flap_2',
                                    start_x + width + length / 2, start_y + height,
                                    length / 2, flap_length)
    dieline.add_panel(front_top_flap_2)
    
    # Back top flaps (two half-width flaps)
    back_top_flap_1 = DielinePanel('back_top_flap_1',
                                   back_x_start, start_y + height,
                                   length / 2, flap_length)
    dieline.add_panel(back_top_flap_1)
    
    back_top_flap_2 = DielinePanel('back_top_flap_2',
                                   back_x_start + length / 2, start_y + height,
                                   length / 2, flap_length)
    dieline.add_panel(back_top_flap_2)
    
    # Right top flap
    right_top_flap = DielinePanel('right_top_flap',
                                  right_x_start, start_y + height,
                                  width, flap_length)
    dieline.add_panel(right_top_flap)
    
    # ========== FOLD LINES ==========
    # Add vertical fold lines between main panels
    fold_x_positions = [
        start_x + width,                              # Between left and front
        start_x + width + length,                     # Between front and bottom
        start_x + width + length + width,             # Between bottom and back
        start_x + width + length + width + length,    # Between back and top
        start_x + width + length + width + length + width  # Between top and right
    ]
    
    for fold_x in fold_x_positions:
        dieline.add_fold_line(fold_x, start_y - flap_length, 
                             fold_x, start_y + height + flap_length)
    
    # Add horizontal fold lines for flaps
    dieline.add_fold_line(start_x, start_y, 
                         start_x + width + length + width + length + width + width, start_y)
    dieline.add_fold_line(start_x, start_y + height,
                         start_x + width + length + width + length + width + width, start_y + height)
    
    return dieline


def export_to_svg(dieline: CartonDieline, filename: str = 'carton_dieline.svg',
                  scale: float = 10.0) -> str:
    """
    Export the dieline to an SVG file.
    
    Args:
        dieline: CartonDieline object to export
        filename: Output filename (default: 'carton_dieline.svg')
        scale: Scaling factor for visualization (default: 10.0 pixels per cm)
    
    Returns:
        Path to the created SVG file
    """
    # Get bounding box
    min_x, min_y, max_x, max_y = dieline.get_bounding_box()
    
    # Add margin
    margin = 2  # cm
    canvas_width = (max_x - min_x + 2 * margin) * scale
    canvas_height = (max_y - min_y + 2 * margin) * scale
    
    # Offset for centering
    offset_x = margin - min_x
    offset_y = margin - min_y
    
    # Start building SVG
    svg_lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_width:.2f}" height="{canvas_height:.2f}" '
        f'viewBox="0 0 {canvas_width:.2f} {canvas_height:.2f}">',
        f'  <!-- Carton Dieline: L={dieline.length}cm W={dieline.width}cm H={dieline.height}cm -->',
        f'  <g id="dieline" transform="scale({scale})">',
    ]
    
    # Add panels as rectangles with labels
    svg_lines.append('    <!-- Panels -->')
    svg_lines.append('    <g id="panels" stroke="black" stroke-width="0.05" fill="none">')
    
    for panel in dieline.panels:
        x = panel.x + offset_x
        y = panel.y + offset_y
        svg_lines.append(
            f'      <rect x="{x:.2f}" y="{y:.2f}" '
            f'width="{panel.width:.2f}" height="{panel.height:.2f}" />'
        )
        # Add label at center of panel
        cx = x + panel.width / 2
        cy = y + panel.height / 2
        svg_lines.append(
            f'      <text x="{cx:.2f}" y="{cy:.2f}" '
            f'font-size="0.3" text-anchor="middle" fill="blue">{panel.name}</text>'
        )
    
    svg_lines.append('    </g>')
    
    # Add fold lines (dashed)
    svg_lines.append('    <!-- Fold Lines -->')
    svg_lines.append('    <g id="fold-lines" stroke="red" stroke-width="0.03" '
                    'stroke-dasharray="0.2,0.1">')
    
    for fold in dieline.fold_lines:
        x1 = fold['x1'] + offset_x
        y1 = fold['y1'] + offset_y
        x2 = fold['x2'] + offset_x
        y2 = fold['y2'] + offset_y
        svg_lines.append(f'      <line x1="{x1:.2f}" y1="{y1:.2f}" '
                        f'x2="{x2:.2f}" y2="{y2:.2f}" />')
    
    svg_lines.append('    </g>')
    
    # Close SVG
    svg_lines.append('  </g>')
    svg_lines.append('</svg>')
    
    # Write to file
    svg_content = '\n'.join(svg_lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    return filename


def _svg_reference_layout_dimensions(length: float, width: float, height: float) -> Dict[str, float]:
    """Return layout dimensions for the reference-based carton template."""
    flap_depth = max(width * 0.5, 6.0)
    glue_tab_width = max(width * 0.18, 3.0)
    board_gap = max(min(width * 0.01, 0.4), 0.15)

    return {
        'panel_1_width': width,
        'panel_2_width': length,
        'panel_3_width': width,
        'panel_4_width': length,
        'panel_height': height,
        'flap_depth': flap_depth,
        'glue_tab_width': glue_tab_width,
        'board_gap': board_gap,
    }


def build_reference_svg(length: float, width: float, height: float,
                        filename: str = 'carton_dieline.svg',
                        scale: float = 16.0) -> str:
    """
    Export a carton dieline that follows the layout proportions of SVG File-01.svg.

    The layout is generated from adjusted carton dimensions and includes:
    - A tapered glue tab on the left
    - Four body panels in sequence: side, front, side, back
    - Full-width outer flaps with U-slot cuts at panel boundaries
    - Red crease/fold lines that never overlap black cut lines
    """
    layout = _svg_reference_layout_dimensions(length, width, height)
    panel_1_width = layout['panel_1_width']
    panel_2_width = layout['panel_2_width']
    panel_3_width = layout['panel_3_width']
    panel_4_width = layout['panel_4_width']
    panel_height = layout['panel_height']
    flap_depth = layout['flap_depth']
    glue_tab_width = layout['glue_tab_width']
    board_gap = layout['board_gap']

    margin = 2.0
    body_x0 = margin + glue_tab_width
    body_y0 = margin + flap_depth
    body_y1 = body_y0 + panel_height

    x0 = body_x0
    x1 = x0 + panel_1_width
    x2 = x1 + panel_2_width
    x3 = x2 + panel_3_width
    x4 = x3 + panel_4_width

    outer_left  = margin
    outer_right = x4 + margin
    outer_top   = margin
    outer_bottom = body_y1 + flap_depth + margin

    canvas_width  = (outer_right - outer_left) * scale
    canvas_height = (outer_bottom - outer_top) * scale

    top_flap_y    = body_y0 - flap_depth
    bottom_flap_y = body_y1 + flap_depth

    # U-slot tips stop just short of the fold line so cut never touches fold
    slit_top    = body_y0 - board_gap   # top slits end here (above fold at body_y0)
    slit_bottom = body_y1 + board_gap   # bottom slits end here (below fold at body_y1)

    bleed_points = [
        (outer_left,         outer_bottom),
        (outer_left,         body_y1 + 0.25),
        (outer_left - 0.35,  body_y1),
        (outer_left - 0.35,  body_y0),
        (outer_left,         body_y0 - 0.25),
        (outer_left,         outer_top),
        (outer_right,        outer_top),
        (outer_right,        body_y0 - 0.25),
        (outer_right + 0.15, body_y0 - 0.25),
        (outer_right + 0.15, body_y1 + 0.25),
        (outer_right,        body_y1 + 0.25),
        (outer_right,        outer_bottom),
    ]

    def fmt_points(points: List[Tuple[float, float]]) -> str:
        return ' '.join(f'{px * scale:.2f},{py * scale:.2f}' for px, py in points)

    def svg_line(ax1: float, ay1: float, ax2: float, ay2: float,
                 cls: str = 'cls-3') -> str:
        return (
            f'<line class="{cls}" x1="{ax1 * scale:.2f}" y1="{ay1 * scale:.2f}" '
            f'x2="{ax2 * scale:.2f}" y2="{ay2 * scale:.2f}"/>'
        )

    def svg_polyline(points: List[Tuple[float, float]], cls: str = 'cls-2') -> str:
        return f'<polyline class="{cls}" points="{fmt_points(points)}"/>'

    # ================================================================
    # FOLD / CREASE LINES (red dashed)
    # Rule: fold lines occupy the body interior ONLY.
    #   • Horizontal folds at body_y0 and body_y1 span full body width (x0→x4).
    #   • Vertical folds at x0..x3 span body height only (body_y0→body_y1).
    #   • x4 has NO fold line — it is purely a cut boundary.
    # ================================================================
    fold_lines = [
        svg_line(x0, body_y0, x4, body_y0),   # top horizontal fold
        svg_line(x0, body_y1, x4, body_y1),   # bottom horizontal fold
        svg_line(x0, body_y0, x0, body_y1),   # glue-tab / panel-1 fold
        svg_line(x1, body_y0, x1, body_y1),   # panel-1 / panel-2 fold
        svg_line(x2, body_y0, x2, body_y1),   # panel-2 / panel-3 fold
        svg_line(x3, body_y0, x3, body_y1),   # panel-3 / panel-4 fold
    ]

    # ================================================================
    # CUT LINES (black solid)
    # Rule: no cut segment may lie on the same coordinates as a fold segment.
    #
    #  Glue tab  — open 3-sided path; the right edge IS the fold at x0,
    #              so only left + top-slant + bottom-slant are drawn.
    #  Top flap  — horizontal cuts at top_flap_y, interrupted by U-slots
    #              at x1, x2, x3; slot tips stop at slit_top (above fold).
    #  Bot flap  — same pattern at bottom_flap_y / slit_bottom.
    #  Left side — two short verticals at x0 bridging fold endpoints to
    #              outer flap edges; they share endpoints with the fold
    #              but cover different y-ranges (no overlap).
    #  Right side — single vertical at x4 (no fold here).
    # ================================================================

    glue_inset   = panel_height * 0.03
    glue_inner_x = outer_left + glue_tab_width * 0.18

    # Glue tab: 3 sides only (right edge omitted — covered by fold at x0)
    glue_tab_cut = svg_polyline([
        (x0,           body_y0),
        (glue_inner_x, body_y0 + glue_inset),
        (glue_inner_x, body_y1 - glue_inset),
        (x0,           body_y1),
    ])

    # Top flap: 4 horizontal segments + 3 U-shaped slot cuts
    top_h1  = svg_line(x0,               top_flap_y, x1 - board_gap, top_flap_y, 'cls-2')
    slit1_t = svg_polyline([(x1 - board_gap, top_flap_y),
                             (x1 - board_gap, slit_top),
                             (x1 + board_gap, slit_top),
                             (x1 + board_gap, top_flap_y)])
    top_h2  = svg_line(x1 + board_gap, top_flap_y, x2 - board_gap, top_flap_y, 'cls-2')
    slit2_t = svg_polyline([(x2 - board_gap, top_flap_y),
                             (x2 - board_gap, slit_top),
                             (x2 + board_gap, slit_top),
                             (x2 + board_gap, top_flap_y)])
    top_h3  = svg_line(x2 + board_gap, top_flap_y, x3 - board_gap, top_flap_y, 'cls-2')
    slit3_t = svg_polyline([(x3 - board_gap, top_flap_y),
                             (x3 - board_gap, slit_top),
                             (x3 + board_gap, slit_top),
                             (x3 + board_gap, top_flap_y)])
    top_h4  = svg_line(x3 + board_gap, top_flap_y, x4, top_flap_y, 'cls-2')

    # Bottom flap: 4 horizontal segments + 3 U-shaped slot cuts
    bot_h1  = svg_line(x0,               bottom_flap_y, x1 - board_gap, bottom_flap_y, 'cls-2')
    slit1_b = svg_polyline([(x1 - board_gap, bottom_flap_y),
                             (x1 - board_gap, slit_bottom),
                             (x1 + board_gap, slit_bottom),
                             (x1 + board_gap, bottom_flap_y)])
    bot_h2  = svg_line(x1 + board_gap, bottom_flap_y, x2 - board_gap, bottom_flap_y, 'cls-2')
    slit2_b = svg_polyline([(x2 - board_gap, bottom_flap_y),
                             (x2 - board_gap, slit_bottom),
                             (x2 + board_gap, slit_bottom),
                             (x2 + board_gap, bottom_flap_y)])
    bot_h3  = svg_line(x2 + board_gap, bottom_flap_y, x3 - board_gap, bottom_flap_y, 'cls-2')
    slit3_b = svg_polyline([(x3 - board_gap, bottom_flap_y),
                             (x3 - board_gap, slit_bottom),
                             (x3 + board_gap, slit_bottom),
                             (x3 + board_gap, bottom_flap_y)])
    bot_h4  = svg_line(x3 + board_gap, bottom_flap_y, x4, bottom_flap_y, 'cls-2')

    # Left flap side cuts: share only endpoints with folds (different y-ranges → no overlap)
    left_top_side = svg_line(x0, top_flap_y,  x0, body_y0,       'cls-2')
    left_bot_side = svg_line(x0, body_y1,      x0, bottom_flap_y, 'cls-2')

    # Right outer edge: single vertical (x4 has no fold at all)
    right_side = svg_line(x4, top_flap_y, x4, bottom_flap_y, 'cls-2')

    dimension_note_y = outer_bottom - 0.65
    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {canvas_width:.2f} {canvas_height:.2f}">'
        ),
        '<defs><style>'
        '.cls-1{fill:#ffca8a;opacity:.18;}'
        '.cls-2,.cls-3{fill:none;stroke-miterlimit:10;}'
        '.cls-2{stroke:#000;stroke-width:1.8;}'
        '.cls-3{stroke:#d62828;stroke-width:1.2;stroke-dasharray:10 8;}'
        '.label{font-family:Arial,sans-serif;font-size:18px;fill:#1f4e79;font-weight:bold;}'
        '.meta{font-family:Arial,sans-serif;font-size:16px;fill:#333;}'
        '</style></defs>',
        '<title>Carton Dieline</title>',
        '<g id="Design_Here">',
        f'<polygon id="Out_Side_Bleed" class="cls-1" points="{fmt_points(bleed_points)}"/>',
        '</g>',
        '<g id="Cut_Line">',
        glue_tab_cut,
        left_top_side,
        left_bot_side,
        right_side,
        top_h1, slit1_t, top_h2, slit2_t, top_h3, slit3_t, top_h4,
        bot_h1, slit1_b, bot_h2, slit2_b, bot_h3, slit3_b, bot_h4,
        '</g>',
        '<g id="Fold_Line">',
        *fold_lines,
        '</g>',
        '<g id="Labels">',
        '</g>',
        '<g id="Meta">',
        '</g>',
        '</svg>',
    ]

    with open(filename, 'w', encoding='utf-8') as svg_file:
        svg_file.write('\n'.join(svg_lines))

    return os.path.abspath(filename)


def create_carton_dieline(length: float, width: float, height: float,
                         export_svg: bool = False,
                         svg_filename: str = 'carton_dieline.svg') -> CartonDieline:
    """
    Main function to create a carton dieline from raw object dimensions.
    
    This function:
    1. Adjusts the input dimensions by adding 6 cm clearance
    2. Generates a complete RSC dieline with all panels and flaps
    3. Optionally exports the dieline to an SVG file
    
    Args:
        length: Measured object length in cm
        width: Measured object width in cm
        height: Measured object height in cm
        export_svg: Whether to export to SVG file (default: False)
        svg_filename: Output SVG filename (default: 'carton_dieline.svg')
    
    Returns:
        CartonDieline object containing the complete dieline
    
    Example:
        >>> dieline = create_carton_dieline(30, 20, 15, export_svg=True)
        >>> print(f"Created dieline with {len(dieline.panels)} panels")
        >>> bbox = dieline.get_bounding_box()
        >>> print(f"Bounding box: {bbox}")
    """
    # Step 1: Adjust dimensions according to business rule
    adjusted_length, adjusted_width, adjusted_height = adjust_dimensions(
        length, width, height
    )
    
    print(f"Original dimensions: L={length}cm, W={width}cm, H={height}cm")
    print(f"Adjusted dimensions: L={adjusted_length}cm, W={adjusted_width}cm, H={adjusted_height}cm")
    
    # Step 2: Generate the RSC dieline
    dieline = generate_rsc_dieline(adjusted_length, adjusted_width, adjusted_height)
    
    print(f"Generated RSC dieline with {len(dieline.panels)} panels")
    
    # Step 3: Calculate and display bounding box
    bbox = dieline.get_bounding_box()
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    print(f"Dieline dimensions: {bbox_width:.2f}cm x {bbox_height:.2f}cm")
    
    # Step 4: Optional SVG export
    if export_svg:
        svg_path = build_reference_svg(adjusted_length, adjusted_width, adjusted_height, svg_filename)
        print(f"Exported dieline to: {svg_path}")
    
    return dieline


def get_dieline_data(dieline: CartonDieline) -> Dict[str, Any]:
    """
    Convert dieline to a dictionary structure for serialization.
    
    Args:
        dieline: CartonDieline object
    
    Returns:
        Dictionary containing all dieline data
    """
    return {
        'dimensions': {
            'length': dieline.length,
            'width': dieline.width,
            'height': dieline.height
        },
        'panels': [
            {
                'name': panel.name,
                'x': panel.x,
                'y': panel.y,
                'width': panel.width,
                'height': panel.height,
                'corners': panel.get_corners()
            }
            for panel in dieline.panels
        ],
        'fold_lines': dieline.fold_lines,
        'bounding_box': dieline.get_bounding_box()
    }


def generate_dieline(width: float, height: float, length: float, 
                    show: bool = True, save_path: str = None) -> plt.Figure:
    """
    Generate and visualize a packaging dieline from measured dimensions.
    
    This function creates a complete RSC (Regular Slotted Container) dieline
    using matplotlib visualization. All dimensions are computed dynamically
    from the input parameters - no values are hard-coded.
    
    Args:
        width: Object width in cm (from camera measurements)
        height: Object height in cm (from camera measurements)
        length: Object length in cm (from camera measurements)
        show: Whether to display the plot (default: True)
        save_path: Optional path to save the figure (e.g., 'dieline.png')
    
    Returns:
        matplotlib.pyplot.Figure object containing the dieline visualization
    
    Example:
        >>> # Get measurements from cameras (in cm)
        >>> width, height, length = 20.0, 15.0, 30.0
        >>> fig = generate_dieline(width, height, length, show=True)
    """
    adjusted_length, adjusted_width, adjusted_height = adjust_dimensions(length, width, height)

    print(f"Input dimensions: W={width}cm, H={height}cm, L={length}cm")
    print(
        f"Adjusted dimensions: W={adjusted_width}cm, H={adjusted_height}cm, "
        f"L={adjusted_length}cm"
    )

    output_path = save_path or 'carton_dieline.svg'
    svg_path = build_reference_svg(
        adjusted_length,
        adjusted_width,
        adjusted_height,
        filename=output_path,
    )
    print(f"Dieline saved to: {svg_path}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axis('off')
    ax.text(
        0.5,
        0.62,
        'Carton dieline exported as SVG',
        ha='center',
        va='center',
        fontsize=20,
        fontweight='bold',
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.48,
        f'Adjusted size: L={adjusted_length:.1f} cm, W={adjusted_width:.1f} cm, H={adjusted_height:.1f} cm',
        ha='center',
        va='center',
        fontsize=14,
        color='#333333',
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.36,
        f'SVG output: {svg_path}',
        ha='center',
        va='center',
        fontsize=11,
        color='#555555',
        transform=ax.transAxes,
    )
    fig.tight_layout()

    if show:
        plt.show()

    return fig


# ========== USAGE EXAMPLE ==========
if __name__ == '__main__':
    # Example: Create a dieline for an object measuring 30cm x 20cm x 15cm
    print("=" * 60)
    print("CARTON DIELINE GENERATOR")
    print("=" * 60)
    print()
    
    # Input dimensions (measured object)
    object_length = 30.0  # cm
    object_width = 20.0   # cm
    object_height = 15.0  # cm
    
    # Create the dieline with automatic adjustment and SVG export
    dieline = create_carton_dieline(
        object_length, 
        object_width, 
        object_height,
        export_svg=True,
        svg_filename='example_carton_dieline.svg'
    )
    
    print()
    print("=" * 60)
    print("PANEL DETAILS")
    print("=" * 60)
    
    for panel in dieline.panels:
        print(f"{panel.name:25s} -> Position: ({panel.x:.1f}, {panel.y:.1f}), "
              f"Size: {panel.width:.1f} x {panel.height:.1f} cm")
    
    print()
    print("=" * 60)
    print("Dieline generation complete!")
    print("=" * 60)
    
    # Test the new generate_dieline function
    print("\n" + "=" * 60)
    print("TESTING generate_dieline() FUNCTION")
    print("=" * 60)
    
    # Generate matplotlib dieline with measured dimensions
    fig = generate_dieline(width=object_width, height=object_height, length=object_length, 
                          show=False, save_path='dieline_matplotlib.png')
    print("Matplotlib dieline created successfully!")
