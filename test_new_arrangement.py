"""Test the improved arrangement algorithm with a real palette generation."""

import sys
import numpy as np
from app import ImagePaletteGenerator, Config

def analyze_arrangement(rgb_colors, title="Palette"):
    """Analyze and display arrangement of colors."""
    gen = ImagePaletteGenerator(Config())
    rgb_arr = np.array(rgb_colors, dtype=np.float32)
    h, s, v = gen._rgb_to_hsv_np(rgb_arr)
    hues = h * 360
    
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # Display colors with their properties
    for i, (rgb, hue, sat, val) in enumerate(zip(rgb_colors, hues, s, v)):
        color_type = "Gray" if sat < 0.1 else "Color"
        hex_code = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        print(f"{i+1:2d}. {hex_code}  │  H: {hue:6.1f}°  S:{sat:.2f}  V:{val:.2f}  │  {color_type}")
    
    # Check for violations
    print(f"\n{'='*80}")
    print("SEPARATION ANALYSIS")
    print(f"{'='*80}")
    
    violations = []
    borderline = []
    n = len(rgb_colors)
    
    for i in range(n):
        if s[i] < 0.1:  # Skip grays
            continue
        
        for j in range(i+1, n):
            if s[j] < 0.1:  # Skip grays
                continue
            
            # Calculate hue difference
            hue_diff = abs(hues[i] - hues[j])
            hue_diff = min(hue_diff, 360 - hue_diff)
            
            # Only check if hues are similar (within 30°)
            if hue_diff < 30:
                # Calculate circular position distance
                pos_diff = abs(j - i)
                circular_dist = min(pos_diff, n - pos_diff)
                
                if circular_dist < 3:
                    violations.append((i, j, hue_diff, circular_dist))
                elif circular_dist == 3:
                    borderline.append((i, j, hue_diff, circular_dist))
    
    if violations:
        print("\n❌ VIOLATIONS FOUND (similar hues < 3 positions apart):")
        print("-" * 80)
        for i, j, hue_diff, dist in violations:
            hex_i = f"#{rgb_colors[i][0]:02X}{rgb_colors[i][1]:02X}{rgb_colors[i][2]:02X}"
            hex_j = f"#{rgb_colors[j][0]:02X}{rgb_colors[j][1]:02X}{rgb_colors[j][2]:02X}"
            print(f"  Position {i+1:2d} ({hex_i}) and {j+1:2d} ({hex_j})")
            print(f"    Hue difference: {hue_diff:.1f}° | Distance: {dist} positions ❌")
    else:
        print("\n✓ NO VIOLATIONS FOUND!")
    
    if borderline:
        print("\n⚠ BORDERLINE CASES (exactly 3 positions apart):")
        print("-" * 80)
        for i, j, hue_diff, dist in borderline:
            hex_i = f"#{rgb_colors[i][0]:02X}{rgb_colors[i][1]:02X}{rgb_colors[i][2]:02X}"
            hex_j = f"#{rgb_colors[j][0]:02X}{rgb_colors[j][1]:02X}{rgb_colors[j][2]:02X}"
            print(f"  Position {i+1:2d} ({hex_i}) and {j+1:2d} ({hex_j})")
            print(f"    Hue difference: {hue_diff:.1f}° | Distance: {dist} positions ⚠")
    
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"Total colors: {n}")
    print(f"Violations: {len(violations)}")
    print(f"Borderline cases: {len(borderline)}")
    
    # Calculate minimum hue difference
    color_hues = [hues[i] for i in range(n) if s[i] >= 0.1]
    if len(color_hues) >= 2:
        min_hue_diff = float('inf')
        for i, h1 in enumerate(color_hues):
            for h2 in color_hues[i+1:]:
                diff = abs(h1 - h2)
                diff = min(diff, 360 - diff)
                min_hue_diff = min(min_hue_diff, diff)
        print(f"Minimum hue difference: {min_hue_diff:.1f}°")


def main():
    """Test the arrangement algorithm."""
    
    # Test case 1: Colors that previously caused violations
    # Simulating a palette with multiple cyan/blue colors
    test_colors = [
        (16, 142, 82),      # Green - 151.4°
        (245, 191, 35),     # Yellow - 44.6°
        (56, 116, 178),     # Blue - 210.5°
        (248, 248, 238),    # Gray
        (229, 156, 59),     # Orange - 34.2°
        (49, 174, 205),     # Cyan - 191.9°
        (147, 185, 207),    # Cyan-blue - 202.0°
        (183, 34, 56),      # Red - 351.1°
        (254, 251, 234),    # Gray
        (253, 254, 254),    # Gray
        (142, 89, 49),      # Brown - 25.8°
        (214, 120, 146),    # Pink - 343.4°
        (130, 32, 117),     # Purple - 308.0°
        (15, 165, 149),     # Teal - 173.6°
        (125, 167, 29),     # Green - 78.3°
    ]
    
    gen = ImagePaletteGenerator(Config())
    
    print("Testing with problematic color set...")
    arranged = gen._arrange_colors_to_avoid_similar_hues(test_colors, min_separation=3)
    analyze_arrangement(arranged, "ARRANGED PALETTE (New Algorithm)")
    
    print("\n" + "="*80)
    print("COMPARISON: Original Order")
    print("="*80)
    analyze_arrangement(test_colors, "ORIGINAL PALETTE (Before Arrangement)")


if __name__ == '__main__':
    main()
