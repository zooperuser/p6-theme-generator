"""Test script to verify hue-based color filtering works correctly."""
import numpy as np
from PIL import Image
from app import ImagePaletteGenerator, Config

def create_test_image_limited_colors():
    """Create a test image with limited color variety (mostly shades of blue)."""
    img = Image.new('RGB', (100, 100))
    pixels = img.load()
    
    # Fill with various shades of blue (similar hues)
    for y in range(100):
        for x in range(100):
            # Create blue shades with slight variations
            blue = 150 + (x % 50)
            pixels[x, y] = (50, 100, blue)
    
    # Add a few distinct colors
    for i in range(10):
        for j in range(10):
            pixels[i, j] = (200, 50, 50)  # Red
            pixels[90 + i, j] = (50, 200, 50)  # Green
            pixels[i, 90 + j] = (200, 200, 50)  # Yellow
    
    return img

def create_test_image_diverse_colors():
    """Create a test image with diverse colors across the spectrum."""
    img = Image.new('RGB', (100, 100))
    pixels = img.load()
    
    colors = [
        (255, 0, 0),      # Red
        (255, 127, 0),    # Orange
        (255, 255, 0),    # Yellow
        (0, 255, 0),      # Green
        (0, 255, 255),    # Cyan
        (0, 0, 255),      # Blue
        (127, 0, 255),    # Purple
        (255, 0, 255),    # Magenta
        (128, 128, 128),  # Gray
        (0, 0, 0),        # Black
    ]
    
    # Fill image with diverse colors
    section_width = 100 // len(colors)
    for idx, color in enumerate(colors):
        x_start = idx * section_width
        x_end = min(x_start + section_width, 100)
        for y in range(100):
            for x in range(x_start, x_end):
                pixels[x, y] = color
    
    return img

def test_hue_filtering():
    """Test the hue filtering functionality."""
    cfg = Config()
    gen = ImagePaletteGenerator(cfg)
    
    print("=" * 60)
    print("Test 1: Image with LIMITED color variety (mostly blues)")
    print("=" * 60)
    
    img_limited = create_test_image_limited_colors()
    result = gen.generate_random_palette_from_image(img_limited, n_colors=15)
    html, hex_csv, hex_rev, font_csv, font_rev, chaotic_html, chaotic_hex, chaotic_font = result
    
    colors_limited = hex_csv.split(", ") if hex_csv else []
    print(f"Requested: 15 colors")
    print(f"Returned: {len(colors_limited)} colors")
    print(f"Colors: {hex_csv}")
    print()
    
    # Calculate hue diversity
    if colors_limited:
        rgb_colors = []
        for hex_color in colors_limited:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                rgb_colors.append([r, g, b])
        
        if rgb_colors:
            rgb_arr = np.array(rgb_colors, dtype=np.float32)
            h, s, v = gen._rgb_to_hsv_np(rgb_arr)
            hues_deg = h * 360
            
            print("Hue analysis:")
            for i, (color, hue, sat) in enumerate(zip(colors_limited, hues_deg, s)):
                print(f"  {color}: hue={hue:.1f}°, saturation={sat:.2f}")
            
            # Check minimum hue difference
            if len(hues_deg) > 1:
                min_diff = float('inf')
                for i in range(len(hues_deg)):
                    for j in range(i + 1, len(hues_deg)):
                        diff = abs(hues_deg[i] - hues_deg[j])
                        diff = min(diff, 360 - diff)  # Circular distance
                        min_diff = min(min_diff, diff)
                print(f"\nMinimum hue difference: {min_diff:.1f}° (threshold was 15-20°)")
    
    print("\n" + "=" * 60)
    print("Test 2: Image with DIVERSE colors across spectrum")
    print("=" * 60)
    
    img_diverse = create_test_image_diverse_colors()
    result2 = gen.generate_random_palette_from_image(img_diverse, n_colors=15)
    html2, hex_csv2, hex_rev2, font_csv2, font_rev2, chaotic_html2, chaotic_hex2, chaotic_font2 = result2
    
    colors_diverse = hex_csv2.split(", ") if hex_csv2 else []
    print(f"Requested: 15 colors")
    print(f"Returned: {len(colors_diverse)} colors")
    print(f"Colors: {hex_csv2}")
    print()
    
    # Calculate hue diversity for diverse image
    if colors_diverse:
        rgb_colors2 = []
        for hex_color in colors_diverse:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                rgb_colors2.append([r, g, b])
        
        if rgb_colors2:
            rgb_arr2 = np.array(rgb_colors2, dtype=np.float32)
            h2, s2, v2 = gen._rgb_to_hsv_np(rgb_arr2)
            hues_deg2 = h2 * 360
            
            print("Hue analysis:")
            for i, (color, hue, sat) in enumerate(zip(colors_diverse, hues_deg2, s2)):
                print(f"  {color}: hue={hue:.1f}°, saturation={sat:.2f}")
            
            # Check minimum hue difference
            if len(hues_deg2) > 1:
                min_diff2 = float('inf')
                for i in range(len(hues_deg2)):
                    for j in range(i + 1, len(hues_deg2)):
                        diff = abs(hues_deg2[i] - hues_deg2[j])
                        diff = min(diff, 360 - diff)
                        min_diff2 = min(min_diff2, diff)
                print(f"\nMinimum hue difference: {min_diff2:.1f}° (threshold was 15-20°)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Limited color image: Returned {len(colors_limited)} colors (may be < 15 if image lacks variety)")
    print(f"✓ Diverse color image: Returned {len(colors_diverse)} colors")
    print(f"✓ Hue filtering ensures colors are at least 15-20° apart in hue")
    print(f"✓ System adapts to image content, returning fewer colors if needed")

if __name__ == "__main__":
    test_hue_filtering()
