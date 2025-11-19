"""Enhanced test to verify smart arrangement prevents adjacent similar hues."""
from PIL import Image
from app import ImagePaletteGenerator, Config
import numpy as np

def test_arrangement():
    """Test that similar hues are properly separated."""
    cfg = Config()
    gen = ImagePaletteGenerator(cfg)
    
    # Create gradient image
    img = Image.new('RGB', (400, 400))
    pixels = img.load()
    for y in range(400):
        color_value = int((y / 400) * 255)
        color = (0, color_value, 255 - int(color_value * 0.5))
        for x in range(400):
            pixels[x, y] = color
    
    print("=" * 70)
    print("Testing Smart Arrangement (prevents adjacent similar hues)")
    print("=" * 70)
    
    result = gen.generate_random_palette_from_image(img, n_colors=15)
    html, hex_csv, *_ = result
    
    colors = hex_csv.split(", ") if hex_csv else []
    print(f"\nGenerated {len(colors)} colors (requested: 15)")
    print("\nColor Order with Hue Analysis:")
    print("-" * 70)
    
    # Parse colors and get hues
    rgb_colors = []
    for hex_color in colors:
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
        
        # Display colors in order with hue differences
        for i, (color, hue, sat) in enumerate(zip(colors, hues_deg, s)):
            # Calculate distance to nearest similar hue
            if sat >= 0.1:  # Only check colored items
                min_dist = float('inf')
                for j, (other_hue, other_sat) in enumerate(zip(hues_deg, s)):
                    if i != j and other_sat >= 0.1:
                        # Calculate position distance
                        pos_dist = min(abs(i - j), len(colors) - abs(i - j))
                        # Calculate hue difference
                        hue_diff = abs(hue - other_hue)
                        hue_diff = min(hue_diff, 360 - hue_diff)
                        
                        if hue_diff < 30:  # Consider "similar" if within 30°
                            min_dist = min(min_dist, pos_dist)
                
                sep_status = "✓" if min_dist >= 2 else "✗"
                print(f"{i+1:2d}. {color}  │  H:{hue:6.1f}°  S:{sat:4.2f}  │  Nearest similar: {min_dist:.0f} positions away {sep_status}")
            else:
                print(f"{i+1:2d}. {color}  │  H:{hue:6.1f}°  S:{sat:4.2f}  │  Gray (no separation check)")
        
        # Check for violations (similar hues within 2 positions)
        violations = []
        for i in range(len(colors)):
            if s[i] < 0.1:  # Skip grays
                continue
            for j in range(len(colors)):
                if i >= j or s[j] < 0.1:
                    continue
                
                pos_dist = min(abs(i - j), len(colors) - abs(i - j))
                hue_diff = abs(hues_deg[i] - hues_deg[j])
                hue_diff = min(hue_diff, 360 - hue_diff)
                
                if hue_diff < 30 and pos_dist < 3:
                    violations.append((i+1, j+1, hue_diff, pos_dist))
        
        print("\n" + "=" * 70)
        if violations:
            print("⚠ VIOLATIONS FOUND (similar hues too close):")
            for i, j, hue_diff, pos_dist in violations:
                print(f"  Positions {i} and {j}: {hue_diff:.1f}° apart, only {pos_dist} positions between")
        else:
            print("✓ SUCCESS: All similar hues are at least 3 positions apart!")
        print("=" * 70)

if __name__ == "__main__":
    test_arrangement()
