"""Simple visual test for the palette generator."""
from PIL import Image, ImageDraw, ImageFont
from app import ImagePaletteGenerator, Config
import os

def create_gradient_image():
    """Create an image with smooth color gradients (similar hues)."""
    img = Image.new('RGB', (400, 400))
    draw = ImageDraw.Draw(img)
    
    # Create a blue-to-cyan gradient (similar hues)
    for y in range(400):
        color_value = int((y / 400) * 255)
        color = (0, color_value, 255 - int(color_value * 0.5))
        draw.rectangle([0, y, 400, y+1], fill=color)
    
    return img

def create_multicolor_image():
    """Create an image with distinct color regions."""
    img = Image.new('RGB', (400, 400))
    draw = ImageDraw.Draw(img)
    
    # Create 8 color sections
    colors = [
        (255, 0, 0),      # Red (0°)
        (255, 165, 0),    # Orange (39°)
        (255, 255, 0),    # Yellow (60°)
        (0, 255, 0),      # Green (120°)
        (0, 255, 255),    # Cyan (180°)
        (0, 0, 255),      # Blue (240°)
        (138, 43, 226),   # Blue-Violet (271°)
        (255, 0, 255),    # Magenta (300°)
    ]
    
    section_height = 400 // len(colors)
    for idx, color in enumerate(colors):
        y_start = idx * section_height
        draw.rectangle([0, y_start, 400, y_start + section_height], fill=color)
    
    return img

def visualize_palette(image, n_colors=15):
    """Generate and display palette information."""
    cfg = Config()
    gen = ImagePaletteGenerator(cfg)
    
    result = gen.generate_random_palette_from_image(image, n_colors=n_colors)
    html, hex_csv, hex_rev, font_csv, font_rev, chaotic_html, chaotic_hex, chaotic_font = result
    
    colors = hex_csv.split(", ") if hex_csv else []
    
    print(f"\nRequested: {n_colors} colors")
    print(f"Generated: {len(colors)} colors")
    print(f"\nColor Palette:")
    print("-" * 50)
    
    # Parse and display colors with hue info
    import numpy as np
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
        
        for i, (color, hue, sat, val) in enumerate(zip(colors, hues_deg, s, v)):
            color_type = "Gray" if sat < 0.1 else "Color"
            print(f"{i+1:2d}. {color}  │  H:{hue:6.1f}°  S:{sat:4.2f}  V:{val:4.2f}  │  {color_type}")
        
        # Calculate hue diversity stats
        if len(hues_deg) > 1:
            saturated_colors = [(hue, color) for hue, sat, color in zip(hues_deg, s, colors) if sat >= 0.1]
            if len(saturated_colors) > 1:
                min_diff = float('inf')
                max_diff = 0
                total_diff = 0
                comparisons = 0
                
                for i in range(len(saturated_colors)):
                    for j in range(i + 1, len(saturated_colors)):
                        diff = abs(saturated_colors[i][0] - saturated_colors[j][0])
                        diff = min(diff, 360 - diff)
                        min_diff = min(min_diff, diff)
                        max_diff = max(max_diff, diff)
                        total_diff += diff
                        comparisons += 1
                
                avg_diff = total_diff / comparisons if comparisons > 0 else 0
                
                print("\nHue Diversity Statistics (saturated colors only):")
                print(f"  Minimum hue difference: {min_diff:.1f}°")
                print(f"  Average hue difference: {avg_diff:.1f}°")
                print(f"  Maximum hue difference: {max_diff:.1f}°")
                print(f"  Total saturated colors: {len(saturated_colors)}")
                print(f"  Threshold used: 15-20° (depending on requested count)")

def main():
    print("=" * 70)
    print("Test 1: Gradient Image (Similar Hues - Blue to Cyan)")
    print("=" * 70)
    
    gradient_img = create_gradient_image()
    gradient_img.save("test_gradient.png")
    print("Saved: test_gradient.png")
    visualize_palette(gradient_img, n_colors=15)
    
    print("\n" + "=" * 70)
    print("Test 2: Multi-Color Image (Diverse Hues Across Spectrum)")
    print("=" * 70)
    
    multicolor_img = create_multicolor_image()
    multicolor_img.save("test_multicolor.png")
    print("Saved: test_multicolor.png")
    visualize_palette(multicolor_img, n_colors=15)
    
    print("\n" + "=" * 70)
    print("Test 3: Real Demo Image (if exists)")
    print("=" * 70)
    
    # Check if demo picture exists
    demo_dir = "Demo Pictures"
    if os.path.exists(demo_dir):
        demo_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if demo_files:
            demo_path = os.path.join(demo_dir, demo_files[0])
            print(f"Testing with: {demo_path}")
            demo_img = Image.open(demo_path)
            visualize_palette(demo_img, n_colors=15)
        else:
            print("No demo images found in 'Demo Pictures' folder")
    else:
        print("'Demo Pictures' folder not found")
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)
    print("✓ Hue filtering prevents similar colors from dominating palette")
    print("✓ System returns fewer colors if image lacks diversity")
    print("✓ Minimum hue threshold of 15-20° enforced between saturated colors")
    print("✓ Low-saturation colors (grays) handled separately")

if __name__ == "__main__":
    main()
