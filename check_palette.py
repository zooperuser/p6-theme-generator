"""Analyze a specific palette for hue separation violations."""
import numpy as np

def rgb_to_hsv_np(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB to HSV."""
    a = arr.astype(np.float32) / 255.0
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    cmax = np.max(a, axis=1)
    cmin = np.min(a, axis=1)
    delta = cmax - cmin + 1e-8
    
    h = np.zeros_like(cmax)
    mask = delta > 1e-7
    rc = ((g - b) / delta) % 6
    gc = ((b - r) / delta) + 2
    bc = ((r - g) / delta) + 4
    choice = np.argmax(a == cmax[:, None], axis=1)
    
    h[mask & (choice == 0)] = rc[mask & (choice == 0)]
    h[mask & (choice == 1)] = gc[mask & (choice == 1)]
    h[mask & (choice == 2)] = bc[mask & (choice == 2)]
    h = (h / 6.0) % 1.0
    
    s = np.where(cmax <= 1e-7, 0.0, delta / (cmax + 1e-8))
    v = cmax
    return h, s, v

# Parse the palette
palette_str = "#108E52, #F5BF23, #3874B2, #F8F8EE, #E59C3B, #31AECD, #93B9CF, #B72238, #FEFBEA, #FDFEFE, #8E5931, #D67892, #822075, #0FA595, #7DA71D"
colors = [c.strip() for c in palette_str.split(',')]

print("=" * 80)
print("PALETTE ANALYSIS: Checking 2-3 Position Separation Rule")
print("=" * 80)
print(f"\nTotal colors: {len(colors)}")
print("\nPalette order:")

# Convert to RGB
rgb_colors = []
for i, hex_color in enumerate(colors):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    rgb_colors.append([r, g, b])
    print(f"{i+1:2d}. {colors[i]}")

# Convert to HSV
rgb_arr = np.array(rgb_colors, dtype=np.float32)
h, s, v = rgb_to_hsv_np(rgb_arr)
hues_deg = h * 360

print("\n" + "=" * 80)
print("HUE ANALYSIS")
print("=" * 80)

for i, (color, hue, sat, val) in enumerate(zip(colors, hues_deg, s, v)):
    color_type = "Gray" if sat < 0.1 else "Color"
    print(f"{i+1:2d}. {color}  │  H:{hue:6.1f}°  S:{sat:4.2f}  V:{val:4.2f}  │  {color_type}")

print("\n" + "=" * 80)
print("SEPARATION ANALYSIS (checking for similar hues within 30°)")
print("=" * 80)

violations = []
warnings = []

for i in range(len(colors)):
    if s[i] < 0.1:  # Skip grays
        continue
    
    for j in range(i + 1, len(colors)):
        if s[j] < 0.1:  # Skip grays
            continue
        
        # Calculate hue difference
        hue_diff = abs(hues_deg[i] - hues_deg[j])
        hue_diff = min(hue_diff, 360 - hue_diff)
        
        # If hues are similar (within 30°)
        if hue_diff < 30:
            # Calculate position distance
            pos_dist = abs(i - j)
            
            if pos_dist < 3:
                violations.append({
                    'pos1': i + 1,
                    'pos2': j + 1,
                    'color1': colors[i],
                    'color2': colors[j],
                    'hue_diff': hue_diff,
                    'pos_dist': pos_dist
                })
            elif pos_dist == 3:
                warnings.append({
                    'pos1': i + 1,
                    'pos2': j + 1,
                    'color1': colors[i],
                    'color2': colors[j],
                    'hue_diff': hue_diff,
                    'pos_dist': pos_dist
                })

if violations:
    print("\n❌ VIOLATIONS FOUND (similar hues < 3 positions apart):")
    print("-" * 80)
    for v in violations:
        print(f"  Position {v['pos1']:2d} ({v['color1']}) and {v['pos2']:2d} ({v['color2']})")
        print(f"    Hue difference: {v['hue_diff']:.1f}° | Distance: {v['pos_dist']} positions ❌")
else:
    print("\n✓ No violations found!")

if warnings:
    print("\n⚠ BORDERLINE CASES (exactly 3 positions apart):")
    print("-" * 80)
    for w in warnings:
        print(f"  Position {w['pos1']:2d} ({w['color1']}) and {w['pos2']:2d} ({w['color2']})")
        print(f"    Hue difference: {w['hue_diff']:.1f}° | Distance: {w['pos_dist']} positions ⚠")

if not violations and not warnings:
    print("\n✓ All similar hues are properly separated (3+ positions apart)!")

# Statistics
saturated_colors = [(i, hues_deg[i]) for i in range(len(colors)) if s[i] >= 0.1]
if len(saturated_colors) > 1:
    min_hue_diff = float('inf')
    for i in range(len(saturated_colors)):
        for j in range(i + 1, len(saturated_colors)):
            hue_diff = abs(saturated_colors[i][1] - saturated_colors[j][1])
            hue_diff = min(hue_diff, 360 - hue_diff)
            min_hue_diff = min(min_hue_diff, hue_diff)
    
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total colors: {len(colors)}")
    print(f"Saturated colors: {len(saturated_colors)}")
    print(f"Gray/low-sat colors: {len(colors) - len(saturated_colors)}")
    print(f"Minimum hue difference: {min_hue_diff:.1f}°")
    print(f"Violations (< 3 positions): {len(violations)}")
    print(f"Borderline cases (= 3 positions): {len(warnings)}")
