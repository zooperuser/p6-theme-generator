# Image Palette Generator (Image-Derived Palettes)

This app generates color palettes from uploaded images. It focuses on:

- Random image pixel palettes (perceptually weighted sampling + LAB k-means)
- A chaotic “painting-style” variant using the same colors

## Features

- Upload an image and generate a palette from random pixels (adjustable count)
- Chaotic “painting-style” palette (reordered colors for visual balance)
- Copyable hex codes and recommended text colors for both palettes

## Prerequisites

- Windows (PowerShell examples below) or any OS with Python 3.10+

## 1. Get the code

Clone or copy this repository to your machine.

## 2. Create & Activate Virtual Environment

```pwsh
python -m venv .venv
./.venv/Scripts/Activate.ps1
```

## 3. Install Dependencies

```pwsh
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure Environment (optional)

No special configuration is required. The app runs locally and only needs Pillow/Numpy/Gradio.

## 5. Run the App

```pwsh
python app.py
```

Then open the local Gradio URL printed in the console (e.g. <http://127.0.0.1:7861/>).

### Image-based Random Colors

1. Upload an image in the center column.
2. Use the "Colors" slider to choose how many colors to sample.
3. Click "Generate" to produce the random palette (left) and the mirrored palette (right).
   - The image palette updates automatically when you change the image.

## 6. Troubleshooting

- h11 LocalProtocolError: Too little data for declared Content-Length
   - Some client+server combinations trigger this when brotli compression engages.
   - This app proactively disables brotli by stripping `br` from `Accept-Encoding` via a tiny middleware.
   - If you still see this, ensure you're on the pinned Gradio version in `requirements.txt` and restart the app.


## 9. License & Attribution

Original concept & color logic adapted from public color palette ideas.
This adaptation is provided for local/offline experimentation.

---

Enjoy creating image-based palettes locally! 🎨
