import glob
import re
import os
import imageio
import numpy as np
import argparse
from PIL import Image

def parse_color(color_str):
    """Convert color string to (R, G, B) tuple in 0-255 range."""
    color_str = color_str.lower()
    if color_str == 'white':
        return (255, 255, 255)
    if color_str == 'black':
        return (0, 0, 0)
    if color_str == 'transparent':
        return None
    if color_str.startswith('#'):
        hex_str = color_str[1:]
        if len(hex_str) == 3:
            r = int(hex_str[0]*2, 16)
            g = int(hex_str[1]*2, 16)
            b = int(hex_str[2]*2, 16)
        elif len(hex_str) == 6:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
        else:
            raise ValueError(f"Invalid hex color: {color_str}")
        return (r, g, b)
    raise ValueError(f"Unsupported color format: {color_str}")

def composite_with_background(image, bg_color):
    """Composite RGBA image over specified background color."""
    if image.shape[2] != 4:
        return image  # Already RGB
    
    alpha = image[..., 3:4].astype(np.float32) / 255.0
    rgb = image[..., :3].astype(np.float32)
    
    if bg_color is None:
        return image[..., :3]  # Drop alpha channel (semi-transparent will be opaque)
    
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    composite = rgb * alpha + bg * (1 - alpha)
    return composite.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Create looping GIF from images with transparency handling')
    parser.add_argument('--output', type=str, default='animation.gif', 
                        help='Output GIF filename (default: animation.gif)')
    parser.add_argument('--duration', type=float, default=0.1, 
                        help='Duration per frame in seconds (default: 0.1)')
    parser.add_argument('--pattern', type=str, default='u_*.png', 
                        help='Filename pattern (default: u_*.png). Example: img_*.jpg')
    parser.add_argument('--background', type=str, default='white',
                        help='Background for compositing: "white", "black", "#rrggbb", or "transparent"')
    parser.add_argument('--loop', type=int, default=0,
                        help='Number of loops (0 = infinite loop, default)')
    args = parser.parse_args()

    # Find and sort files
    files = glob.glob(args.pattern)
    if not files:
        print(f"Error: No files found matching pattern '{args.pattern}'")
        return 1

    file_list = []
    for f in files:
        if not os.path.isfile(f):
            continue
        match = re.search(r'(\d+)(?=\.[^.]*$)', os.path.basename(f))
        if match:
            file_list.append((int(match.group(1)), f))
    
    if not file_list:
        print("Error: No valid numbered files found. Files should contain numbers before the extension")
        return 1

    file_list.sort(key=lambda x: x[0])
    sorted_files = [f for _, f in file_list]

    # Process background color
    try:
        bg_color = parse_color(args.background)
    except ValueError as e:
        print(f"Error: {str(e)}")
        print('Valid options: "white", "black", "#ffffff", or "transparent"')
        return 1

    # Warn about transparency limitations
    if bg_color is None:
        print("WARNING: GIF format only supports BINARY transparency (not partial transparency).")
        print("         Semi-transparent pixels will appear as opaque in the final GIF.")
        print("         For best results, composite over a background color matching your use case.\n")

    # Create GIF with looping
    try:
        with imageio.get_writer(
            args.output, 
            mode='I', 
            duration=args.duration,
            loop=args.loop  # Critical for looping!
        ) as writer:
            for i, filename in enumerate(sorted_files, 1):
                img = Image.open(filename)
                
                # Convert to RGBA if not already
                if img.mode != 'RGBA':
                    bg = Image.new('RGBA', img.size, bg_color or (0, 0, 0, 0))
                    bg.paste(img, (0, 0), img if img.mode == 'RGBA' else None)
                    img = bg
                
                # Convert to numpy array for processing
                rgba = np.array(img)
                
                # Composite over background if needed
                if bg_color is not None:
                    rgb = composite_with_background(rgba, bg_color)
                else:
                    rgb = rgba[..., :3]  # Drop alpha for transparency preservation
                
                writer.append_data(rgb)
                print(f"Processed: {os.path.basename(filename)}", end='\r')
            
            print(f"\nSuccessfully created GIF: {args.output}")
            print(f"Frames: {len(sorted_files)} | Duration: {args.duration:.2f}s per frame")
            print(f"Looping: {'infinite' if args.loop == 0 else f'{args.loop} times'}")
            if bg_color is not None:
                print(f"Transparency: composited over {args.background}")
            else:
                print(f"Transparency: preserved (with GIF limitations)")
        return 0
    except Exception as e:
        print(f"\nError creating GIF: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())