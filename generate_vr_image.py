import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def create_vr_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    print(f"Processing image: {image_path}")
    
    # Set standard eye resolution
    eye_size = 600
    img = cv2.resize(img, (eye_size, eye_size))
    
    # A simple disparity shift for the two eyes (3% shift) to create depth
    shift = int(eye_size * 0.03)
    
    left_eye = np.zeros_like(img)
    right_eye = np.zeros_like(img)
    
    # Left eye: shift image slightly to the right
    left_eye[:, shift:] = img[:, :-shift]
    left_eye[:, :shift] = img[:, 0:1] # Edge padding
    
    # Right eye: shift image slightly to the left
    right_eye[:, :-shift] = img[:, shift:]
    right_eye[:, -shift:] = img[:, -1:] # Edge padding
    
    # Add a circular vignette mask to mimic a VR headset lens
    mask = np.zeros((eye_size, eye_size), dtype=np.uint8)
    cv2.circle(mask, (eye_size//2, eye_size//2), int(eye_size * 0.48), 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0) / 255.0
    mask = np.dstack([mask]*3)
    
    left_eye = (left_eye * mask).astype(np.uint8)
    right_eye = (right_eye * mask).astype(np.uint8)

    # Add a border/background for the Side-By-Side (SBS) view
    vr_bg = np.zeros((eye_size + 100, eye_size * 2 + 150, 3), dtype=np.uint8)
    
    # Paste left and right eyes side by side
    vr_bg[50:50+eye_size, 50:50+eye_size] = left_eye
    vr_bg[50:50+eye_size, 100+eye_size:100+2*eye_size] = right_eye
    
    # Add descriptive text overlays
    cv2.putText(vr_bg, "VR Integrated View (Stereoscopic SBS)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vr_bg, "Left Eye", (50 + eye_size//2 - 50, eye_size + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(vr_bg, "Right Eye", (100 + eye_size + eye_size//2 - 60, eye_size + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Save the stereoscopic output right next to the original file
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_dir = os.path.dirname(image_path)
    output_path = os.path.join(output_dir, f"{name}_vr_integrated{ext}")
    
    cv2.imwrite(output_path, vr_bg)
    print(f"Successfully generated VR image at {output_path}")

    # Display the final output image directly to the user
    cv2.imshow("VR Output Image - Press any key to close", vr_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output_path

if __name__ == '__main__':
    # Initialize a file picker so the user can easily select any image
    root = tk.Tk()
    root.withdraw() # Hide the main root window
    
    print("A file dialog should appear. Please select the image you want to convert...")
    
    # Force window to top
    root.attributes("-topmost", True)
    
    file_path = filedialog.askopenfilename(
        title="Select your workspace image to convert to VR",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if file_path:
        out = create_vr_image(file_path)
        print(f"\nDone! The VR output image is saved side-by-side with your original at: {out}")
    else:
        print("\nNo file selected. Exiting script.")
