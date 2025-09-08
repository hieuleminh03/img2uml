#!/usr/bin/env python3
"""
Image to UML Converter Playground UI

A Streamlit-based web interface for converting diagram images to UML syntax
with image cropping capabilities.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np
from typing import Tuple, Optional, List
import base64
import io

from image_to_uml_converter import ImageToUMLConverter

# Page configuration
st.set_page_config(
    page_title="Image to UML Converter Playground",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ImageCropper:
    """Handles image cropping functionality."""
    
    @staticmethod
    def crop_image(image: Image.Image, crop_coords: Tuple[int, int, int, int]) -> Image.Image:
        """Crop image using coordinates (left, top, right, bottom)."""
        return image.crop(crop_coords)
    
    @staticmethod
    def split_image_grid(image: Image.Image, rows: int, cols: int) -> List[Image.Image]:
        """Split image into a grid of smaller images."""
        width, height = image.size
        crop_width = width // cols
        crop_height = height // rows
        
        crops = []
        for row in range(rows):
            for col in range(cols):
                left = col * crop_width
                top = row * crop_height
                right = left + crop_width
                bottom = top + crop_height
                
                crop = image.crop((left, top, right, bottom))
                crops.append(crop)
        
        return crops
    
    @staticmethod
    def detect_content_regions(image: Image.Image, min_area: int = 1000) -> List[Tuple[int, int, int, int]]:
        """Detect content regions using edge detection."""
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > min_area:  # Filter small regions
                regions.append((x, y, x + w, y + h))
        
        return regions

def save_temp_image(image: Image.Image, suffix: str = ".png") -> str:
    """Save PIL image to temporary file and return path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    image.save(temp_file.name)
    return temp_file.name

def display_image_with_crops(image: Image.Image, crops: List[Tuple[int, int, int, int]]) -> Image.Image:
    """Display image with crop regions highlighted."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    for i, (left, top, right, bottom) in enumerate(crops):
        color = colors[i % len(colors)]
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left + 5, top + 5), f"Region {i+1}", fill=color)
    
    return img_copy

def main():
    st.title("🎨 Image to UML Converter Playground")
    st.markdown("Upload diagram images and convert them to UML syntax with optional cropping features.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Gemini API key. You can also set it in the .env file."
    )
    
    # Initialize converter
    try:
        if api_key:
            converter = ImageToUMLConverter(api_key=api_key)
        else:
            converter = ImageToUMLConverter()
        st.sidebar.success("✅ API key configured")
    except ValueError as e:
        st.sidebar.error(f"❌ {str(e)}")
        st.error("Please configure your Gemini API key to continue.")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload & Crop")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            help="Upload a diagram image to convert to UML"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(image, caption=f"Size: {image.size[0]}x{image.size[1]}", use_container_width=True)
            
            # Cropping options
            st.subheader("🔧 Cropping Options")
            
            crop_method = st.selectbox(
                "Select cropping method:",
                ["No cropping", "Manual crop", "Grid split", "Auto-detect regions"]
            )
            
            cropped_images = []
            crop_names = []
            
            if crop_method == "Manual crop":
                st.info("Use the sliders below to define crop region")
                
                # Manual crop controls
                left = st.slider("Left", 0, image.size[0], 0)
                top = st.slider("Top", 0, image.size[1], 0)
                right = st.slider("Right", left, image.size[0], image.size[0])
                bottom = st.slider("Bottom", top, image.size[1], image.size[1])
                
                if st.button("Preview Crop"):
                    crop_coords = (left, top, right, bottom)
                    preview_img = display_image_with_crops(image, [crop_coords])
                    st.image(preview_img, caption="Crop Preview", use_container_width=True)
                    
                    cropped = ImageCropper.crop_image(image, crop_coords)
                    cropped_images = [cropped]
                    crop_names = ["Manual Crop"]
            
            elif crop_method == "Grid split":
                st.info("Split image into a grid of smaller sections")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    rows = st.number_input("Rows", min_value=1, max_value=10, value=2)
                with col_b:
                    cols = st.number_input("Columns", min_value=1, max_value=10, value=2)
                
                if st.button("Generate Grid"):
                    cropped_images = ImageCropper.split_image_grid(image, rows, cols)
                    crop_names = [f"Grid {i//cols + 1}-{i%cols + 1}" for i in range(len(cropped_images))]
            
            elif crop_method == "Auto-detect regions":
                st.info("Automatically detect content regions using edge detection")
                
                min_area = st.slider("Minimum region area", 500, 10000, 1000)
                
                if st.button("Detect Regions"):
                    regions = ImageCropper.detect_content_regions(image, min_area)
                    
                    if regions:
                        preview_img = display_image_with_crops(image, regions)
                        st.image(preview_img, caption=f"Detected {len(regions)} regions", use_container_width=True)
                        
                        cropped_images = [ImageCropper.crop_image(image, region) for region in regions]
                        crop_names = [f"Region {i+1}" for i in range(len(cropped_images))]
                    else:
                        st.warning("No content regions detected. Try adjusting the minimum area.")
            
            else:  # No cropping
                cropped_images = [image]
                crop_names = ["Original"]
            
            # Display cropped images
            if cropped_images:
                st.subheader("🖼️ Images to Process")
                
                # Show thumbnails
                cols = st.columns(min(len(cropped_images), 4))
                for i, (img, name) in enumerate(zip(cropped_images, crop_names)):
                    with cols[i % 4]:
                        st.image(img, caption=name, use_container_width=True)
    
    with col2:
        st.header("🔄 Convert to UML")
        
        if uploaded_file is not None and cropped_images:
            # Conversion options
            st.subheader("⚙️ Conversion Settings")
            
            selected_images = st.multiselect(
                "Select images to convert:",
                options=list(range(len(cropped_images))),
                default=list(range(len(cropped_images))),
                format_func=lambda x: crop_names[x]
            )
            
            if st.button("🚀 Convert to UML", type="primary"):
                if not selected_images:
                    st.warning("Please select at least one image to convert.")
                    return
                
                progress_bar = st.progress(0)
                results_container = st.container()
                
                for i, img_idx in enumerate(selected_images):
                    progress_bar.progress((i + 1) / len(selected_images))
                    
                    img = cropped_images[img_idx]
                    name = crop_names[img_idx]
                    
                    with results_container:
                        st.subheader(f"📋 Results for {name}")
                        
                        with st.spinner(f"Converting {name}..."):
                            # Save image temporarily
                            temp_path = save_temp_image(img)
                            
                            try:
                                # Convert to UML
                                result = converter.convert_image_to_uml(temp_path)
                                
                                if result["success"]:
                                    st.success(f"✅ Conversion successful!")
                                    
                                    # Display results
                                    col_img, col_uml = st.columns([1, 2])
                                    
                                    with col_img:
                                        st.image(img, caption=name, use_container_width=True)
                                    
                                    with col_uml:
                                        st.markdown(result["uml_content"])
                                    
                                    # Usage info
                                    if result.get("usage"):
                                        st.info(f"Tokens used: {result['usage'].get('total_tokens', 'N/A')}")
                                    
                                    # Download button
                                    st.download_button(
                                        label=f"📥 Download UML for {name}",
                                        data=result["uml_content"],
                                        file_name=f"{name.lower().replace(' ', '_')}_uml.md",
                                        mime="text/markdown"
                                    )
                                
                                else:
                                    st.error(f"❌ Conversion failed: {result['error']}")
                            
                            finally:
                                # Clean up temporary file
                                try:
                                    os.unlink(temp_path)
                                except:
                                    pass
                
                progress_bar.progress(1.0)
                st.success("🎉 All conversions completed!")
        
        else:
            st.info("👆 Upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** "
        "• Use cropping to focus on specific parts of complex diagrams "
        "• Grid split works well for multi-diagram images "
        "• Auto-detect helps find individual diagram components"
    )

if __name__ == "__main__":
    main()