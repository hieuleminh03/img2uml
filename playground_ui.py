#!/usr/bin/env python3
"""
Image to UML Converter Playground UI

A Streamlit-based web interface for converting diagram images to UML syntax
with image cropping capabilities.
"""

import streamlit as st
import os
import tempfile
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from typing import Tuple, Optional, List, Dict
import base64
import io

# Try to import OpenCV, but make it optional
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("⚠️ OpenCV not available. Auto-detect regions feature will use simplified edge detection.")

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
        if OPENCV_AVAILABLE:
            return ImageCropper._detect_regions_opencv(image, min_area)
        else:
            return ImageCropper._detect_regions_pil(image, min_area)
    
    @staticmethod
    def _detect_regions_opencv(image: Image.Image, min_area: int) -> List[Tuple[int, int, int, int]]:
        """OpenCV-based region detection."""
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
    
    @staticmethod
    def _detect_regions_pil(image: Image.Image, min_area: int) -> List[Tuple[int, int, int, int]]:
        """PIL-based simplified region detection."""
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        # Apply edge detection using PIL filters
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Convert to numpy for processing
        edge_array = np.array(edges)
        
        # Simple thresholding
        threshold = np.mean(edge_array) + np.std(edge_array)
        binary = edge_array > threshold
        
        # Find connected components (simplified)
        regions = []
        height, width = binary.shape
        
        # Simple region growing approach
        visited = np.zeros_like(binary)
        
        for y in range(0, height, 20):  # Sample every 20 pixels
            for x in range(0, width, 20):
                if binary[y, x] and not visited[y, x]:
                    # Find bounding box of this region
                    min_x, max_x = x, x
                    min_y, max_y = y, y
                    
                    # Expand region
                    for dy in range(-50, 51, 10):
                        for dx in range(-50, 51, 10):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width and binary[ny, nx]:
                                min_x = min(min_x, nx)
                                max_x = max(max_x, nx)
                                min_y = min(min_y, ny)
                                max_y = max(max_y, ny)
                                visited[ny, nx] = True
                    
                    # Check if region is large enough
                    area = (max_x - min_x) * (max_y - min_y)
                    if area > min_area:
                        regions.append((min_x, min_y, max_x, max_y))
        
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

def git_commit_and_push(message: str = "Auto-commit: UML conversion results") -> bool:
    """Commit and push changes to git repository."""
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            return False
        
        # Add all changes
        subprocess.run(['git', 'add', '.'], check=True)
        
        # Commit changes
        subprocess.run(['git', 'commit', '-m', message], check=True)
        
        # Push to remote
        subprocess.run(['git', 'push'], check=True)
        
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

def combine_uml_results(results: List[Dict]) -> str:
    """Combine multiple UML conversion results into a single document."""
    if not results:
        return ""
    
    combined = "# Combined UML Results\n\n"
    combined += f"Generated from {len(results)} image sections\n\n"
    
    for i, result in enumerate(results, 1):
        if result.get("success") and result.get("uml_content"):
            combined += f"## Section {i}: {result.get('name', f'Part {i}')}\n\n"
            combined += result["uml_content"]
            combined += "\n\n---\n\n"
    
    return combined

def create_horizontal_results_view(results: List[Dict]) -> None:
    """Create a horizontal scrollable view for results."""
    if not results:
        return
    
    st.subheader("📋 Conversion Results")
    
    # Create tabs for horizontal navigation
    tab_names = [f"{result.get('name', f'Part {i+1}')}" for i, result in enumerate(results)]
    tabs = st.tabs(tab_names)
    
    for i, (tab, result) in enumerate(zip(tabs, results)):
        with tab:
            if result.get("success"):
                col_img, col_uml = st.columns([1, 2])
                
                with col_img:
                    if result.get("image"):
                        st.image(result["image"], caption=result.get("name", f"Part {i+1}"), use_container_width=True)
                
                with col_uml:
                    st.markdown(result["uml_content"])
                
                # Usage info
                if result.get("usage"):
                    st.info(f"Tokens used: {result['usage'].get('total_tokens', 'N/A')}")
                
                # Individual download button
                st.download_button(
                    label=f"📥 Download UML for {result.get('name', f'Part {i+1}')}",
                    data=result["uml_content"],
                    file_name=f"{result.get('name', f'part_{i+1}').lower().replace(' ', '_')}_uml.md",
                    mime="text/markdown",
                    key=f"download_{i}"
                )
            else:
                st.error(f"❌ Conversion failed: {result.get('error', 'Unknown error')}")

def main():
    st.title("🎨 Image to UML Converter Playground")
    st.markdown("Upload diagram images and convert them to UML syntax with optional cropping features.")
    
    # Initialize session state
    if 'cropped_images' not in st.session_state:
        st.session_state.cropped_images = []
    if 'crop_names' not in st.session_state:
        st.session_state.crop_names = []
    if 'conversion_results' not in st.session_state:
        st.session_state.conversion_results = []
    if 'combined_uml' not in st.session_state:
        st.session_state.combined_uml = ""
    
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
        
        # Clear session state when new file is uploaded
        if uploaded_file is not None:
            if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.cropped_images = []
                st.session_state.crop_names = []
                st.session_state.conversion_results = []
                st.session_state.combined_uml = ""
                st.session_state.last_uploaded_file = uploaded_file.name
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
                    st.session_state.cropped_images = [cropped]
                    st.session_state.crop_names = ["Manual Crop"]
            
            elif crop_method == "Grid split":
                st.info("Split image into a grid of smaller sections")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    rows = st.number_input("Rows", min_value=1, max_value=10, value=2)
                with col_b:
                    cols = st.number_input("Columns", min_value=1, max_value=10, value=2)
                
                if st.button("Generate Grid"):
                    st.session_state.cropped_images = ImageCropper.split_image_grid(image, rows, cols)
                    st.session_state.crop_names = [f"Grid {i//cols + 1}-{i%cols + 1}" for i in range(len(st.session_state.cropped_images))]
            
            elif crop_method == "Auto-detect regions":
                if OPENCV_AVAILABLE:
                    st.info("Automatically detect content regions using OpenCV edge detection")
                else:
                    st.info("Automatically detect content regions using simplified edge detection (OpenCV not available)")
                
                min_area = st.slider("Minimum region area", 500, 10000, 1000)
                
                if st.button("Detect Regions"):
                    with st.spinner("Detecting regions..."):
                        regions = ImageCropper.detect_content_regions(image, min_area)
                    
                    if regions:
                        preview_img = display_image_with_crops(image, regions)
                        st.image(preview_img, caption=f"Detected {len(regions)} regions", use_container_width=True)
                        
                        st.session_state.cropped_images = [ImageCropper.crop_image(image, region) for region in regions]
                        st.session_state.crop_names = [f"Region {i+1}" for i in range(len(st.session_state.cropped_images))]
                    else:
                        st.warning("No content regions detected. Try adjusting the minimum area or use manual/grid cropping.")
            
            else:  # No cropping
                st.session_state.cropped_images = [image]
                st.session_state.crop_names = ["Original"]
            
            # Display cropped images
            if st.session_state.cropped_images:
                st.subheader("🖼️ Images to Process")
                
                # Show thumbnails
                cols = st.columns(min(len(st.session_state.cropped_images), 4))
                for i, (img, name) in enumerate(zip(st.session_state.cropped_images, st.session_state.crop_names)):
                    with cols[i % 4]:
                        st.image(img, caption=name, use_container_width=True)
    
    with col2:
        st.header("🔄 Convert to UML")
        
        if uploaded_file is not None and st.session_state.cropped_images:
            # Conversion options
            st.subheader("⚙️ Conversion Settings")
            
            selected_images = st.multiselect(
                "Select images to convert:",
                options=list(range(len(st.session_state.cropped_images))),
                default=list(range(len(st.session_state.cropped_images))),
                format_func=lambda x: st.session_state.crop_names[x]
            )
            
            # Git options
            st.subheader("🔧 Git Options")
            auto_commit = st.checkbox("Auto-commit and push after conversion", value=False)
            commit_message = st.text_input("Commit message", value="Auto-commit: UML conversion results")
            
            if st.button("🚀 Convert to UML", type="primary"):
                if not selected_images:
                    st.warning("Please select at least one image to convert.")
                    return
                
                progress_bar = st.progress(0)
                st.session_state.conversion_results = []
                
                # Process all selected images
                for i, img_idx in enumerate(selected_images):
                    progress_bar.progress((i + 1) / len(selected_images))
                    
                    img = st.session_state.cropped_images[img_idx]
                    name = st.session_state.crop_names[img_idx]
                    
                    with st.spinner(f"Converting {name}..."):
                        # Save image temporarily
                        temp_path = save_temp_image(img)
                        
                        try:
                            # Convert to UML
                            result = converter.convert_image_to_uml(temp_path)
                            
                            # Store result with additional metadata
                            result_data = {
                                "name": name,
                                "image": img,
                                "success": result["success"],
                                "uml_content": result.get("uml_content", ""),
                                "usage": result.get("usage", {}),
                                "error": result.get("error", "")
                            }
                            st.session_state.conversion_results.append(result_data)
                        
                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
                
                progress_bar.progress(1.0)
                
                # Generate combined UML
                st.session_state.combined_uml = combine_uml_results(st.session_state.conversion_results)
                
                st.success("🎉 All conversions completed!")
                
                # Auto-commit if enabled
                if auto_commit and st.session_state.combined_uml:
                    with st.spinner("Committing and pushing to git..."):
                        if git_commit_and_push(commit_message):
                            st.success("✅ Successfully committed and pushed to git!")
                        else:
                            st.warning("⚠️ Git commit/push failed. Make sure you're in a git repository with proper remote setup.")
            
            # Display results if available
            if st.session_state.conversion_results:
                # Horizontal scrollable results view
                create_horizontal_results_view(st.session_state.conversion_results)
                
                # Combined results section
                st.subheader("📄 Combined Results")
                
                if st.session_state.combined_uml:
                    # Show combined UML in an expandable section
                    with st.expander("View Combined UML", expanded=False):
                        st.markdown(st.session_state.combined_uml)
                    
                    # Download combined results
                    st.download_button(
                        label="📥 Download Combined UML",
                        data=st.session_state.combined_uml,
                        file_name="combined_uml_results.md",
                        mime="text/markdown",
                        type="primary"
                    )
                    
                    # Summary statistics
                    successful_conversions = sum(1 for r in st.session_state.conversion_results if r["success"])
                    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in st.session_state.conversion_results)
                    
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("Successful Conversions", f"{successful_conversions}/{len(st.session_state.conversion_results)}")
                    with col_stats2:
                        st.metric("Total Tokens Used", total_tokens)
                    with col_stats3:
                        st.metric("Grid Sections", len(st.session_state.conversion_results))
        
        elif uploaded_file is None:
            st.info("👆 Upload an image to get started")
        else:
            st.info("👈 Select a cropping method and generate images to convert")
    
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