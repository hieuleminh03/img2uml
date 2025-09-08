# Image to UML Converter Playground

A user-friendly web interface for converting diagram images to UML syntax with advanced cropping capabilities.

## Features

- 🖼️ **Image Upload**: Support for PNG, JPG, JPEG, GIF, and WebP formats
- ✂️ **Smart Cropping**: Multiple cropping methods to optimize conversion
  - Manual crop with interactive sliders
  - Grid split for multi-diagram images
  - Auto-detect content regions using edge detection
- 🔄 **Real-time Conversion**: Convert images to UML using OpenAI's GPT-4 Vision
- 📥 **Export Results**: Download UML output as Markdown files
- 🎨 **Visual Preview**: See crop regions before processing

## Setup Instructions

### 1. Set up Virtual Environment

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 2. Configure API Key

Add your Gemini API key to the `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Or enter it directly in the web interface.

### 3. Launch the Playground

**Option 1: Using the launcher script**
```bash
python run_playground.py
```

**Option 2: Direct Streamlit command**
```bash
streamlit run playground_ui.py
```

The playground will open in your browser at `http://localhost:8501`

## Usage Guide

### Basic Workflow

1. **Upload Image**: Click "Choose an image file" and select your diagram
2. **Choose Cropping Method**:
   - **No cropping**: Process the entire image
   - **Manual crop**: Use sliders to define a specific region
   - **Grid split**: Divide image into equal sections
   - **Auto-detect regions**: Automatically find diagram components
3. **Select Images**: Choose which cropped sections to convert
4. **Convert**: Click "Convert to UML" to process selected images
5. **Download**: Save UML output as Markdown files

### Cropping Methods Explained

#### Manual Crop
- Use sliders to define exact crop coordinates
- Perfect for focusing on specific diagram sections
- Preview shows the selected region

#### Grid Split
- Divides image into rows × columns grid
- Ideal for images containing multiple separate diagrams
- Each grid cell is processed independently

#### Auto-detect Regions
- Uses edge detection to find content areas
- Automatically identifies diagram boundaries
- Adjustable minimum area threshold to filter noise

### Tips for Best Results

1. **Image Quality**: Use high-resolution, clear images
2. **Cropping Strategy**: 
   - For complex diagrams: Use auto-detect or manual crop
   - For multiple diagrams: Use grid split
   - For single diagrams: No cropping usually works best
3. **Content Focus**: Crop to remove unnecessary whitespace and focus on diagram content
4. **Text Clarity**: Ensure text in diagrams is readable at the cropped resolution

## Troubleshooting

### Common Issues

**"API key not configured"**
- Ensure your OpenAI API key is set in `.env` or entered in the interface
- Verify the key is valid and has sufficient credits

**"No content regions detected"**
- Try adjusting the minimum area threshold
- Use manual crop instead for better control

**"Conversion failed"**
- Check your internet connection
- Verify the image is a valid diagram
- Try cropping to focus on clearer sections

### Performance Tips

- Crop images to focus on relevant content (reduces token usage)
- Process one image at a time for large/complex diagrams
- Use smaller grid sizes for better processing speed

## File Structure

```
├── playground_ui.py          # Main Streamlit application
├── run_playground.py         # Launcher script
├── image_to_uml_converter.py # Core conversion logic
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── images/                   # Sample images directory
```

## Dependencies

- **streamlit**: Web interface framework
- **Pillow**: Image processing
- **opencv-python**: Computer vision for auto-detection (optional)
- **numpy**: Numerical operations
- **requests**: HTTP client for API calls
- **python-dotenv**: Environment variable management

**Note**: OpenCV is optional. If not available, the app will use PIL-based edge detection for auto-crop features.

## API Usage

The playground uses Google's Gemini 1.5 Flash API with vision capabilities. Token usage is displayed after each conversion. Cropping can help reduce token consumption by focusing on relevant diagram content.

## Deployment

### Streamlit Cloud Deployment

1. **Fork/Clone** this repository
2. **Connect** to Streamlit Cloud
3. **Add** your `GEMINI_API_KEY` in Streamlit Cloud secrets
4. **Deploy** - the app will automatically handle OpenCV dependencies

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd img2uml

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the playground
streamlit run playground_ui.py
```

## Contributing

Feel free to enhance the playground with additional features:
- More cropping algorithms
- Different UML output formats
- Batch processing capabilities
- Image preprocessing options