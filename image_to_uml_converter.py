#!/usr/bin/env python3
"""
Image to UML Converter Script

This script processes images from the /images directory and sends them to OpenAI's API
to convert diagram images into UML syntax.
"""

import os
import base64
import json
from pathlib import Path
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ImageToUMLConverter:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the converter with Gemini API key."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Error encoding image {image_path}: {str(e)}")
    
    def get_image_files(self, images_dir: str) -> List[str]:
        """Get all supported image files from the images directory."""
        images_path = Path(images_dir)
        
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory '{images_dir}' not found.")
        
        image_files = []
        for file_path in images_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def create_uml_prompt(self) -> str:
        """Create the prompt for UML conversion."""
        return """
        Please analyze this diagram image and convert it to UML syntax. 

        Instructions:
        1. Identify the type of diagram (class diagram, sequence diagram, use case diagram, etc.)
        2. Extract all elements, relationships, and annotations from the image
        3. Convert to appropriate UML syntax (PlantUML format preferred)
        4. Include all visible text, labels, and relationships
        5. Maintain the structure and hierarchy shown in the diagram
        6. If the image is unclear or not a diagram, please indicate that

        Please provide:
        - Diagram type identification
        - Complete UML syntax
        - Any notes about unclear elements

        Format your response as:
        **Diagram Type:** [type]
        
        **UML Syntax:**
        ```plantuml
        [UML code here]
        ```
        
        **Notes:** [any additional notes]
        """
    
    def convert_image_to_uml(self, image_path: str) -> Dict:
        """Send image to Gemini API and get UML conversion."""
        try:
            # Encode image
            base64_image = self.encode_image_to_base64(image_path)
            
            # Get image mime type
            image_ext = Path(image_path).suffix.lower()
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_type_map.get(image_ext, 'image/jpeg')
            
            # Prepare the request
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": self.create_uml_prompt()
                            },
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 4096
                }
            }
            
            # Make the API request
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract content from Gemini response
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                usage_metadata = result.get('usageMetadata', {})
            else:
                raise Exception("No valid response from Gemini API")
            
            return {
                "success": True,
                "image_path": image_path,
                "uml_content": content,
                "usage": usage_metadata,
                "error": None
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "image_path": image_path,
                "uml_content": None,
                "usage": {},
                "error": f"API request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "image_path": image_path,
                "uml_content": None,
                "usage": {},
                "error": f"Conversion failed: {str(e)}"
            }
    
    def save_uml_output(self, result: Dict, output_dir: str = "uml_output") -> str:
        """Save UML output to a file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_name = Path(result["image_path"]).stem
        output_file = output_path / f"{image_name}_uml.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# UML Conversion for {Path(result['image_path']).name}\n\n")
            f.write(f"**Source Image:** {result['image_path']}\n\n")
            
            if result["success"]:
                f.write(result["uml_content"])
                f.write(f"\n\n**API Usage:** {result['usage']}\n")
            else:
                f.write(f"**Error:** {result['error']}\n")
        
        return str(output_file)
    
    def process_all_images(self, images_dir: str = "images", output_dir: str = "uml_output") -> List[Dict]:
        """Process all images in the directory and convert to UML."""
        print(f"🔍 Scanning for images in '{images_dir}' directory...")
        
        try:
            image_files = self.get_image_files(images_dir)
            
            if not image_files:
                print(f"❌ No supported image files found in '{images_dir}'")
                print(f"   Supported formats: {', '.join(self.supported_formats)}")
                return []
            
            print(f"📸 Found {len(image_files)} image(s) to process")
            
            results = []
            total_tokens = 0
            
            for i, image_path in enumerate(image_files, 1):
                print(f"\n🔄 Processing {i}/{len(image_files)}: {Path(image_path).name}")
                
                result = self.convert_image_to_uml(image_path)
                results.append(result)
                
                if result["success"]:
                    output_file = self.save_uml_output(result, output_dir)
                    print(f"✅ Success! UML saved to: {output_file}")
                    
                    tokens_used = result["usage"].get("total_tokens", 0)
                    total_tokens += tokens_used
                    print(f"   Tokens used: {tokens_used}")
                else:
                    print(f"❌ Failed: {result['error']}")
            
            print(f"\n📊 Summary:")
            print(f"   Total images processed: {len(image_files)}")
            print(f"   Successful conversions: {sum(1 for r in results if r['success'])}")
            print(f"   Failed conversions: {sum(1 for r in results if not r['success'])}")
            print(f"   Total tokens used: {total_tokens}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing images: {str(e)}")
            return []


def main():
    """Main function to run the image to UML converter."""
    print("🚀 Image to UML Converter")
    print("=" * 50)
    
    try:
        # Initialize converter
        converter = ImageToUMLConverter()
        
        # Process all images
        results = converter.process_all_images()
        
        if results:
            print(f"\n✨ Processing complete! Check the 'uml_output' directory for results.")
        else:
            print(f"\n⚠️  No images were processed. Please check your images directory and API key.")
            
    except ValueError as e:
        print(f"❌ Configuration error: {str(e)}")
        print("\n💡 Setup instructions:")
        print("   1. Get a Gemini API key from https://aistudio.google.com/app/apikey")
        print("   2. Add it to your .env file: GEMINI_API_KEY=your_key_here")
        print("   3. Create an 'images' directory and add your diagram images")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()