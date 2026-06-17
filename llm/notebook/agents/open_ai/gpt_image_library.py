"""
GPT Image 1.5 Python Examples
=============================
OpenAI's latest image generation model with better instruction following,
precise editing, and 4x faster generation.

Requirements:
    pip install openai pillow

Set your API key:
    export OPENAI_API_KEY="your-api-key"
"""

import base64
import httpx
import os
from dotenv import load_dotenv
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image

load_dotenv()

# Initialize client with custom SSL certificate for corporate proxy
ssl_cert_path = os.environ.get("SSL_CERT_FILE", "~/netskope.pem")
http_client = httpx.Client(verify=ssl_cert_path)
# client = OpenAI(http_client=http_client)  # Uses OPENAI_API_KEY env var
client = OpenAI()


# =============================================================================
# 1. BASIC IMAGE GENERATION
# =============================================================================

def generate_basic_image():
    """Generate a simple image from text prompt."""
    result = client.images.generate(
        model="gpt-image-1.5",
        prompt="A cute baby sea otter floating on its back in calm blue water",
        n=1,
        size="1024x1024"
    )

    # Decode and save the image
    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("basic_output.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Basic image saved to basic_output.png")
    return result


# =============================================================================
# 2. QUALITY AND SIZE OPTIONS
# =============================================================================

def generate_with_quality_options():
    """Generate images with different quality and size settings."""

    # Quality options: "low", "medium", "high"
    # Size options: "1024x1024", "1536x1024" (landscape), "1024x1536" (portrait), "auto"

    result = client.images.generate(
        model="gpt-image-1.5",
        prompt="A professional product photo of a sleek modern smartphone",
        quality="high",  # Best quality (costs ~$0.17/image)
        size="1536x1024",  # Landscape format
        n=1
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("high_quality_landscape.png", "wb") as f:
        f.write(image_bytes)

    print("✅ High quality landscape image saved")
    return result


def generate_low_cost_image():
    """Generate a low-cost image (~$0.01/image)."""
    result = client.images.generate(
        model="gpt-image-1.5",
        prompt="A simple icon of a house",
        quality="low",
        size="1024x1024",
        n=1
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("low_cost_output.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Low cost image saved")
    return result


# =============================================================================
# 3. OUTPUT FORMAT OPTIONS
# =============================================================================

def generate_with_format_options():
    """Generate images with different output formats."""

    # Format options: "png", "jpeg", "webp"
    # Compression: 0-100 (only for webp and jpeg)

    result = client.images.generate(
        model="gpt-image-1.5",
        prompt="A colorful abstract painting with geometric shapes",
        quality="medium",
        size="1024x1024",
        output_format="webp",  # Smaller file size
        output_compression=80,  # 80% compression
        n=1
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("compressed_output.webp", "wb") as f:
        f.write(image_bytes)

    print("✅ Compressed WebP image saved")
    return result


# =============================================================================
# 4. TRANSPARENT BACKGROUND
# =============================================================================

def generate_transparent_background():
    """Generate an image with transparent background (PNG only)."""
    result = client.images.generate(
        model="gpt-image-1.5",
        prompt="A 3D rendered logo of a golden phoenix bird, isolated on transparent background",
        quality="high",
        size="1024x1024",
        output_format="png",
        background="transparent",  # Requires PNG format
        n=1
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("transparent_logo.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Transparent background image saved")
    return result


# =============================================================================
# 5. TEXT RENDERING IN IMAGES
# =============================================================================

def generate_image_with_text():
    """GPT Image 1.5 excels at rendering text in images."""

    prompt = """
    Create a professional business card design:
    - Name: "John Smith"
    - Title: "Senior Software Engineer"
    - Company: "TechCorp Inc."
    - Email: "john.smith@techcorp.com"
    - Phone: "+1 (555) 123-4567"

    Modern minimalist design with dark blue accents on white background.
    Clear, readable typography.
    """

    result = client.images.generate(
        model="gpt-image-1.5",
        prompt=prompt,
        quality="high",
        size="1536x1024",  # Landscape for business card
        n=1
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("business_card.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Business card with text saved")
    return result


# =============================================================================
# 6. IMAGE EDITING (INPAINTING)
# =============================================================================

def edit_image_with_mask(image_path: str, mask_path: str):
    """
    Edit specific parts of an image using a mask.
    The mask should have an alpha channel where transparent areas will be edited.
    """

    # Read the original image
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Read the mask (areas to edit should be transparent)
    with open(mask_path, "rb") as f:
        mask_data = f.read()

    result = client.images.edit(
        model="gpt-image-1.5",
        image=image_data,
        mask=mask_data,
        prompt="Replace the masked area with a beautiful sunset sky",
        size="1024x1024"
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("edited_output.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Edited image saved")
    return result


def edit_image_without_mask(image_path: str):
    """Edit an image with natural language instructions (no mask required)."""

    with open(image_path, "rb") as f:
        image_data = f.read()

    result = client.images.edit(
        model="gpt-image-1.5",
        image=image_data,
        prompt="Change the background to a tropical beach scene while keeping the main subject intact",
        size="1024x1024"
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("edited_no_mask.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Edited image (no mask) saved")
    return result


# =============================================================================
# 7. MULTIPLE REFERENCE IMAGES
# =============================================================================

def generate_from_multiple_images(image_paths: list):
    """
    Use multiple reference images to generate a new composite image.
    Supports up to 10 input images.
    """

    images = []
    for path in image_paths:
        with open(path, "rb") as f:
            images.append(f.read())

    result = client.images.edit(
        model="gpt-image-1.5",
        image=images,  # List of image bytes
        prompt="Create a gift basket containing all the items shown in the reference images, arranged beautifully with a red ribbon",
        size="1024x1024"
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("composite_output.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Composite image from multiple references saved")
    return result


# =============================================================================
# 8. STYLE TRANSFER
# =============================================================================

def style_transfer(content_image_path: str, style_description: str):
    """Apply a style to an existing image."""

    with open(content_image_path, "rb") as f:
        image_data = f.read()

    prompt = f"""
    Transform this image into {style_description}.
    Maintain the original composition and subject matter.
    Apply the style consistently across the entire image.
    """

    result = client.images.edit(
        model="gpt-image-1.5",
        image=image_data,
        prompt=prompt,
        quality="high",
        size="1024x1024"
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("style_transfer_output.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Style transfer image saved")
    return result


# =============================================================================
# 9. UI MOCKUP GENERATION
# =============================================================================

def generate_ui_mockup():
    """Generate a UI mockup - GPT Image 1.5 excels at this."""

    prompt = """
    Create a mobile app UI mockup for a farmer's market app:

    - Clean white header with "Fresh Market" title and a green leaf icon
    - Search bar below the header
    - Grid of vendor cards showing:
      - Small thumbnail image
      - Vendor name
      - Category (Produce, Bakery, etc.)
      - Rating stars
    - Bottom navigation with Home, Search, Cart, Profile icons

    Modern, minimal design. iOS style. 
    Place the UI in an iPhone frame.
    """

    result = client.images.generate(
        model="gpt-image-1.5",
        prompt=prompt,
        quality="high",
        size="1024x1536",  # Portrait for mobile
        n=1
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("ui_mockup.png", "wb") as f:
        f.write(image_bytes)

    print("✅ UI mockup saved")
    return result


# =============================================================================
# 10. BATCH GENERATION
# =============================================================================

def generate_batch(prompts: list, quality: str = "medium"):
    """Generate multiple images from a list of prompts."""

    results = []
    for i, prompt in enumerate(prompts):
        result = client.images.generate(
            model="gpt-image-1.5",
            prompt=prompt,
            quality=quality,
            size="1024x1024",
            n=1
        )

        image_bytes = base64.b64decode(result.data[0].b64_json)
        filename = f"batch_output_{i + 1}.png"
        with open(filename, "wb") as f:
            f.write(image_bytes)

        results.append(filename)
        print(f"✅ Generated {filename}")

    return results


# =============================================================================
# 11. STREAMING GENERATION (for real-time preview)
# =============================================================================

def generate_with_streaming():
    """
    Generate image with streaming to get partial results.
    Useful for showing progress to users.
    """

    result = client.images.generate(
        model="gpt-image-1.5",
        prompt="A detailed fantasy castle on a mountain peak at sunset",
        quality="high",
        size="1024x1024",
        stream=True,  # Enable streaming
        partial_images=4,  # Number of partial images to receive
        n=1
    )

    # Handle streaming response
    for i, partial in enumerate(result):
        if partial.data and partial.data[0].b64_json:
            image_bytes = base64.b64decode(partial.data[0].b64_json)
            filename = f"streaming_partial_{i + 1}.png"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            print(f"✅ Partial image {i + 1} saved: {filename}")

    print("✅ Streaming generation complete")


# =============================================================================
# 12. HELPER FUNCTIONS
# =============================================================================

def save_and_resize(b64_data: str, output_path: str, max_size: tuple = None):
    """Decode base64 image, optionally resize, and save."""
    image_bytes = base64.b64decode(b64_data)
    image = Image.open(BytesIO(image_bytes))

    if max_size:
        image.thumbnail(max_size, Image.LANCZOS)

    # Determine format from extension
    ext = Path(output_path).suffix.lower()
    format_map = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.webp': 'WEBP'}
    img_format = format_map.get(ext, 'PNG')

    image.save(output_path, format=img_format, quality=85, optimize=True)
    print(f"✅ Saved to {output_path} ({image.size[0]}x{image.size[1]})")


def estimate_cost(quality: str, count: int = 1) -> float:
    """Estimate API cost for image generation."""
    prices = {
        "low": 0.01,
        "medium": 0.04,
        "high": 0.17
    }
    cost = prices.get(quality, 0.04) * count
    return cost


# =============================================================================
# MAIN - Run Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GPT Image 1.5 Python Examples")
    print("=" * 60)

    # Uncomment the examples you want to run:

    # 1. Basic generation
    generate_basic_image()

    # 2. Quality options
    # generate_with_quality_options()
    # generate_low_cost_image()

    # 3. Format options
    # generate_with_format_options()

    # 4. Transparent background
    # generate_transparent_background()

    # 5. Text in images
    # generate_image_with_text()

    # 6. Image editing (requires existing images)
    # edit_image_without_mask("input.png")

    # 7. Style transfer
    # style_transfer("photo.png", "a watercolor painting style")

    # 8. UI mockup
    # generate_ui_mockup()

    # 9. Batch generation
    # prompts = [
    #     "A red apple on a white background",
    #     "A green pear on a white background",
    #     "An orange on a white background"
    # ]
    # generate_batch(prompts, quality="low")

    # Cost estimation
    print("\n💰 Cost Estimates:")
    print(f"   Low quality:    ${estimate_cost('low'):.2f}/image")
    print(f"   Medium quality: ${estimate_cost('medium'):.2f}/image")
    print(f"   High quality:   ${estimate_cost('high'):.2f}/image")
    print(f"   100 medium images: ${estimate_cost('medium', 100):.2f}")

    print("\n✨ Uncomment examples in main() to run them!")
