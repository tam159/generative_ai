"""
GPT Image 1.5 - HTTP Request Examples (No SDK)
===============================================
Using requests library directly with OpenAI API endpoints.
No openai library required - just standard HTTP requests.

Requirements:
    pip install requests pillow

Set your API key:
    export OPENAI_API_KEY="your-api-key"
"""

import base64
import os
import json
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv

import requests
from PIL import Image

load_dotenv()
# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://api.openai.com/v1"

# SSL certificate for corporate proxy (Netskope)
SSL_CERT_PATH = os.environ.get("SSL_CERT_FILE", "~/netskope.pem")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Create a session with SSL verification configured
session = requests.Session()
session.headers.update(HEADERS)
# session.verify = SSL_CERT_PATH


# =============================================================================
# 1. BASIC IMAGE GENERATION
# =============================================================================

def generate_basic_image():
    """Generate a simple image using POST request."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": "A cute baby sea otter floating on its back in calm blue water",
        "n": 1,
        "size": "1024x1024"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    image_b64 = data["data"][0]["b64_json"]

    # Decode and save
    image_bytes = base64.b64decode(image_b64)
    with open("basic_http_output.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Basic image saved to basic_http_output.png")
    return data


# =============================================================================
# 2. FULL PARAMETERS EXAMPLE
# =============================================================================

def generate_with_all_options():
    """Generate image with all available parameters."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": "A professional product photo of a sleek smartwatch on a marble surface",
        "n": 1,
        "size": "1536x1024",  # Landscape
        "quality": "high",  # low, medium, high
        "response_format": "b64_json",  # b64_json or url
        "output_format": "png",  # png, jpeg, webp
        # "output_compression": 80,     # 0-100 (jpeg/webp only)
        # "background": "transparent",  # transparent or opaque (png only)
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    image_b64 = data["data"][0]["b64_json"]

    image_bytes = base64.b64decode(image_b64)
    with open("full_options_output.png", "wb") as f:
        f.write(image_bytes)

    print("✅ Full options image saved")
    print(f"   Revised prompt: {data['data'][0].get('revised_prompt', 'N/A')}")
    return data


# =============================================================================
# 3. DIFFERENT QUALITY LEVELS
# =============================================================================

def generate_low_quality():
    """Low quality - fastest and cheapest (~$0.01/image)."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": "A simple red apple icon",
        "n": 1,
        "size": "1024x1024",
        "quality": "low"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    save_b64_image(data["data"][0]["b64_json"], "low_quality.png")
    print("✅ Low quality image saved (~$0.01)")
    return data


def generate_medium_quality():
    """Medium quality - balanced (~$0.04/image)."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": "A detailed illustration of a coffee shop interior",
        "n": 1,
        "size": "1024x1024",
        "quality": "medium"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    save_b64_image(data["data"][0]["b64_json"], "medium_quality.png")
    print("✅ Medium quality image saved (~$0.04)")
    return data


def generate_high_quality():
    """High quality - best results (~$0.17/image)."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": "A photorealistic portrait of an astronaut on Mars, cinematic lighting",
        "n": 1,
        "size": "1024x1024",
        "quality": "high"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    save_b64_image(data["data"][0]["b64_json"], "high_quality.png")
    print("✅ High quality image saved (~$0.17)")
    return data


# =============================================================================
# 4. DIFFERENT OUTPUT FORMATS
# =============================================================================

def generate_jpeg_compressed():
    """Generate compressed JPEG output."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": "A landscape photo of mountains at sunset",
        "n": 1,
        "size": "1536x1024",
        "quality": "medium",
        "output_format": "jpeg",
        "output_compression": 75  # 75% quality
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    save_b64_image(data["data"][0]["b64_json"], "compressed.jpg")
    print("✅ Compressed JPEG saved")
    return data


def generate_webp():
    """Generate WebP format (smaller file size)."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": "Abstract colorful geometric pattern",
        "n": 1,
        "size": "1024x1024",
        "quality": "medium",
        "output_format": "webp",
        "output_compression": 80
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    save_b64_image(data["data"][0]["b64_json"], "output.webp")
    print("✅ WebP image saved")
    return data


def generate_transparent_png():
    """Generate PNG with transparent background."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": "A 3D logo of a golden eagle, isolated, no background",
        "n": 1,
        "size": "1024x1024",
        "quality": "high",
        "output_format": "png",
        "background": "transparent"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    save_b64_image(data["data"][0]["b64_json"], "transparent.png")
    print("✅ Transparent PNG saved")
    return data


# =============================================================================
# 5. IMAGE EDITING (INPAINTING)
# =============================================================================

def edit_image(image_path: str, prompt: str, mask_path: str = None):
    """
    Edit an existing image with optional mask.
    Uses multipart/form-data for file uploads.
    """

    url = f"{BASE_URL}/images/edits"

    # Headers without Content-Type (requests sets it for multipart)
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Prepare files
    files = {
        "image": ("image.png", open(image_path, "rb"), "image/png"),
    }

    if mask_path:
        files["mask"] = ("mask.png", open(mask_path, "rb"), "image/png")

    # Form data
    data = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": "1",
        "size": "1024x1024"
    }

    response = requests.post(url, headers=headers, files=files, data=data, verify=SSL_CERT_PATH)
    response.raise_for_status()

    result = response.json()
    save_b64_image(result["data"][0]["b64_json"], "edited_output.png")
    print("✅ Edited image saved")
    return result


def edit_image_base64(image_path: str, prompt: str):
    """
    Edit image using JSON payload with base64-encoded image.
    Alternative to multipart form upload.
    """

    url = f"{BASE_URL}/images/edits"

    # Read and encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "gpt-image-1.5",
        "image": image_b64,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    result = response.json()
    save_b64_image(result["data"][0]["b64_json"], "edited_b64_output.png")
    print("✅ Edited image (base64 method) saved")
    return result


# =============================================================================
# 6. MULTIPLE REFERENCE IMAGES
# =============================================================================

def generate_from_multiple_images(image_paths: list, prompt: str):
    """
    Use multiple reference images (up to 10) to generate new image.
    Uses multipart/form-data.
    """

    url = f"{BASE_URL}/images/edits"

    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Multiple images use the same field name "image"
    files = []
    for i, path in enumerate(image_paths):
        files.append(("image", (f"image_{i}.png", open(path, "rb"), "image/png")))

    data = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": "1",
        "size": "1024x1024"
    }

    response = requests.post(url, headers=headers, files=files, data=data, verify=SSL_CERT_PATH)
    response.raise_for_status()

    result = response.json()
    save_b64_image(result["data"][0]["b64_json"], "multi_ref_output.png")
    print("✅ Multi-reference image saved")
    return result


# =============================================================================
# 7. TEXT RENDERING IN IMAGES
# =============================================================================

def generate_text_heavy_image():
    """GPT Image 1.5 excels at rendering text."""

    url = f"{BASE_URL}/images/generations"

    prompt = """
    Create a professional event poster:

    Title: "TECH SUMMIT 2025"
    Subtitle: "Innovation Meets Tomorrow"
    Date: "March 15-17, 2025"
    Location: "San Francisco Convention Center"
    Website: "www.techsummit2025.com"

    Modern design with dark blue gradient background,
    white and gold text, geometric accents.
    Clean typography, high contrast, readable text.
    """

    payload = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1536",  # Portrait
        "quality": "high"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    save_b64_image(data["data"][0]["b64_json"], "text_poster.png")
    print("✅ Text-heavy poster saved")
    return data


# =============================================================================
# 8. UI MOCKUP GENERATION
# =============================================================================

def generate_ui_mockup():
    """Generate mobile app UI mockup."""

    url = f"{BASE_URL}/images/generations"

    prompt = """
    Create a mobile banking app UI mockup:

    Screen 1 - Dashboard:
    - Header: "Welcome, John" with profile icon
    - Balance card: "$12,458.32" in large text
    - Quick actions: Send, Request, Pay Bills, More
    - Recent transactions list (3-4 items)
    - Bottom nav: Home, Cards, Transfer, Settings

    Modern fintech design, iOS style.
    Dark mode with blue accent color.
    Place in iPhone 15 Pro frame.
    """

    payload = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1536",  # Portrait for mobile
        "quality": "high"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    save_b64_image(data["data"][0]["b64_json"], "ui_mockup.png")
    print("✅ UI mockup saved")
    return data


# =============================================================================
# 9. BATCH GENERATION (Multiple prompts)
# =============================================================================

def generate_batch(prompts: list, quality: str = "medium"):
    """Generate multiple images from different prompts."""

    url = f"{BASE_URL}/images/generations"
    results = []

    for i, prompt in enumerate(prompts):
        payload = {
            "model": "gpt-image-1.5",
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": quality
        }

        response = session.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        filename = f"batch_{i + 1}.png"
        save_b64_image(data["data"][0]["b64_json"], filename)
        results.append(filename)
        print(f"✅ Generated {filename}")

    return results


# =============================================================================
# 10. GENERATE MULTIPLE VARIATIONS (n > 1)
# =============================================================================

def generate_variations(prompt: str, count: int = 4):
    """Generate multiple variations of the same prompt."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": min(count, 10),  # Max 10
        "size": "1024x1024",
        "quality": "medium"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()

    for i, item in enumerate(data["data"]):
        filename = f"variation_{i + 1}.png"
        save_b64_image(item["b64_json"], filename)
        print(f"✅ Saved {filename}")

    return data


# =============================================================================
# 11. STREAMING GENERATION
# =============================================================================

def generate_with_streaming():
    """
    Generate image with streaming to receive partial results.
    Uses Server-Sent Events (SSE).
    """

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": "A detailed fantasy castle on a mountain peak at sunset",
        "n": 1,
        "size": "1024x1024",
        "quality": "high",
        "stream": True,
        "partial_images": 4  # Number of progressive previews
    }

    # Stream response
    response = session.post(
        url,
        json=payload,
        stream=True
    )
    response.raise_for_status()

    partial_count = 0
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                json_str = line[6:]  # Remove "data: " prefix
                if json_str == "[DONE]":
                    break

                data = json.loads(json_str)
                if data.get("data") and data["data"][0].get("b64_json"):
                    partial_count += 1
                    filename = f"stream_partial_{partial_count}.png"
                    save_b64_image(data["data"][0]["b64_json"], filename)
                    print(f"✅ Partial {partial_count} saved: {filename}")

    print("✅ Streaming complete")


# =============================================================================
# 12. ERROR HANDLING
# =============================================================================

def generate_with_error_handling(prompt: str):
    """Robust generation with proper error handling."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "quality": "medium"
    }

    try:
        response = session.post(url, json=payload, timeout=120)

        # Check for HTTP errors
        if response.status_code == 429:
            print("❌ Rate limited. Wait and retry.")
            retry_after = response.headers.get("Retry-After", "60")
            print(f"   Retry after: {retry_after} seconds")
            return None

        if response.status_code == 400:
            error = response.json().get("error", {})
            print(f"❌ Bad request: {error.get('message', 'Unknown error')}")
            return None

        if response.status_code == 401:
            print("❌ Invalid API key")
            return None

        response.raise_for_status()

        data = response.json()
        save_b64_image(data["data"][0]["b64_json"], "safe_output.png")
        print("✅ Image generated successfully")
        return data

    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


# =============================================================================
# 13. USING SESSION FOR MULTIPLE REQUESTS
# =============================================================================

def create_session():
    """Create a reusable session with headers preset."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def generate_with_session(session: requests.Session, prompt: str, filename: str):
    """Generate image using a pre-configured session."""

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "quality": "medium"
    }

    response = session.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    save_b64_image(data["data"][0]["b64_json"], filename)
    print(f"✅ Saved {filename}")
    return data


# =============================================================================
# 14. ASYNC GENERATION (using asyncio + aiohttp)
# =============================================================================

async def generate_async(prompt: str, filename: str):
    """
    Async image generation using aiohttp.
    Requires: pip install aiohttp
    """
    import ssl
    import aiohttp

    url = f"{BASE_URL}/images/generations"

    payload = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "quality": "medium"
    }

    # Create SSL context with custom certificate
    ssl_context = ssl.create_default_context(cafile=SSL_CERT_PATH)
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as aio_session:
        async with aio_session.post(url, headers=HEADERS, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            save_b64_image(data["data"][0]["b64_json"], filename)
            print(f"✅ Async saved {filename}")
            return data


async def generate_batch_async(prompts: list):
    """Generate multiple images concurrently."""
    import ssl
    import asyncio
    import aiohttp

    url = f"{BASE_URL}/images/generations"

    async def generate_one(aio_session, prompt, index):
        payload = {
            "model": "gpt-image-1.5",
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "low"  # Use low for batch to save costs
        }
        async with aio_session.post(url, headers=HEADERS, json=payload) as response:
            data = await response.json()
            filename = f"async_batch_{index}.png"
            save_b64_image(data["data"][0]["b64_json"], filename)
            return filename

    # Create SSL context with custom certificate
    ssl_context = ssl.create_default_context(cafile=SSL_CERT_PATH)
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as aio_session:
        tasks = [generate_one(aio_session, p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        print(f"✅ Generated {len(results)} images concurrently")
        return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_b64_image(b64_data: str, filename: str):
    """Decode base64 and save to file."""
    image_bytes = base64.b64decode(b64_data)
    with open(filename, "wb") as f:
        f.write(image_bytes)


def save_and_resize(b64_data: str, filename: str, max_size: tuple = None):
    """Decode, optionally resize, and save image."""
    image_bytes = base64.b64decode(b64_data)
    image = Image.open(BytesIO(image_bytes))

    if max_size:
        image.thumbnail(max_size, Image.LANCZOS)

    ext = Path(filename).suffix.lower()
    fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}.get(ext[1:], "PNG")
    image.save(filename, format=fmt, quality=85, optimize=True)


def estimate_cost(quality: str, count: int = 1) -> float:
    """Estimate API cost."""
    prices = {"low": 0.01, "medium": 0.04, "high": 0.17}
    return prices.get(quality, 0.04) * count


# =============================================================================
# CURL COMMAND EXAMPLES (for reference)
# =============================================================================

CURL_EXAMPLES = """
# Basic generation
curl https://api.openai.com/v1/images/generations \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -d '{
    "model": "gpt-image-1.5",
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024"
  }'

# High quality with options
curl https://api.openai.com/v1/images/generations \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -d '{
    "model": "gpt-image-1.5",
    "prompt": "Professional product photo of headphones",
    "n": 1,
    "size": "1536x1024",
    "quality": "high",
    "output_format": "png"
  }'

# Image editing with mask
curl https://api.openai.com/v1/images/edits \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -F model="gpt-image-1.5" \\
  -F image="@input.png" \\
  -F mask="@mask.png" \\
  -F prompt="Replace with sunset sky" \\
  -F n=1 \\
  -F size="1024x1024"
"""

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GPT Image 1.5 - HTTP Request Examples")
    print("=" * 60)

    if not API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Run: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    # Uncomment examples to run:

    # 1. Basic generation
    # generate_basic_image()

    # 2. Full options
    # generate_with_all_options()

    # 3. Quality levels
    generate_low_quality()
    # generate_medium_quality()
    # generate_high_quality()

    # 4. Format options
    # generate_jpeg_compressed()
    # generate_webp()
    # generate_transparent_png()

    # 5. Image editing (requires input.png)
    # edit_image("input.png", "Add a sunset background")

    # 6. Text rendering
    # generate_text_heavy_image()

    # 7. UI mockup
    # generate_ui_mockup()

    # 8. Batch generation
    # prompts = ["A red apple", "A green pear", "An orange"]
    # generate_batch(prompts, quality="low")

    # 9. Multiple variations
    # generate_variations("A fantasy dragon", count=4)

    # 10. With error handling
    # generate_with_error_handling("A peaceful garden")

    # 11. Using session (efficient for multiple requests)
    # session = create_session()
    # generate_with_session(session, "A mountain landscape", "session_output.png")
    # session.close()

    # 12. Async batch (requires aiohttp)
    # import asyncio
    # prompts = ["A cat", "A dog", "A bird"]
    # asyncio.run(generate_batch_async(prompts))

    # Cost estimation
    print("\n💰 Cost Estimates:")
    print(f"   Low quality:    ${estimate_cost('low'):.2f}/image")
    print(f"   Medium quality: ${estimate_cost('medium'):.2f}/image")
    print(f"   High quality:   ${estimate_cost('high'):.2f}/image")

    print("\n📖 See CURL_EXAMPLES variable for command-line examples")
    print("\n✨ Uncomment examples in main() to run them!")
