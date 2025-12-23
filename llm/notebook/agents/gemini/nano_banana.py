"""
AI Agent Icon Generator Backend
Generates light and dark mode icons for AI agents using Gemini's image generation.
"""
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# Configuration
IMAGE_SIZE = 320  # Total image size
ICON_SIZE = 160   # Actual icon circle size (centered, with effects around it)
MODEL = "gemini-2.5-flash-image"  # Model with image generation capability

# Paths
SCRIPT_DIR = Path(__file__).parent
EXAMPLES_DIR = SCRIPT_DIR / "agent-icons"
OUTPUT_DIR = SCRIPT_DIR / "generated-icons"

# System instruction for icon generation (flexible style)
SYSTEM_INSTRUCTION = f"""You are a creative AI agent icon designer. Create unique, expressive avatar icons.

CORE STRUCTURE (must follow):
1. Image dimensions: {IMAGE_SIZE}x{IMAGE_SIZE} pixels
2. Centered circular icon of approximately {ICON_SIZE}x{ICON_SIZE} pixels
3. The icon should have glow effects, reflections, or ambient lighting around it
4. Futuristic, premium feel with 3D depth

CREATIVE FREEDOM (adapt based on user input):
- Colors: Use colors that match the user's concept (e.g., "pink house" = pink tones, "colorful car" = vibrant rainbow)
- Internal design: Can include abstract patterns, symbols, silhouettes, or stylized representations of the concept
- The orb can have tinted glass, colored glow, or themed internal elements
- Express the personality and essence of the user's concept through color and form

DARK MODE:
- Dark/black background
- Glowing accent lighting (color matches the concept)
- Luminous internal glow

LIGHT MODE:
- Light/white background
- Soft glass reflections
- Gentle shadows underneath

Be creative! The icon should clearly evoke the user's concept while maintaining a polished, app-icon quality."""


def load_example_image(filename: str) -> Optional[types.Part]:
    """Load a single example image as a Part."""
    filepath = EXAMPLES_DIR / filename
    if filepath.exists():
        with open(filepath, "rb") as f:
            image_data = f.read()
        return types.Part.from_bytes(data=image_data, mime_type="image/png")
    return None


def generate_agent_icon(user_input: str, mode: str = "dark") -> Optional[bytes]:
    """
    Generate an AI agent icon based on user input.

    Args:
        user_input: Description of the agent (e.g., "smart robot", "data analyst")
        mode: "dark" or "light" for the color scheme

    Returns:
        Image bytes if successful, None otherwise
    """
    # Select appropriate example based on mode
    if mode == "dark":
        example_file = "dard-mode-custom-agent-1.png"
    else:
        example_file = "light-mode-default-agents.png"

    example_image = load_example_image(example_file)

    # Build the prompt
    mode_desc = "DARK MODE" if mode == "dark" else "LIGHT MODE"
    background = (
        "dark/black background with glowing cyan/blue effects"
        if mode == "dark"
        else "light/white background with soft glass reflections"
    )

    prompt = f"""{SYSTEM_INSTRUCTION}

---

Create a {mode_desc} AI agent icon for: "{user_input}"

Requirements:
- Size: {IMAGE_SIZE}x{IMAGE_SIZE} pixels with centered {ICON_SIZE}x{ICON_SIZE} icon
- {background}
- Use the reference image for structural inspiration (circular orb with effects)
- BUT be creative with colors and internal design to match "{user_input}"
- If the concept has specific colors (like "pink house"), use those colors prominently
- The internal design should visually represent or symbolize "{user_input}"

Create a unique, expressive icon that captures the essence of "{user_input}"!"""

    # Build content: example image + prompt
    contents: list = []
    if example_image:
        contents.append("Reference style example:")
        contents.append(example_image)
    contents.append(prompt)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            )
        )

        # Extract image from response
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
                elif part.text:
                    print(f"Model response: {part.text}")

    except Exception as e:
        print(f"Error generating {mode} mode icon: {e}")

    return None


def resize_image(image_data: bytes, target_size: int = IMAGE_SIZE) -> bytes:
    """Resize image to target dimensions."""
    import io
    img = Image.open(io.BytesIO(image_data))
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def generate_agent_icons(user_input: str, output_name: Optional[str] = None) -> dict:
    """
    Generate both light and dark mode icons for an AI agent.

    Args:
        user_input: Description of the agent
        output_name: Base name for output files (optional, derived from input if not provided)

    Returns:
        Dictionary with paths to generated icons
    """
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate output filename from input if not provided
    if output_name is None:
        output_name = user_input.lower().replace(" ", "_")[:30]

    results = {}

    # Generate dark mode icon
    print(f"Generating dark mode icon for: {user_input}")
    dark_icon = generate_agent_icon(user_input, mode="dark")
    if dark_icon:
        # Ensure correct size
        dark_icon = resize_image(dark_icon)
        dark_path = OUTPUT_DIR / f"{output_name}_dark.png"
        with open(dark_path, "wb") as f:
            f.write(dark_icon)
        results["dark"] = str(dark_path)
        print(f"Dark mode icon saved to: {dark_path}")

    # Generate light mode icon
    print(f"Generating light mode icon for: {user_input}")
    light_icon = generate_agent_icon(user_input, mode="light")
    if light_icon:
        # Ensure correct size
        light_icon = resize_image(light_icon)
        light_path = OUTPUT_DIR / f"{output_name}_light.png"
        with open(light_path, "wb") as f:
            f.write(light_icon)
        results["light"] = str(light_path)
        print(f"Light mode icon saved to: {light_path}")

    return results


# Simple API-like interface for backend integration
class AgentIconGenerator:
    """Backend service for generating AI agent icons."""

    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)

    def generate(self, agent_description: str, agent_id: Optional[str] = None) -> dict:
        """
        Generate icons for an agent.

        Args:
            agent_description: What the agent does/represents
            agent_id: Unique identifier for the agent (for file naming)

        Returns:
            {
                "success": bool,
                "dark_icon_path": str or None,
                "light_icon_path": str or None,
                "error": str or None
            }
        """
        try:
            icon_results = generate_agent_icons(agent_description, agent_id)
            return {
                "success": len(icon_results) > 0,
                "dark_icon_path": icon_results.get("dark"),
                "light_icon_path": icon_results.get("light"),
                "error": None if icon_results else "Failed to generate icons"
            }
        except Exception as e:
            return {
                "success": False,
                "dark_icon_path": None,
                "light_icon_path": None,
                "error": str(e)
            }


# CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("Enter agent description (e.g., 'smart robot'): ").strip()
        if not user_input:
            user_input = "smart robot"

    print(f"\nGenerating AI agent icons for: '{user_input}'")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, Icon circle: {ICON_SIZE}x{ICON_SIZE}")
    print("-" * 50)

    results = generate_agent_icons(user_input)

    print("-" * 50)
    if results:
        print("Generation complete!")
        for mode, path in results.items():
            print(f"  {mode.capitalize()} mode: {path}")
    else:
        print("Failed to generate icons.")
