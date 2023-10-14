import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

def organize_images_by_xml(base_dir: str, xml_dir: str) -> None:
    """
    Organize validation images into class-specific folders based on XML annotations.

    Parameters:
    - base_dir: Directory containing validation images.
    - xml_dir: Directory containing XML annotation files for validation images.
    """
    # List all XML files in the xml_dir
    xml_files = list(Path(xml_dir).glob('*.xml'))

    for xml_file in xml_files:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract filename and WordNet ID
        filename = root.find('filename').text + ".JPEG"
        wn_id = root.find('object').find('name').text

        # Ensure the class-specific folder exists
        class_folder = Path(base_dir) / wn_id
        class_folder.mkdir(parents=True, exist_ok=True)

        # Move the image to the class-specific folder
        image_path = Path(base_dir) / filename
        target_path = class_folder / filename
        
        if image_path.exists():  # Ensure the image exists before moving
            # print(f"Moving {image_path} to {target_path}")
            shutil.move(str(image_path), str(target_path))

    print(f"Organized {len(xml_files)} validation images into class-specific folders based on XML annotations.")

base_dir = "/project/3011213.01/imagenet/ILSVRC/Data/CLS-LOC/val"
xml_dir = "/project/3011213.01/imagenet/ILSVRC/Annotations/CLS-LOC/val"
organize_images_by_xml(base_dir, xml_dir)
