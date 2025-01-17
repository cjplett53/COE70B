"""
Use this file to build, or add to, personnel database 
1. Add images as .png or something like it to personnel directory
    - You can add multiple images
2. Add path(s) to list variable image_paths
3. Call user defined generate_average_embedding_for_person with two parameters
    - Parameter 1: list of path(s) to images
    - Parameter 2: name of individual
"""

from functions import generate_average_embedding_for_person

# Example usage with multiple images for the same person
image_paths = ["Personnel/Person7/Person7.png"]
generate_average_embedding_for_person(image_paths, "Person7")

