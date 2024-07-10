import pdfplumber
import os
from PIL import Image


def pdf_to_markdown(pdf_path, markdown_path):
    with pdfplumber.open(pdf_path) as pdf:
        markdown_content = ""
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            markdown_content += text + "\n\n"

            # Extract images
            for i, img in enumerate(page.images):
                img_path = f"{os.path.splitext(pdf_path)[0]}_page{page.page_number}_img{i}.png"
                img_obj = page.to_image(resolution=150)
                img_obj.save(img_path)
                markdown_content += f"![Image {i}]({img_path})\n\n"

    # Write markdown content to file
    with open(markdown_path, "w") as md_file:
        md_file.write(markdown_content)


# Example usage
pdf_file = "../data/inputs/1011Physics.pdf"
markdown_file = "../data/outputs/question.md"
pdf_to_markdown(pdf_file, markdown_file)
