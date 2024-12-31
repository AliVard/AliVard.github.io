# gpt generated code to add toc to markdown files
# Usage: python files/toc.py

import re
import sys

def generate_toc(markdown_file):
    with open(markdown_file, 'r') as file:
        lines = file.readlines()

    toc = ['# Table of Contents', '']
    commented = False
    for line in lines:
        if line.startswith("<!--"):
            commented = True
            continue
        if line.endswith("-->"):
            commented = False
            continue
        if commented:
            continue

        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))  # Number of '#' defines the level
            title = match.group(2).strip()

            # Extract text inside [] if inline links are present
            link_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)

            # Generate a valid slug
            link = re.sub(r'[^\w\s-]', '', link_text).strip().lower().replace(' ', '-')
            toc.append(f"{'  ' * (level - 2)}- [{link_text}](#{link})")


    print('\n'.join(toc))


generate_toc(sys.argv[1])