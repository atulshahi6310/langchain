#!/usr/bin/env python
# coding: utf-8

# ### how to split by html header

from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Basic HTML Profile</title>
</head>
<body>

    <!-- Main Heading -->
    <h1>Jane Doe</h1>

    <!-- Subheading and Paragraph -->
    <h2>About Me</h2>
    <p>Welcome to my sample webpage. I am a web developer based in Gujarat, India, and I enjoy building clean and responsive layouts.</p>

    <!-- Unordered List with data -->
    <h3>Quick Facts</h3>
    <h3>Contact Information</h3>
    <h3>Education</h3>
    <ul>
        <li><strong>Role:</strong> Frontend Developer</li>
        <li><strong>Location:</strong> Gujarat, India</li>
        <li><strong>Favorite Tech:</strong> HTML, CSS, JavaScript</li>
    </ul>

    <!-- Link -->
    <p>Check out more web resources on <a href="https://developer.mozilla.org" target="_blank">MDN Web Docs</a>.</p>

</body>
</html>

"""

header_to_split = [
    ("h1" ,"Header 1"),
    ("h2" ,"Header 2"),
    ("h3" ,"Header 3"),
    
]

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

html_header_splits = html_splitter.split_text(html_string)

print(html_header_splits)

html_header_splits

import requests
url = "https://plato.stanford.edu/entries/goedel-incompleteness/sup2.html"
html_content = requests.get(url).text
print(html_content)
header_to_split = [
    ("h1" ,"Header 1"),
    ("h2" ,"Header 2"),
    ("h3" ,"Header 3"),
    # ("li" ,"List Item")
]

html_splitter = HTMLHeaderTextSplitter(
       headers_to_split_on = header_to_split,
         

    )

html_header_splits = html_splitter.split_text(html_content)

html_header_splits
