# Prutor-ai-Final-project
This repo contains the final project to be submitted to prutor ai IIT kanpur

How to convert jupyter notebook into pdf
To use this bundler you need to install it:

python -m pip install -U notebook-as-pdf
pyppeteer-install

The second command will download and setup Chromium. It is used to perform the HTML to PDF conversion.

On linux you probably also need to install some or all of the APT packages listed in binder/apt.txt.
Use it

Create a notebook and the click "File -> Download As". Click the new menu entry called "PDF via HTML". Your notebook will be converted to a PDF on the fly and then downloaded.

You can also use it with nbconvert:

jupyter-nbconvert --to pdfviahtml example.ipynb
