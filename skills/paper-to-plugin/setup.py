from setuptools import setup, find_packages

setup(
    name="paper-to-plugin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "Jinja2",
        "requests",
        # PyMuPDF
        "PyMuPDF"
    ],
    entry_points={
        'console_scripts': [
            'paper-to-plugin=skills.paper_to_plugin.cli.main:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A skill to analyze research papers and generate plugin code.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
