import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

setuptools.setup(
    name="mcp_insurance",
    version=__version__,
    author="k17hawk",
    author_email="kumardahal536@gmail.com",
    description="A project for test",
    long_description=long_description,
    long_description_content="text/markdown",
    url="https://github.com/k17hawk/mcp_insurance",
    project_urls={
        "Bug Tracker": "https://github.com/k17hawk/mcp_insurance/issues",
    },
    packages=setuptools.find_packages(),   # looks in current directory
)