import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="caseify",
    version="0.0.1",
    author="Dennis Egan",
    author_email="d.james.egan@gmail.com",
    description="Module to perform case restoration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deneganisme/winter-camp-ling-cuny.git",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"caseify": "caseify"},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=required
)