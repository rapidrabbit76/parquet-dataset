from setuptools import setup, find_packages

version = "0.0.1.dev4"

project_urls = {
    "Source": "https://github.com/rapidrabbit76/parquet-dataset",
}
install_requires = [
    "pandas>=1.4.1",
    "pyarrow>=7.0.0",
    "torch>=1.6",
]

setup(
    name="parquet-dataset",
    version=version,
    description="Dataset for parquet group",
    author="yslee",
    author_email="yslee.dev@gmail.com",
    maintainer="yslee",
    maintainer_email="yslee.dev@gamil.com",
    license="MIT",
    keywords="parquet pytorch dataset",
    project_urls=project_urls,
    python_requires=">=3.6",
    install_requires=install_requires,
    packages=find_packages(),
)
