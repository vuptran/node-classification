from setuptools import find_packages, setup


if __name__ == "__main__":
    setup(
        name="node_classification",
        version="0.0.1",
        license="MIT",
        description="Large-Scale Node Classification",
        author="vuptran",
        author_email="somewhere",
        packages=find_packages(include=("nodecls", "nodecls.*")),
        include_package_data=True,
        install_requires=[
            "numpy==2.2.6",
            "torch-geometric==2.6.1",
            "lightning==2.5.4",
            "ogb==1.3.6",
            "seaborn",
            "jupyter",
            "tensorboard",
            "scikit-learn",
            "pandas",
            "natsort",
            "einops",
            "pyyaml",
            "fastparquet",
            "networkx",
            "matplotlib",
        ]
    )
