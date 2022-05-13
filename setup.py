from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup (
    name = 'Smart-Home-Intent-Classification-&-Slot-Filling',
    version = '0.1',
    description = 'Intent Classification & Slot Filling',
    long_description = long_description,
    long_description_content_type="text/markdown", 
    author = 'Vu Quoc Hien',
    author_email = 'hien.vu@genestory.ai',
    url = 'https://github.com/NeiH4207/Smart-Home-Intent-Detection-and-Slot-Filling',
    packages = ["src", "models"],
    keywords = 'nlu, phobert',
    install_requires = [
        'torch==1.9.0',
        'transformers==4.1.1',
        'numpy==1.22.3',
        'pandas==1.4.2',
        'scikit-learn==0.24.2',
        'sklearn==0.0',
        'tqdm==4.62.2',
        'vncorenlp==1.0.3',
        'fastBPE==0.1.0',
        'matplotlib==3.4.3',
        'pytorch-crf==0.7.2',
        'scipy==1.8.0',
        'seqeval==0.0.12'
    ],
    python_requires = '>=3.8',
    entry_points = {
        'console_scripts': [
            'augment=augment_data:main',
            'filter=filter_data:main',
            'train=train:main',
            'inference=inference:main'
        ]
    },
    
)