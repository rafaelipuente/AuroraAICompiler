#!/usr/bin/env python3
"""
Script to download model files for AuroraAICompiler

This script downloads pre-trained model files from various sources
and saves them in the correct format for use with the AuroraAICompiler.
"""

import argparse
import os
import sys
import urllib.request
import hashlib
import zipfile
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Root directory of the project
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Models directory
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Model definitions
MODELS = {
    'resnet50': {
        'url': 'https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx',
        'filename': 'resnet50.onnx',
        'sha256': '5ca5c6ocb36db41b238358af0d5cd6d0193a8bc89647df54077e48d7ee843788',  # Placeholder hash
        'description': 'ResNet-50 image classification model'
    },
    'mobilenet': {
        'url': 'https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx',
        'filename': 'mobilenet.onnx',
        'sha256': '8b4a2e1d1b22a36170bbfca87d0c2c3fec21b42375f0b7228bd7a626c21b9b9b',  # Placeholder hash
        'description': 'MobileNet v2 image classification model'
    },
    'bert': {
        'url': 'https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-8.onnx',
        'filename': 'bert.onnx',
        'sha256': '680e97b1e6de424cf89b273e7d59a7187310a0b534d00d247c8c7881f799a5ab',  # Placeholder hash
        'description': 'BERT model for question answering'
    }
}

def download_file(url: str, filename: str) -> bool:
    """
    Download a file from a URL and save it to the specified filename.
    
    Args:
        url: URL to download from
        filename: Destination filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url} to {filename}")
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Download with progress indicator
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = block_num * block_size * 100 / total_size
                sys.stdout.write(f"\r{percent:.1f}% downloaded")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filename, reporthook=report_progress)
        print()  # Newline after progress indicator
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def verify_file(filename: str, expected_hash: str) -> bool:
    """
    Verify a file's SHA-256 hash.
    
    Args:
        filename: File to verify
        expected_hash: Expected SHA-256 hash
        
    Returns:
        bool: True if hash matches, False otherwise
    """
    try:
        logger.info(f"Verifying {filename}")
        sha256 = hashlib.sha256()
        with open(filename, 'rb') as f:
            for block in iter(lambda: f.read(4096), b''):
                sha256.update(block)
        
        actual_hash = sha256.hexdigest()
        if actual_hash != expected_hash:
            logger.warning(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")
            logger.warning("The file may be corrupted or the expected hash may be incorrect")
            return False
        
        logger.info("Hash verification successful")
        return True
    except Exception as e:
        logger.error(f"Error verifying file: {e}")
        return False

def download_model(model_name: str) -> bool:
    """
    Download a model by name.
    
    Args:
        model_name: Name of the model to download
        
    Returns:
        bool: True if successful, False otherwise
    """
    if model_name not in MODELS:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_name]
    filename = os.path.join(MODELS_DIR, model_info['filename'])
    
    # Check if the file already exists
    if os.path.exists(filename):
        logger.info(f"{filename} already exists. Skipping download.")
        return True
    
    # Download the file
    if not download_file(model_info['url'], filename):
        return False
    
    # Verify the file (optional in this implementation)
    # verify_file(filename, model_info['sha256'])
    
    logger.info(f"Successfully downloaded {model_name} to {filename}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Download model files for AuroraAICompiler')
    parser.add_argument('models', nargs='*', help='Models to download')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--all', action='store_true', help='Download all models')
    
    args = parser.parse_args()
    
    # List available models
    if args.list:
        logger.info("Available models:")
        for name, info in MODELS.items():
            logger.info(f"  {name}: {info['description']}")
        return 0
    
    # Download all models
    if args.all:
        success = True
        for name in MODELS:
            if not download_model(name):
                success = False
        return 0 if success else 1
    
    # Download specified models
    if not args.models:
        logger.error("No models specified. Use --list to see available models.")
        return 1
    
    success = True
    for name in args.models:
        if not download_model(name):
            success = False
            
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
