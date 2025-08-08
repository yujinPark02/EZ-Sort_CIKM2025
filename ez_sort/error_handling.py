
# error_handling.py
"""
Error handling utilities for EZ-Sort
"""

import functools
import traceback
import logging
from typing import Callable, Any


logger = logging.getLogger("ez_sort")


class EZSortError(Exception):
    """Base exception for EZ-Sort"""
    pass


class DatasetError(EZSortError):
    """Error in dataset loading or processing"""
    pass


class CLIPError(EZSortError):
    """Error in CLIP model processing"""
    pass


class AnnotationError(EZSortError):
    """Error in annotation process"""
    pass


def handle_errors(fallback_return=None, reraise=True):
    """
    Decorator for error handling in EZ-Sort functions
    
    Args:
        fallback_return: Value to return if error occurs and reraise=False
        reraise: Whether to reraise the exception after logging
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                else:
                    return fallback_return
        
        return wrapper
    return decorator


def validate_dataset_format(csv_path: str, image_dir: str, 
                          image_col: str, label_col: str) -> None:
    """Validate dataset format and raise informative errors"""
    
    import os
    import pandas as pd
    
    # Check CSV file
    if not os.path.exists(csv_path):
        raise DatasetError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise DatasetError(f"Failed to read CSV file {csv_path}: {e}")
    
    # Check required columns
    if image_col not in df.columns:
        raise DatasetError(f"Image column '{image_col}' not found in CSV. Available columns: {list(df.columns)}")
    
    if label_col not in df.columns:
        raise DatasetError(f"Label column '{label_col}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Check image directory
    if not os.path.exists(image_dir):
        raise DatasetError(f"Image directory not found: {image_dir}")
    
    # Check if images exist
    missing_images = []
    for idx, img_path in enumerate(df[image_col].iloc[:5]):  # Check first 5
        full_path = os.path.join(image_dir, img_path)
        if not os.path.exists(full_path):
            missing_images.append(img_path)
    
    if missing_images:
        raise DatasetError(f"Missing images in {image_dir}: {missing_images[:3]}{'...' if len(missing_images) > 3 else ''}")
    
    logger.info(f"✅ Dataset validation passed: {len(df)} items in {csv_path}")

