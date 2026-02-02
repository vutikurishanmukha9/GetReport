import os
import time
import logging

logger = logging.getLogger(__name__)

def cleanup_old_files(directory: str, max_age_seconds: int = 86400):
    """
    Deletes files in the specified directory that are older than max_age_seconds.
    
    Args:
        directory: Path to the directory to clean.
        max_age_seconds: Max file age in seconds (default: 24h).
    """
    if not os.path.exists(directory):
        return

    now = time.time()
    count = 0
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            # Skip directories
            if not os.path.isfile(file_path):
                continue
                
            file_age = now - os.path.getmtime(file_path)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete old file {filename}: {e}")
        
        if count > 0:
            logger.info(f"Cleanup: Removed {count} old files (>{max_age_seconds}s) from {directory}.")
            
    except Exception as e:
        logger.error(f"Cleanup failed for {directory}: {e}")
