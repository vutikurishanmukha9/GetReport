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
        for root, dirs, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    file_age = now - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to inspect or delete old file {file_path}: {e}")

        if count > 0:
            logger.info(f"Cleanup: Removed {count} old files (>{max_age_seconds}s) recursively from {directory}.")
            
    except Exception as e:
        logger.error(f"Cleanup failed for {directory}: {e}")
