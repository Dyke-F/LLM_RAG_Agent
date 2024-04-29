"""Simple logger init file to import in all modules that shall be logged"""
import os
from loguru import logger

if not os.path.exists("LOGS"):
    os.makedirs("LOGS")

logger.add(
    "LOGS/preprocessing.log", level="DEBUG", rotation="500 MB", backtrace=True, diagnose=True
)
