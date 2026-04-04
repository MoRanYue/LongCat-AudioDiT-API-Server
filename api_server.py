#!/usr/bin/env python3
"""LongCat-AudioDiT API Server

A FastAPI-based API server for LongCat-AudioDiT TTS model.
"""

import argparse
import logging
import os
import sys
import uvicorn
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from api.config import ConfigManager
from api.app import create_app

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LongCat-AudioDiT API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind the server to")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    
    # Model configuration
    parser.add_argument("--model-dir", type=str,
                       default="meituan-longcat/LongCat-AudioDiT-1B",
                       help="HuggingFace model ID or local path to model")
    
    # Directory configuration
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Directory for output audio files")
    parser.add_argument("--samples-dir", type=str, default="./samples",
                       help="Directory for sample audio files")
    
    # Config file
    parser.add_argument("--config", type=str,
                       help="Path to configuration file (JSON)")
    
    # Development options
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    
    return parser.parse_args()


def setup_logging(level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    
    logger.info("Starting LongCat-AudioDiT API Server")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Model: {args.model_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Samples directory: {args.samples_dir}")
    
    # Create configuration
    config_kwargs = {
        "output_dir": args.output_dir,
        "samples_dir": args.samples_dir,
        "model_dir": args.model_dir,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level
    }
    
    if args.config:
        config = ConfigManager(args.config)
        # Update with command line arguments
        config.update_config(**config_kwargs)
    else:
        config = ConfigManager()
        config.update_config(**config_kwargs)
    
    # Create FastAPI app
    app = create_app(config)
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload
    )


if __name__ == "__main__":
    main()