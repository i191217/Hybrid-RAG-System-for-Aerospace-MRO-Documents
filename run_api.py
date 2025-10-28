#!/usr/bin/env python3
"""
Script to run the Harfang Document Processing API.
"""

import argparse
import uvicorn
from core.config import config

def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description="Harfang Document Processing API Server")
    parser.add_argument("--host", default=config.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.API_PORT, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=config.API_WORKERS, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", default=config.API_RELOAD, help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    print(f"Starting Harfang Document Processing API...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print(f"Log Level: {args.log_level}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Alternative Docs: http://{args.host}:{args.port}/redoc")
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,  # Can't use workers with reload
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main() 