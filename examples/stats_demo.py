#!/usr/bin/env python3
"""
Example script demonstrating the new --stats performance feature in Docling CLI.

This script shows how the --stats flag provides detailed performance insights
for document conversion operations.
"""

import subprocess
import sys
from pathlib import Path


def run_docling_with_stats():
    """Demonstrate the --stats feature with example documents."""
    
    print("🚀 Docling CLI Performance Statistics Demo")
    print("=" * 50)
    print()
    
    # Example 1: Single document with stats
    print("📄 Example 1: Single Document Performance Analysis")
    print("-" * 40)
    
    cmd = [
        "docling", 
        "tests/data/pdf/2305.03393v1-pg9.pdf", 
        "--stats", 
        "--output", "/tmp/stats_demo_single"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Conversion completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n" + "=" * 50)
    print()
    
    # Example 2: Multiple documents with stats
    print("📄 Example 2: Batch Processing Performance Analysis")
    print("-" * 40)
    
    cmd = [
        "docling", 
        "tests/data/pdf/2305.03393v1-pg9.pdf",
        "tests/data/pdf/code_and_formula.pdf",
        "--stats", 
        "--output", "/tmp/stats_demo_batch"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Batch conversion completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n🎉 Demo completed! The --stats feature provides valuable insights into:")
    print("  • Overall conversion performance (throughput, timing)")
    print("  • Detailed pipeline operation breakdowns")
    print("  • Processing bottlenecks identification")
    print("  • Batch processing analytics")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("tests/data/pdf").exists():
        print("❌ Error: This script must be run from the Docling repository root directory")
        print("   Please run: cd /path/to/docling && python examples/stats_demo.py")
        sys.exit(1)
        
    run_docling_with_stats()