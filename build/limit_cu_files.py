#!/usr/bin/env python3

"""
Script to limit .cu files in a directory to 500 files
Sorts files by filename and keeps only the first 500, removes the rest
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

def find_cu_files(directory: Path) -> List[Path]:
    """Find all .cu files in the given directory recursively"""
    cu_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cu'):
                cu_files.append(Path(root) / file)
    return cu_files

def main():
    parser = argparse.ArgumentParser(
        description="Limit .cu files in a directory to 500 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Process tools/library/generated/gemm (default)
  %(prog)s /path/to/directory                 # Process specific directory
  %(prog)s --dry-run                         # Show what would be removed without doing it
  %(prog)s --limit 300                       # Keep only first 300 files instead of 500
        """
    )
    
    parser.add_argument('directory', nargs='?', 
                       help='Directory to process (default: tools/library/generated/gemm)')
    parser.add_argument('--limit', type=int, default=500,
                       help='Maximum number of .cu files to keep (default: 500)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be removed without actually removing files')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output showing all operations')
    
    args = parser.parse_args()
    
    # Determine directory to process
    if args.directory:
        target_dir = Path(args.directory).absolute()
    else:
        # Default to the generated/gemm directory relative to script location
        script_dir = Path(__file__).parent.absolute()
        target_dir = script_dir / "tools/library/generated/gemm"
    
    print(f"=== CU File Limiter ===")
    print(f"Target directory: {target_dir}")
    print(f"File limit: {args.limit}")
    print(f"Dry run: {'YES' if args.dry_run else 'NO'}")
    print()
    
    # Check if directory exists
    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}")
        return 1
        
    if not target_dir.is_dir():
        print(f"Error: Path is not a directory: {target_dir}")
        return 1
    
    # Find all .cu files
    print("Scanning for .cu files...")
    cu_files = find_cu_files(target_dir)
    total_files = len(cu_files)
    
    print(f"Found {total_files} .cu files")
    
    if total_files <= args.limit:
        print(f"Directory has {total_files} files, which is <= {args.limit} limit.")
        print("No files need to be removed.")
        return 0
    
    # Sort files by filename (not full path)
    cu_files.sort(key=lambda x: x.name)
    
    # Split into keep and remove lists
    files_to_keep = cu_files[:args.limit]
    files_to_remove = cu_files[args.limit:]
    
    print(f"Will keep first {len(files_to_keep)} files (sorted by filename)")
    print(f"Will remove {len(files_to_remove)} files")
    print()
    
    if args.verbose:
        print("Files to keep:")
        for i, file_path in enumerate(files_to_keep[:10], 1):  # Show first 10
            rel_path = file_path.relative_to(target_dir)
            print(f"  {i:3d}: {rel_path}")
        if len(files_to_keep) > 10:
            print(f"  ... and {len(files_to_keep) - 10} more")
        print()
        
        print("Files to remove:")
        for i, file_path in enumerate(files_to_remove[:10], 1):  # Show first 10
            rel_path = file_path.relative_to(target_dir)
            print(f"  {i:3d}: {rel_path}")
        if len(files_to_remove) > 10:
            print(f"  ... and {len(files_to_remove) - 10} more")
        print()
    
    if args.dry_run:
        print("DRY RUN - No files were actually removed")
        print(f"Would remove {len(files_to_remove)} files")
        return 0
    
    # Confirm deletion
    print(f"About to permanently delete {len(files_to_remove)} .cu files")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("Operation cancelled")
        return 0
    
    # Remove files
    print("Removing files...")
    removed_count = 0
    failed_removals = []
    
    for file_path in files_to_remove:
        try:
            file_path.unlink()  # Remove the file
            removed_count += 1
            
            if args.verbose:
                rel_path = file_path.relative_to(target_dir)
                print(f"  Removed: {rel_path}")
            elif removed_count % 100 == 0:
                print(f"  Progress: {removed_count}/{len(files_to_remove)} files removed")
                
        except Exception as e:
            failed_removals.append((file_path, str(e)))
            if args.verbose:
                rel_path = file_path.relative_to(target_dir)
                print(f"  Failed to remove: {rel_path} - {e}")
    
    # Clean up empty directories
    print("Cleaning up empty directories...")
    empty_dirs_removed = 0
    
    # Walk bottom-up to remove empty directories
    for root, dirs, files in os.walk(target_dir, topdown=False):
        root_path = Path(root)
        if root_path != target_dir:  # Don't remove the target directory itself
            try:
                # Check if directory is empty
                if not any(root_path.iterdir()):
                    root_path.rmdir()
                    empty_dirs_removed += 1
                    if args.verbose:
                        rel_path = root_path.relative_to(target_dir)
                        print(f"  Removed empty directory: {rel_path}")
            except Exception as e:
                if args.verbose:
                    print(f"  Could not remove directory {root_path}: {e}")
    
    # Final summary
    print()
    print("=== Operation Complete ===")
    print(f"Files removed: {removed_count}")
    print(f"Files remaining: {total_files - removed_count}")
    print(f"Empty directories removed: {empty_dirs_removed}")
    
    if failed_removals:
        print(f"Failed to remove {len(failed_removals)} files:")
        for file_path, error in failed_removals[:5]:  # Show first 5 failures
            rel_path = file_path.relative_to(target_dir)
            print(f"  {rel_path}: {error}")
        if len(failed_removals) > 5:
            print(f"  ... and {len(failed_removals) - 5} more failures")
    
    # Verify final count
    remaining_files = find_cu_files(target_dir)
    print(f"Verification: {len(remaining_files)} .cu files now in directory")
    
    if len(remaining_files) <= args.limit:
        print("✓ Success: Directory now has <= limit files")
    else:
        print("⚠ Warning: Directory still has more files than expected")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())