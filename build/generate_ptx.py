#!/usr/bin/env python3

"""
Script to generate PTX files from CUTLASS GEMM kernels for SM80
Based on the original CUTLASS build configuration
"""

import os
import sys
import subprocess
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Tuple, Optional

# Configuration
BUILD_DIR = "/home/ubuntu/anjiang/PTX_dataset/cutlass_ptx/build"
SOURCE_DIR = "/home/ubuntu/anjiang/PTX_dataset/cutlass_ptx"
INPUT_DIR = f"{BUILD_DIR}/tools/library/generated/gemm"
OUTPUT_DIR = f"{BUILD_DIR}/tools/library/generated_ptx/gemm"
NVCC = "/usr/local/cuda/bin/nvcc"

# Compilation flags from the original build
CUDA_FLAGS = [
    "-DCUTLASS_VERSIONS_GENERATED",
    "-O3",
    "-DNDEBUG", 
    "-std=c++17",
    "--generate-code=arch=compute_80,code=sm_80",
    "-Xcompiler=-fPIC",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
    "--expt-relaxed-constexpr",
    "-ftemplate-backtrace-limit=0",
    "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
    "-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-Xcompiler=-Wconversion",
    "-Xcompiler=-fno-strict-aliasing"
]

# Include paths from the original build
INCLUDE_PATHS = [
    f"-I{SOURCE_DIR}/include",
    f"-I{BUILD_DIR}/include", 
    f"-I{SOURCE_DIR}/tools/library/include",
    f"-I{SOURCE_DIR}/tools/util/include",
    f"-I{SOURCE_DIR}/tools/library/src",
    "-isystem", "/usr/local/cuda/include"
]

def generate_ptx_single(cu_file: str, job_id: int = 0, verbose: bool = True) -> Tuple[bool, str, str]:
    """
    Generate PTX file for a single .cu file
    
    Args:
        cu_file: Path to the .cu file
        job_id: Job identifier for logging
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (success, relative_path, error_message)
    """
    relative_path = str(cu_file)  # Default fallback
    try:
        cu_path = Path(cu_file)
        
        # Try to get relative path, but handle case where cu_file is not under INPUT_DIR
        try:
            relative_path = str(cu_path.relative_to(INPUT_DIR))
        except ValueError:
            # File is not under INPUT_DIR, use relative to parent directory
            if len(cu_path.parts) >= 2:
                relative_path = "/".join(cu_path.parts[-2:])
            else:
                relative_path = cu_path.name
        
        # Create output file path
        output_file = Path(OUTPUT_DIR) / relative_path.replace('.cu', '.ptx')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"[{job_id:4d}] Processing: {relative_path}")
        
        # Build nvcc command
        cmd = [NVCC, "-ptx", str(cu_file)] + CUDA_FLAGS + INCLUDE_PATHS + ["-o", str(output_file)]
        
        # Run compilation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per file
        )
        
        if result.returncode == 0:
            if verbose:
                print(f"[{job_id:4d}] ✓ Success: {output_file.name}")
            return True, relative_path, ""
        else:
            error_msg = result.stderr.strip()
            if verbose:
                print(f"[{job_id:4d}] ✗ Failed: {relative_path}")
                if error_msg:
                    print(f"[{job_id:4d}]   Error: {error_msg[:100]}...")
            return False, relative_path, error_msg
            
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"[{job_id:4d}] ✗ Timeout: {relative_path}")
        return False, relative_path, "Compilation timeout"
    except Exception as e:
        if verbose:
            print(f"[{job_id:4d}] ✗ Exception: {relative_path} - {str(e)}")
        return False, relative_path, str(e)

def find_cu_files(directory: str) -> List[str]:
    """Find all .cu files in the given directory"""
    cu_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cu'):
                cu_files.append(os.path.join(root, file))
    return sorted(cu_files)

def process_directory_serial(directory: str, arch: str, verbose: bool = True) -> Tuple[int, int]:
    """Process files in a directory serially"""
    print(f"Processing SM{arch} kernels in {directory}...")
    
    cu_files = find_cu_files(directory)
    total_count = len(cu_files)
    
    if total_count == 0:
        print(f"No .cu files found in {directory}")
        return 0, 0
        
    print(f"Found {total_count} .cu files to process")
    
    success_count = 0
    failed_files = []
    
    for i, cu_file in enumerate(cu_files, 1):
        success, relative_path, error = generate_ptx_single(cu_file, i, verbose)
        if success:
            success_count += 1
        else:
            failed_files.append((relative_path, error))
            
        # Progress indicator
        if i % 100 == 0:
            print(f"Progress: {i}/{total_count} ({success_count} successful)")
    
    failed_count = total_count - success_count
    print(f"SM{arch} Summary: {success_count} successful, {failed_count} failed out of {total_count} files")
    
    if failed_files and verbose:
        print("Failed files:")
        for rel_path, error in failed_files[:10]:  # Show first 10 failures
            print(f"  {rel_path}: {error[:80]}...")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    return success_count, failed_count

def process_directory_parallel(directory: str, arch: str, max_workers: int = None, verbose: bool = True) -> Tuple[int, int]:
    """Process files in a directory in parallel"""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
        
    print(f"Processing SM{arch} kernels in {directory} with {max_workers} parallel jobs...")
    
    cu_files = find_cu_files(directory)
    total_count = len(cu_files)
    
    if total_count == 0:
        print(f"No .cu files found in {directory}")
        return 0, 0
        
    print(f"Found {total_count} .cu files to process")
    
    success_count = 0
    failed_files = []
    completed = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(generate_ptx_single, cu_file, i, False): (cu_file, i) 
            for i, cu_file in enumerate(cu_files, 1)
        }
        
        # Process completed jobs
        for future in as_completed(future_to_file):
            cu_file, job_id = future_to_file[future]
            completed += 1
            
            try:
                success, relative_path, error = future.result()
                if success:
                    success_count += 1
                    if verbose:
                        print(f"[{job_id:4d}] ✓ Success: {Path(relative_path).name}")
                else:
                    failed_files.append((relative_path, error))
                    if verbose:
                        print(f"[{job_id:4d}] ✗ Failed: {relative_path}")
                        
            except Exception as e:
                failed_files.append((str(cu_file), str(e)))
                if verbose:
                    print(f"[{job_id:4d}] ✗ Exception: {str(e)}")
            
            # Progress indicator
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total_count - completed) / rate if rate > 0 else 0
                print(f"Progress: {completed}/{total_count} ({success_count} successful) "
                      f"- {rate:.1f} files/sec - ETA: {eta/60:.1f}min")
    
    elapsed = time.time() - start_time
    failed_count = total_count - success_count
    print(f"SM{arch} Summary: {success_count} successful, {failed_count} failed out of {total_count} files")
    print(f"Total time: {elapsed/60:.1f} minutes ({total_count/elapsed:.1f} files/sec)")
    
    if failed_files and verbose:
        print("Failed files:")
        for rel_path, error in failed_files[:10]:  # Show first 10 failures
            print(f"  {rel_path}: {error[:80]}...")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    return success_count, failed_count

def get_cuda_flags_for_arch(arch: str) -> List[str]:
    """Get CUDA flags modified for specific architecture"""
    flags = CUDA_FLAGS.copy()
    # Replace compute_80,code=sm_80 with the target architecture
    for i, flag in enumerate(flags):
        if flag.startswith("--generate-code=arch=compute_80,code=sm_80"):
            flags[i] = f"--generate-code=arch=compute_{arch},code=sm_{arch}"
            break
    return flags

def main():
    parser = argparse.ArgumentParser(
        description="Generate PTX files from CUTLASS GEMM kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Serial processing
  %(prog)s -j                 # Parallel processing with default workers
  %(prog)s -j 8               # Parallel processing with 8 workers
  %(prog)s --arch 80          # Process only SM80 kernels
  %(prog)s --arch all         # Process all available architectures
        """
    )
    
    parser.add_argument('-j', '--parallel', nargs='?', const=True, type=int,
                       help='Enable parallel processing (optionally specify number of workers)')
    parser.add_argument('--arch', default='80', 
                       help='Target architecture (50,60,61,70,75,80,all) (default: 80)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without running compilation')
    
    args = parser.parse_args()
    
    # Determine parallelism
    if args.parallel is True:
        max_workers = multiprocessing.cpu_count()
        parallel = True
    elif isinstance(args.parallel, int):
        max_workers = args.parallel
        parallel = True
    else:
        max_workers = 1
        parallel = False
    
    # Print configuration
    print("=== CUTLASS PTX Generation Script ===")
    print(f"Source directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target architecture: SM{args.arch}")
    print(f"Parallel mode: {'ENABLED' if parallel else 'DISABLED'}")
    if parallel:
        print(f"Max workers: {max_workers}")
    print(f"NVCC: {NVCC}")
    print()
    
    # Check if nvcc is available
    if not os.path.exists(NVCC):
        print(f"Error: nvcc not found at {NVCC}")
        return 1
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return 1
        
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine architectures to process
    if args.arch == 'all':
        architectures = ['50', '60', '61', '70', '75', '80']
    else:
        architectures = [args.arch]
    
    total_success = 0
    total_failed = 0
    
    # Process each architecture
    for arch in architectures:
        arch_dir = os.path.join(INPUT_DIR, arch)
        if not os.path.exists(arch_dir):
            print(f"Warning: SM{arch} directory not found at {arch_dir}")
            continue
            
        print(f"Found SM{arch} kernels directory")
        
        if args.dry_run:
            cu_files = find_cu_files(arch_dir)
            print(f"Would process {len(cu_files)} .cu files from SM{arch}")
            continue
        
        # Update CUDA flags for this architecture if not SM80
        if arch != '80':
            global CUDA_FLAGS
            original_flags = CUDA_FLAGS
            CUDA_FLAGS = get_cuda_flags_for_arch(arch)
        
        try:
            if parallel:
                success, failed = process_directory_parallel(arch_dir, arch, max_workers, args.verbose)
            else:
                success, failed = process_directory_serial(arch_dir, arch, args.verbose)
                
            total_success += success
            total_failed += failed
            
        finally:
            # Restore original flags
            if arch != '80':
                CUDA_FLAGS = original_flags
        
        print()
    
    if not args.dry_run:
        print("=== PTX Generation Complete ===")
        print(f"Total: {total_success} successful, {total_failed} failed")
        print(f"Output files are in: {OUTPUT_DIR}")
        print()
        print("To verify generated PTX files:")
        print(f"  find {OUTPUT_DIR} -name '*.ptx' | wc -l")
        print()
        print("To check a specific PTX file:")
        print(f"  head -20 $(find {OUTPUT_DIR} -name '*.ptx' | head -1)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())