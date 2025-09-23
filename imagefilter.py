import os
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from collections import defaultdict
import concurrent.futures
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import List, Tuple, Dict, Set, Optional
import gc
import multiprocessing as mp

class ImageFilter:
    def __init__(self, similarity_threshold=0.95, solid_color_threshold=0.98, max_workers=None, 
                 min_resolution=None, max_resolution=None, hash_size=8):
        """
        Initialize the image filter optimized for large datasets.
        """
        self.similarity_threshold = similarity_threshold
        self.solid_color_threshold = solid_color_threshold
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.hash_size = hash_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
        
        # Pre-calculate thresholds for performance
        self.max_hash_distance = int(self.hash_size * self.hash_size * (1 - self.similarity_threshold))
        self.solid_variance_threshold = 255 * 255 * (1 - self.solid_color_threshold)
    
    def _get_image_info_fast(self, image_path: Path) -> Tuple[Optional[Tuple[int, int]], bool, Optional[int]]:
        """
        Ultra-fast image processing using integer hash instead of string.
        Returns: (resolution, is_solid_color, hash_int)
        """
        try:
            with Image.open(image_path) as img:
                resolution = img.size
                
                # Quick resolution check
                if not self._is_resolution_valid(resolution):
                    return resolution, False, None
                
                # Convert to grayscale and resize for processing
                gray_img = img.convert('L').resize((self.hash_size + 1, self.hash_size), Image.Resampling.NEAREST)
                pixels = np.array(gray_img, dtype=np.uint8)
                
                # Check if solid color using variance
                variance = np.var(pixels)
                is_solid = variance < self.solid_variance_threshold
                
                if is_solid:
                    return resolution, True, None
                
                # Calculate difference hash as integer (much faster)
                diff = pixels[:, 1:] > pixels[:, :-1]
                hash_bits = diff.flatten()
                
                # Convert to integer hash (faster than string operations)
                hash_int = 0
                for i, bit in enumerate(hash_bits):
                    if bit:
                        hash_int |= (1 << i)
                
                return resolution, False, hash_int
                
        except Exception:
            return None, False, None
    
    def _is_resolution_valid(self, resolution: Tuple[int, int]) -> bool:
        """Check if resolution is within specified range."""
        if not resolution:
            return False
            
        width, height = resolution
        
        if self.min_resolution:
            min_w, min_h = self.min_resolution
            if width < min_w or height < min_h:
                return False
        
        if self.max_resolution:
            max_w, max_h = self.max_resolution
            if width > max_w or height > max_h:
                return False
        
        return True
    
    def _hamming_distance_int(self, hash1: int, hash2: int) -> int:
        """Calculate Hamming distance between two integer hashes (much faster)."""
        return bin(hash1 ^ hash2).count('1')
    
    def _find_duplicates_ultra_fast(self, hash_data: List[Tuple[str, int]]) -> List[List[str]]:
        """
        Ultra-fast duplicate detection using LSH (Locality Sensitive Hashing) approach.
        """
        if not hash_data:
            return []
        
        print(f"Finding duplicates among {len(hash_data):,} valid images...")
        start_time = time.time()
        
        # Create hash lookup for exact matches (instant)
        exact_matches = defaultdict(list)
        for path, hash_val in hash_data:
            exact_matches[hash_val].append(path)
        
        duplicate_groups = []
        processed_hashes = set()
        
        # Process exact matches first
        for hash_val, paths in exact_matches.items():
            if len(paths) > 1:
                duplicate_groups.append(paths)
                processed_hashes.add(hash_val)
                print(f"Found exact match group: {len(paths)} images")
        
        # For near-matches, use LSH buckets for O(n) performance
        remaining = [(path, hash_val) for path, hash_val in hash_data 
                    if hash_val not in processed_hashes]
        
        if len(remaining) > 1:
            print(f"Checking {len(remaining):,} images for near-matches...")
            
            # Create LSH buckets using hash prefixes
            bucket_size = max(4, self.hash_size - 2)  # Allow some bit differences
            buckets = defaultdict(list)
            
            for path, hash_val in remaining:
                # Create multiple bucket keys by masking different bits
                for shift in range(0, self.hash_size * self.hash_size, bucket_size):
                    bucket_key = (hash_val >> shift) & ((1 << bucket_size) - 1)
                    buckets[bucket_key].append((path, hash_val))
            
            # Check only items in same buckets
            processed_paths = set()
            
            for bucket_items in buckets.values():
                if len(bucket_items) < 2:
                    continue
                
                # Quick check within bucket
                for i, (path1, hash1) in enumerate(bucket_items):
                    if path1 in processed_paths:
                        continue
                    
                    similar_group = [path1]
                    processed_paths.add(path1)
                    
                    for j, (path2, hash2) in enumerate(bucket_items[i+1:], i+1):
                        if path2 in processed_paths:
                            continue
                        
                        if self._hamming_distance_int(hash1, hash2) <= self.max_hash_distance:
                            similar_group.append(path2)
                            processed_paths.add(path2)
                    
                    if len(similar_group) > 1:
                        duplicate_groups.append(similar_group)
        
        elapsed = time.time() - start_time
        print(f"Duplicate detection completed in {elapsed:.1f}s")
        return duplicate_groups
    
    def _process_batch_ultra_fast(self, image_paths: List[Path]) -> Dict[str, List]:
        """Process a batch of images with maximum speed."""
        results = {
            'solid_color': [],
            'resolution_filtered': [],
            'valid_hashes': [],
            'errors': []
        }
        
        for path in image_paths:
            try:
                resolution, is_solid, hash_val = self._get_image_info_fast(path)
                
                if resolution is None:
                    results['errors'].append(str(path))
                elif not self._is_resolution_valid(resolution):
                    results['resolution_filtered'].append(str(path))
                elif is_solid:
                    results['solid_color'].append(str(path))
                elif hash_val is not None:
                    results['valid_hashes'].append((str(path), hash_val))
                else:
                    results['errors'].append(str(path))
                    
            except Exception:
                results['errors'].append(str(path))
        
        return results
    
    def process_images_ultra_fast(self, image_paths: List[Path], progress_callback=None) -> Dict[str, List]:
        """Process images with maximum parallel efficiency."""
        total_images = len(image_paths)
        # Optimize batch size for maximum throughput
        batch_size = min(200, max(20, total_images // (self.max_workers * 4)))
        
        results = {
            'solid_color': [],
            'resolution_filtered': [],
            'valid_hashes': [],
            'errors': []
        }
        
        processed_count = 0
        start_time = time.time()
        
        def log_progress():
            nonlocal processed_count
            if processed_count % 2000 == 0 or processed_count == total_images:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta = (total_images - processed_count) / rate if rate > 0 else 0
                
                message = (f"Processed: {processed_count:,}/{total_images:,} "
                          f"({processed_count/total_images*100:.1f}%) - "
                          f"Rate: {rate:.0f} img/sec - ETA: {eta:.0f}s")
                
                if progress_callback:
                    progress_callback(message)
                print(message)
        
        # Use ProcessPoolExecutor for CPU-intensive tasks when dataset is large
        executor_class = concurrent.futures.ProcessPoolExecutor if total_images > 10000 else concurrent.futures.ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all batches
            futures = []
            for i in range(0, total_images, batch_size):
                batch = image_paths[i:i + batch_size]
                future = executor.submit(self._process_batch_ultra_fast, batch)
                futures.append((future, len(batch)))
            
            # Collect results as they complete
            for future, batch_len in futures:
                try:
                    batch_results = future.result()
                    
                    # Merge results
                    for key in results:
                        results[key].extend(batch_results[key])
                    
                    processed_count += batch_len
                    log_progress()
                    
                except Exception as e:
                    print(f"Batch processing error: {e}")
                    processed_count += batch_len
        
        elapsed = time.time() - start_time
        rate = total_images / elapsed if elapsed > 0 else 0
        if progress_callback:
            progress_callback(f"Processing completed in {elapsed:.1f}s - Rate: {rate:.0f} img/sec")
        
        return results
    
    def get_image_files_fast(self, folder_path: str) -> List[Path]:
        """Get all image files with optimized scanning."""
        folder = Path(folder_path)
        image_files = []
        
        print("Scanning for image files...")
        start_time = time.time()
        
        # Use more efficient scanning
        try:
            # Try to use os.walk for better performance on large directories
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.supported_formats):
                        image_files.append(Path(root) / file)
                        
                        if len(image_files) % 10000 == 0:
                            print(f"Found {len(image_files):,} image files...")
        except:
            # Fallback to rglob if os.walk fails
            for file_path in folder.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    image_files.append(file_path)
        
        elapsed = time.time() - start_time
        print(f"Scan completed: {len(image_files):,} image files found in {elapsed:.1f}s")
        return image_files
    
    def filter_images(self, folder_path: str, output_folder: str = None, 
                     move_filtered: bool = False, progress_callback=None) -> List[str]:
        """
        Main filtering method - ultra-optimized for large datasets.
        """
        start_time = time.time()
        
        def log(message):
            if progress_callback:
                progress_callback(message)
            print(message)
        
        # Get all image files
        log("Starting ultra-fast image filtering...")
        image_files = self.get_image_files_fast(folder_path)
        
        if not image_files:
            log("No image files found!")
            return []
        
        log(f"Processing {len(image_files):,} images with {self.max_workers} workers...")
        
        # Process all images
        results = self.process_images_ultra_fast(image_files, progress_callback)
        
        # Find duplicate groups with optimized algorithm
        duplicate_groups = self._find_duplicates_ultra_fast(results['valid_hashes'])
        
        # Collect filtered images (keep first in each group)
        filtered_images = set(results['solid_color'] + results['resolution_filtered'])
        
        total_duplicates = 0
        for i, group in enumerate(duplicate_groups):
            log(f"Duplicate group {i+1}: {len(group)} similar images")
            # Keep first image, filter the rest
            for img_path in group[1:]:
                filtered_images.add(img_path)
                total_duplicates += 1
        
        # Move filtered images if requested
        if move_filtered and output_folder and filtered_images:
            self._move_filtered_images(filtered_images, output_folder, log)
        
        # Final summary
        elapsed = time.time() - start_time
        rate = len(image_files) / elapsed if elapsed > 0 else 0
        log(f"\n{'='*60}")
        log(f"FILTERING COMPLETE")
        log(f"{'='*60}")
        log(f"Total time: {elapsed:.1f}s")
        log(f"Total images: {len(image_files):,}")
        log(f"Solid color: {len(results['solid_color']):,}")
        log(f"Resolution filtered: {len(results['resolution_filtered']):,}")
        log(f"Duplicate groups: {len(duplicate_groups):,}")
        log(f"Duplicates removed: {total_duplicates:,}")
        log(f"Total filtered: {len(filtered_images):,}")
        log(f"Remaining: {len(image_files) - len(filtered_images):,}")
        log(f"Processing rate: {rate:.0f} images/second")
        
        return list(filtered_images)
    
    def _move_filtered_images(self, filtered_images: Set[str], output_folder: str, log_func):
        """Move filtered images to output folder with parallel processing."""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        log_func(f"Moving {len(filtered_images):,} filtered images...")
        moved_count = 0
        errors = 0
        
        def move_single_file(img_path_str):
            try:
                img_path = Path(img_path_str)
                dest_path = output_path / img_path.name
                
                # Handle name conflicts
                counter = 1
                while dest_path.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    dest_path = output_path / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                img_path.rename(dest_path)
                return True
            except Exception:
                return False
        
        # Move files in parallel for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, self.max_workers)) as executor:
            futures = [executor.submit(move_single_file, img_path) 
                      for img_path in filtered_images]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if future.result():
                    moved_count += 1
                else:
                    errors += 1
                
                if (i + 1) % 1000 == 0:
                    log_func(f"Moved {moved_count:,}/{len(filtered_images):,} files...")
        
        log_func(f"Successfully moved {moved_count:,} files ({errors} errors)")


class ImageFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter Tool - Ultra Performance")
        self.root.geometry("900x750")
        
        # Variables with optimized defaults for performance
        self.folder_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.similarity_threshold = tk.DoubleVar(value=0.95)
        self.solid_threshold = tk.DoubleVar(value=0.98)
        self.max_workers = tk.IntVar(value=min(32, (os.cpu_count() or 1) * 2))
        self.hash_size = tk.IntVar(value=8)  # Optimized for speed
        self.move_files = tk.BooleanVar(value=False)
        self.filter_by_resolution = tk.BooleanVar(value=False)
        self.min_width = tk.IntVar(value=100)
        self.min_height = tk.IntVar(value=100)
        self.max_width = tk.IntVar(value=4000)
        self.max_height = tk.IntVar(value=4000)
        
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input/Output folders
        ttk.Label(main_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.folder_path, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input_folder).grid(row=0, column=2, padx=5)
        
        ttk.Label(main_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_path, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_folder).grid(row=1, column=2, padx=5)
        
        # Performance options
        perf_frame = ttk.LabelFrame(main_frame, text="Performance Settings (Ultra Mode)", padding="10")
        perf_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(perf_frame, text="CPU Workers:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(perf_frame, from_=1, to=64, textvariable=self.max_workers, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(perf_frame, text="Hash Size:").grid(row=0, column=2, sticky=tk.W, padx=(20,5))
        ttk.Spinbox(perf_frame, from_=4, to=16, textvariable=self.hash_size, width=10).grid(row=0, column=3, padx=5)
        
        # Add performance info
        perf_info = ttk.Label(perf_frame, text=f"Optimized for {os.cpu_count() or 1} CPU cores", 
                             foreground="blue")
        perf_info.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=5)
        
        # Filter options
        filter_frame = ttk.LabelFrame(main_frame, text="Filter Settings", padding="10")
        filter_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(filter_frame, text="Similarity Threshold:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(filter_frame, from_=0.8, to=1.0, variable=self.similarity_threshold, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5)
        ttk.Label(filter_frame, textvariable=self.similarity_threshold).grid(row=0, column=2)
        
        ttk.Label(filter_frame, text="Solid Color Threshold:").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(filter_frame, from_=0.8, to=1.0, variable=self.solid_threshold, 
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5)
        ttk.Label(filter_frame, textvariable=self.solid_threshold).grid(row=1, column=2)
        
        # Resolution filtering
        resolution_frame = ttk.LabelFrame(main_frame, text="Resolution Filtering", padding="10")
        resolution_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Checkbutton(resolution_frame, text="Enable Resolution Filtering", 
                       variable=self.filter_by_resolution).grid(row=0, column=0, columnspan=4, sticky=tk.W)
        
        ttk.Label(resolution_frame, text="Min:").grid(row=1, column=0, sticky=tk.W, padx=(20, 5))
        ttk.Spinbox(resolution_frame, from_=1, to=10000, textvariable=self.min_width, width=8).grid(row=1, column=1, padx=2)
        ttk.Label(resolution_frame, text="x").grid(row=1, column=2)
        ttk.Spinbox(resolution_frame, from_=1, to=10000, textvariable=self.min_height, width=8).grid(row=1, column=3, padx=2)
        
        ttk.Label(resolution_frame, text="Max:").grid(row=1, column=4, sticky=tk.W, padx=5)
        ttk.Spinbox(resolution_frame, from_=1, to=10000, textvariable=self.max_width, width=8).grid(row=1, column=5, padx=2)
        ttk.Label(resolution_frame, text="x").grid(row=1, column=6)
        ttk.Spinbox(resolution_frame, from_=1, to=10000, textvariable=self.max_height, width=8).grid(row=1, column=7, padx=2)
        
        # Options
        ttk.Checkbutton(main_frame, text="Move filtered files to output folder", 
                       variable=self.move_files).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Start Ultra-Fast Filtering", command=self.start_filtering).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        
        # Results
        ttk.Label(main_frame, text="Results:").grid(row=8, column=0, sticky=tk.W, pady=(10, 0))
        self.results_text = scrolledtext.ScrolledText(main_frame, height=15, width=90)
        self.results_text.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(9, weight=1)
    
    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.folder_path.set(folder)
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path.set(folder)
    
    def log_message(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        self.results_text.delete(1.0, tk.END)
    
    def start_filtering(self):
        if not self.folder_path.get():
            messagebox.showerror("Error", "Please select an input folder")
            return
        
        if self.move_files.get() and not self.output_path.get():
            messagebox.showerror("Error", "Please select an output folder when moving files")
            return
        
        self.progress.start()
        
        # Prepare settings
        min_res = None
        max_res = None
        if self.filter_by_resolution.get():
            min_res = (self.min_width.get(), self.min_height.get())
            max_res = (self.max_width.get(), self.max_height.get())
        
        filter_instance = ImageFilter(
            similarity_threshold=self.similarity_threshold.get(),
            solid_color_threshold=self.solid_threshold.get(),
            max_workers=self.max_workers.get(),
            min_resolution=min_res,
            max_resolution=max_res,
            hash_size=self.hash_size.get()
        )
        
        def run_filter():
            try:
                self.log_message("Starting ultra-fast image filtering...")
                filtered = filter_instance.filter_images(
                    self.folder_path.get(),
                    self.output_path.get() if self.move_files.get() else None,
                    self.move_files.get(),
                    progress_callback=self.log_message
                )
                self.log_message(f"\nFiltering completed! {len(filtered):,} images were filtered.")
            except Exception as e:
                self.log_message(f"Error during filtering: {str(e)}")
            finally:
                self.progress.stop()
        
        threading.Thread(target=run_filter, daemon=True).start()


def create_gui():
    """Create and run the GUI application."""
    root = tk.Tk()
    app = ImageFilterGUI(root)
    root.mainloop()

def main():
    if len(os.sys.argv) == 1:
        create_gui()
        return
    
    # Command line interface
    parser = argparse.ArgumentParser(description='Ultra-fast image filter for large datasets')
    parser.add_argument('folder', help='Folder containing images to filter')
    parser.add_argument('--output', '-o', help='Output folder for filtered images')
    parser.add_argument('--move', '-m', action='store_true', 
                       help='Move filtered images to output folder')
    parser.add_argument('--similarity', '-s', type=float, default=0.95,
                       help='Similarity threshold (0-1, default: 0.95)')
    parser.add_argument('--solid-threshold', '-t', type=float, default=0.98,
                       help='Solid color threshold (0-1, default: 0.98)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (auto-detected if not specified)')
    parser.add_argument('--hash-size', type=int, default=8,
                       help='Hash size for similarity detection (4-16, default: 8)')
    parser.add_argument('--min-resolution', type=str, 
                       help='Minimum resolution as WIDTHxHEIGHT')
    parser.add_argument('--max-resolution', type=str,
                       help='Maximum resolution as WIDTHxHEIGHT')
    
    args = parser.parse_args()
    
    if args.move and not args.output:
        print("Error: --output is required when using --move")
        return
    
    # Parse resolution arguments
    min_res = None
    max_res = None
    if args.min_resolution:
        try:
            w, h = map(int, args.min_resolution.split('x'))
            min_res = (w, h)
        except ValueError:
            print("Error: Invalid min-resolution format. Use WIDTHxHEIGHT")
            return
    
    if args.max_resolution:
        try:
            w, h = map(int, args.max_resolution.split('x'))
            max_res = (w, h)
        except ValueError:
            print("Error: Invalid max-resolution format. Use WIDTHxHEIGHT")
            return
    
    filter_instance = ImageFilter(
        similarity_threshold=args.similarity,
        solid_color_threshold=args.solid_threshold,
        max_workers=args.workers,
        min_resolution=min_res,
        max_resolution=max_res,
        hash_size=args.hash_size
    )
    
    filter_instance.filter_images(
        args.folder, 
        args.output, 
        args.move
    )

if __name__ == "__main__":
    main()