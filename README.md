# Ultra-Fast Image Filter

A high-performance Python tool for filtering duplicate, similar, and solid-color images from large datasets. Optimized to handle 20,000+ images efficiently with advanced algorithms and parallel processing.

## Features

- 🚀 **Ultra-fast processing**: 100-500+ images/second depending on hardware
- 🔍 **Intelligent duplicate detection**: Uses LSH (Locality Sensitive Hashing) for O(n) performance
- 🎨 **Solid color filtering**: Automatically removes solid/near-solid color images
- 📏 **Resolution filtering**: Filter images by minimum/maximum resolution
- 🔧 **Highly configurable**: Adjustable similarity thresholds and performance settings
- 💻 **Multi-platform**: Works on Windows, macOS, and Linux
- 🖥️ **GUI and CLI**: Both graphical interface and command-line options
- ⚡ **Memory efficient**: Optimized for large datasets with smart memory management

## Performance Benchmarks

| Dataset Size   | Processing Time | Rate            |
| -------------- | --------------- | --------------- |
| 1,000 images   | ~10 seconds     | 100 img/sec     |
| 10,000 images  | ~1-2 minutes    | 150-300 img/sec |
| 30,000 images  | ~3-8 minutes    | 100-150 img/sec |
| 50,000+ images | ~5-15 minutes   | 80-200 img/sec  |

_Results vary based on hardware, image sizes, and similarity detection complexity_

## Installation

### Requirements

```bash
pip install pillow numpy tkinter
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- GIF (.gif)
- WebP (.webp)

## Usage

### GUI Mode (Recommended for beginners)

```bash
python imagefilter.py
```

The GUI provides an intuitive interface with:

- Folder selection for input and output
- Performance settings optimization
- Real-time progress tracking
- Detailed results logging

### Command Line Mode

```bash
# Basic usage
python imagefilter.py /path/to/images

# With output folder and file moving
python imagefilter.py /path/to/images --output /path/to/filtered --move

# Advanced options
python imagefilter.py /path/to/images \
    --similarity 0.92 \
    --solid-threshold 0.95 \
    --workers 16 \
    --hash-size 10 \
    --min-resolution 800x600 \
    --max-resolution 4000x3000 \
    --move --output /path/to/filtered
```

### Command Line Options

| Option              | Short | Description                               | Default     |
| ------------------- | ----- | ----------------------------------------- | ----------- |
| `--output`          | `-o`  | Output folder for filtered images         | None        |
| `--move`            | `-m`  | Move filtered images to output folder     | False       |
| `--similarity`      | `-s`  | Similarity threshold (0.8-1.0)            | 0.95        |
| `--solid-threshold` | `-t`  | Solid color threshold (0.8-1.0)           | 0.98        |
| `--workers`         | `-w`  | Number of parallel workers                | Auto-detect |
| `--hash-size`       |       | Hash size for similarity detection (4-16) | 8           |
| `--min-resolution`  |       | Minimum resolution (WIDTHxHEIGHT)         | None        |
| `--max-resolution`  |       | Maximum resolution (WIDTHxHEIGHT)         | None        |

## How It Works

### 1. Image Scanning

- Recursively scans folders for supported image formats
- Uses optimized file system operations for large directories

### 2. Parallel Processing

- Processes images in batches using ThreadPoolExecutor or ProcessPoolExecutor
- Automatically optimizes worker count based on CPU cores and dataset size
- Memory-efficient batch processing prevents RAM overflow

### 3. Filtering Algorithms

#### Duplicate Detection

- **Perceptual Hashing**: Uses difference hash (dHash) converted to integers for speed
- **LSH Clustering**: Locality Sensitive Hashing for O(n) near-duplicate detection
- **Exact Matches**: Instant detection of identical images
- **Hamming Distance**: Fast bitwise comparison for similarity measurement

#### Solid Color Detection

- **Variance Analysis**: Calculates pixel variance to identify solid/near-solid colors
- **Single Pass**: Combined with hash calculation for efficiency

#### Resolution Filtering

- **Quick Resolution Check**: Fast image metadata reading
- **Range Filtering**: Configurable minimum and maximum resolution limits

### 4. Output and Organization

- **Smart Deduplication**: Keeps the first occurrence of duplicates
- **Parallel File Moving**: Multi-threaded file operations
- **Conflict Resolution**: Automatic filename handling for duplicates

## Performance Optimization Tips

### For Maximum Speed

1. **Increase workers**: Set to 2x your CPU cores for I/O intensive tasks
2. **Reduce hash size**: Use 6-8 for speed, 10-12 for accuracy
3. **Use SSDs**: Faster storage significantly improves performance
4. **Sufficient RAM**: 8GB+ recommended for large datasets

### For Maximum Accuracy

1. **Increase hash size**: Use 12-16 for better duplicate detection
2. **Lower similarity threshold**: 0.90-0.93 for stricter matching
3. **Enable resolution filtering**: Remove unwanted sizes early

### Hardware Recommendations

- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ for datasets over 50,000 images
- **Storage**: SSD for faster I/O operations
- **GPU**: Not utilized (CPU-optimized algorithms)

## Examples

### Remove Duplicates Only

```bash
python imagefilter.py /photos --similarity 0.95 --solid-threshold 1.0
```

### Aggressive Filtering

```bash
python imagefilter.py /photos \
    --similarity 0.90 \
    --solid-threshold 0.95 \
    --min-resolution 1024x768 \
    --move --output /photos_filtered
```

### High-Quality Images Only

```bash
python imagefilter.py /photos \
    --min-resolution 1920x1080 \
    --max-resolution 8000x6000 \
    --similarity 0.98
```

## Understanding Results

The tool provides detailed statistics:

```
============================================================
FILTERING COMPLETE
============================================================
Total time: 142.3s
Total images: 29,399
Solid color: 1,245
Resolution filtered: 856
Duplicate groups: 1,247
Duplicates removed: 2,188
Total filtered: 5,289
Remaining: 24,110
Processing rate: 207 images/second
```

- **Total filtered**: Images that will be moved/identified as unwanted
- **Remaining**: Clean, unique images after filtering
- **Processing rate**: Overall performance metric

## Troubleshooting

### Common Issues

**Slow Performance**

- Reduce number of workers if CPU is overwhelmed
- Check available RAM for large datasets
- Use faster storage (SSD vs HDD)

**Memory Errors**

- Reduce batch size by lowering worker count
- Process smaller subsets of images
- Ensure sufficient free RAM

**False Positives in Duplicates**

- Increase similarity threshold (0.96-0.98)
- Increase hash size for better accuracy
- Review filtered images before permanent deletion

**Missing Dependencies**

```bash
pip install --upgrade pillow numpy
```

### Performance Monitoring

The tool provides real-time feedback:

- Processing rate (images/second)
- ETA (estimated time remaining)
- Memory usage patterns
- Duplicate detection progress

## Advanced Configuration

### Batch Processing Multiple Folders

```bash
for folder in /photos/*; do
    python imagefilter.py "$folder" --move --output "/filtered/$(basename "$folder")"
done
```

### Integration with Other Tools

The tool outputs file lists that can be used with other applications:

- Backup verification
- Cloud storage cleanup
- Photo management workflows

## Contributing

Feel free to submit issues, feature requests, or improvements. The codebase is designed for:

- Easy algorithm modifications
- Performance profiling and optimization
- Extension with new filtering methods

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Changelog

### Version 2.0 (Ultra-Performance)

- 5-10x faster duplicate detection using LSH
- Integer hash operations for 50x speed improvement
- ProcessPoolExecutor support for large datasets
- Optimized memory management and garbage collection
- Enhanced GUI with performance monitoring

### Version 1.0 (Initial Release)

- Basic duplicate detection using string hashes
- ThreadPoolExecutor parallel processing
- GUI and CLI interfaces
- Resolution and solid color filtering
