# Realistic DVS Noise Enhancement for v2ref.py

## Overview
Your v2ref.py has been enhanced with realistic DVS camera noise models that go far beyond basic threshold-based event generation. These improvements address the "synthetic" appearance by incorporating real-world camera imperfections and variability.

## New Noise Components Added

### 1. **Lighting-Dependent Noise Profiles**
- **Low Light**: High shot noise (5-15Hz), increased threshold variation, lower bandwidth
- **Normal Light**: Moderate noise levels (1-5Hz shot noise)
- **High Light**: Minimal shot noise (0.1-1Hz), higher bandwidth, less variation
- **Random Selection**: Each video gets a random lighting condition (30% low, 50% normal, 20% high)

### 2. **Spatial Noise Variation (Fixed Pattern Noise)**
- `noise_rate_cov_decades`: Creates pixel-to-pixel variation in noise rates
- Simulates manufacturing variations across the sensor array
- Range: 0.05-0.25 depending on lighting conditions

### 3. **Intensity-Dependent Shot Noise**
- Shot noise automatically varies with scene brightness
- Higher noise in dark regions, lower in bright regions
- Uses the built-in `SHOT_NOISE_INTEN_FACTOR = 0.25` mechanism

### 4. **Photoreceptor Noise Model**
- `photoreceptor_noise = True`: More realistic temporal noise statistics
- Creates alternating ON/OFF noise events (like real cameras)
- Gaussian noise injection before lowpass filtering

### 5. **Bandwidth Limitations**
- Random cutoff frequencies (15-100Hz) based on lighting
- Simulates finite photoreceptor response times
- Lower bandwidth in low light conditions

### 6. **Temporal Effects**
- **Refractory Period**: Random 0.2-0.8ms per video
- **Leak Jitter**: 10-20% timing variation for leak events
- **Temperature Effects**: Simulates different operating temperatures

### 7. **Camera Imperfections**
- **Hot Pixels**: 30% chance of increased spatial noise (1.5-2.5x)
- **Dead/Stuck Pixels**: 10% chance of extreme threshold variation
- **Temperature Drift**: Affects leak and shot noise rates

### 8. **Video-to-Video Randomization**
- Each video gets different noise characteristics
- Random seeds ensure reproducibility within video
- Different noise scales and profiles per video

## Technical Parameters

### Shot Noise Rates (per pixel)
- **Low Light**: 5-15 Hz
- **Normal Light**: 1-5 Hz  
- **High Light**: 0.1-1 Hz

### Leak Noise Rates
- **Low Light**: 0.1-0.3 Hz
- **Normal Light**: 0.03-0.1 Hz
- **High Light**: 0.01-0.05 Hz

### Threshold Variations
- **Low Light**: σ = 0.06-0.12 (high variation)
- **Normal Light**: σ = 0.03-0.07
- **High Light**: σ = 0.02-0.05 (low variation)

### Bandwidth Characteristics
- **Low Light**: 15-35 Hz (slow response)
- **Normal Light**: 25-60 Hz
- **High Light**: 50-100 Hz (fast response)

## Key Improvements Over Previous "Synthetic" Approach

1. **Realistic Noise Statistics**: Photoreceptor noise creates proper temporal correlations
2. **Spatial Heterogeneity**: Pixel-to-pixel variations like real sensors
3. **Lighting Adaptation**: Noise characteristics change with scene conditions
4. **Temporal Realism**: Proper refractory periods and jitter
5. **Manufacturing Variations**: Hot pixels, dead pixels, temperature effects
6. **Video Diversity**: Each video has unique noise characteristics

## Usage
The enhanced `set_args()` function automatically applies these realistic noise models:
```python
def set_args():
    # ... basic setup ...
    set_realistic_noise_profile(args, 'random')  # Random lighting condition
    add_burst_and_saturation_effects(args)       # Camera imperfections
```

## Expected Results
- More realistic event distributions
- Proper noise characteristics matching real DVS cameras
- Better generalization for downstream ML applications
- Reduced "synthetic" appearance in generated event data

## References
- v2e technical paper on DVS simulation
- Real DVS camera noise characteristics studies
- Photoreceptor noise modeling from neuromorphic vision literature
