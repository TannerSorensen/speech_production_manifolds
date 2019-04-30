# speech_production_manifolds

Manifold representations of vocal tract movements in real-time magnetic resonance images

## Getting started

1. Download data as avi files

```bash
scp -r tsorense@siren.usc.edu:/span/home/tsorense/speech_production_manifolds_data/ .
```

The folder `speech_production_manifolds_data` contains one subfolder for each speaker (`F101`, `M101`, `F102`, `M102`, ...).
Inside each speaker folder there is a folder `avi` with imaging data in avi format
and a folder `wav` with concurrently recorded audio files in wav format.

2. Download data as avi files

```bash
scp -r tsorense@siren.usc.edu:/span/home/tsorense/speech_production_manifolds_alignments/ .
```

The folder `speech_production_manifolds_alignments` contains one subfolder for each speaker (`F101`, `M101`, `F102`, `M102`, ...).
Inside each speaker folder there is a folder `align` with forced alignments obtained with Gentle from the wav files in folder `wav` and transcripts in folder `trans`.
Other files and folders can be ignored (or simply not downloaded), as they were only used to make the alignments and are not suitable for analysis.

3. Convert avi files to png series

Navigate to a speaker directory (e.g., `speech_production_manifolds_data/F101/`) and run the following command.

```bash
./write_png_series.sh avi/
```

This command creates a folder `png` containing one subfolder for each avi file containing the images in png format.

Repeat for as many speakers as desired.

4. Save png files and labels to npz file.

Run the python script `save_npz.py` 

```bash
./save_npz.py speech_production_manifolds_data/
```

This will create a file `nsf_vtsf.npz`, which contains image data as NumPy array, phone labels, and speaker labels. The data-set is split into training, validation, and test sets.

The following Python code chunk illustrates how to load `nsf_vtsf.npz` in Python.

```python
from save_npz import read_nsf_vtsf
x_train, x_val, x_test, phone_train, phone_val, phone_test, spkr_train, spkr_val, spkr_test, \
    filenames_train, filenames_val, filenames_test = read_nsf_vtsf('nsf_vtsf.npz')
```
