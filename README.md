# Speech Production Manifolds

Manifold representations of vocal tract movements in real-time magnetic resonance images

## Direct download links

You do not need to run the code in this repository to access the data. The following list indicates where to find various data-sets on `siren.usc.edu`.

- avi files are available in folder `/span/home/tsorense/speech_production_manifolds_data/`.
- Gentle forced alignments are available as json files in folder `/span/home/tsorense/speech_production_manifolds_alignments/`.
- image data as NumPy array, phone labels, and speaker labels are available in the file `/span/home/tsorense/speech_production_manifolds/nsf_vtsf.npz`.

## Getting started

1. Download data as avi files

```bash
scp -r tsorense@siren.usc.edu:/span/home/tsorense/speech_production_manifolds_data/ .
```

The folder `speech_production_manifolds_data` contains one subfolder for each speaker (`F101`, `M101`, `F102`, `M102`, ...).
Inside each speaker folder there is a folder `avi` with imaging data in avi format
and a folder `wav` with concurrently recorded audio files in wav format.

2. Convert avi files to png series

Navigate to a speaker directory (e.g., `speech_production_manifolds_data/F101/`) and run the following command.

```bash
./write_png_image_series.sh avi/
```

This command creates a folder `png` containing one subfolder for each avi file containing the images in png format.

Repeat for as many speakers as desired.

3. Download Gentle forced alignments as json files

```bash
scp -r tsorense@siren.usc.edu:/span/home/tsorense/speech_production_manifolds_alignments/ .
```

The folder `speech_production_manifolds_alignments` contains one subfolder for each speaker (`F101`, `M101`, `F102`, `M102`, ...).
Inside each speaker folder there is a folder `align` with forced alignments in json format.
Other files and folders can be ignored (or simply not downloaded), as they were only used to make the alignments and are not suitable for analysis.

4. Save png files and labels to npz file.

Run the python script `save_npz.py` 

```bash
./save_npz.py speech_production_manifolds_data/ speech_production_manifolds_alignments/
```

This will create a file `nsf_vtsf.npz`, which contains image data as NumPy array, phone labels, and speaker labels. The data-set is split into training, validation, and test sets.

The following Python code chunk illustrates how to load `nsf_vtsf.npz` in Python.

```python
from save_npz import read_nsf_vtsf
x_train, x_val, x_test, phone_train, phone_val, phone_test, spkr_train, spkr_val, spkr_test, \
    align_train, align_val, align_test, filenames_train, filenames_val, filenames_test = read_nsf_vtsf('nsf_vtsf.npz')
```
