# catnap

View [CATMAID](https://catmaid.org) data using [napari](https://napari.org), primarily for generating ground truth pixel label data for neuronal reconstructions.

`catnap` is optimised for datasets which fit comfortably into RAM - i.e. for dense labelling in a small volume.

For more complex tasks, consider

- [BigCAT](https://github.com/saalfeldlab/bigcat)
- [Paintera](https://github.com/saalfeldlab/paintera)
- [TrakEM2](https://imagej.net/TrakEM2)
- [Ilastik](https://www.ilastik.org/)

## Usage

### Command line

#### Data preparation

Create a file for use with catnap using an existing raw image dataset, fetching annotation data from CATMAID, and creating a volume for labels with seed labels around treenodes.

```sh
catnap-create existing_data.hdf5:/raw catnap_format.hdf5 --credentials my_credentials.json --seed-radius=3
```

See `catnap-create --help` for more information.

#### Annotation

Open a napari window viewing the pre-formatted data for label annotation.

```sh
catnap catnap_format.hdf5
```

### Library

Assuming you have a chunk of image data as a numpy array in ZYX,
with a given resolution and offset inside a CATMAID project,
and a [catpy-style JSON credentials file](https://catpy.readthedocs.io/en/latest/catpy.client.html#catpy.client.CatmaidClient.from_json) for your CATMAID instance:

```python
from catnap import Image, Catmaid, CatnapIO, CatnapViewer, gui_qt

# attach the necessary metadata to our plain numpy array
img = Image(my_image_data, resolution=my_resolution, offset=my_offset)

# fetch the skeleton and connector data for our subvolume
cio = CatnapIO.from_catmaid(Catmaid.from_json(my_credentials_path), img)

# generate a seed label volume where every treenode has a small patch of label
# unique to the skeleton ID
cio.make_labels(set_labels=True)

# save all this to an HDF5 file
cio.to_hdf5("path/to/cdata.hdf5")

# you can retrieve it later with
cio_2 = CatnapIO.from_hdf5("path/to/cdata.hdf5")

with gui_qt():  # this is a re-export from napari
    my_cviewer = CatnapViewer(cio)
    my_cviewer.show()

```

Then make your labels, using the color picker to select the labels underneath treenodes.
Viewing in 3D is not supported.

In the `napari` console, the CatnapViewer is available as the `cviewer` variable.

```python
>>> # Save your labels. The timestamp will be included as a dataset attribute.
>>> cviewer.export_labels("path/to/labels.hdf5", "my_labels")
>>> # Include the original raw and annotation data (changes to these are not saved).
>>> # By default, internal structure is compatible with CatnapIO.from_hdf5
>>> cviewer.export_labels("path/to/full_labels.hdf5", with_source=True)
>>> # You can navigate in "real-world" coordinates with
>>> cviewer.jump_to(z=19.8, y=19355)
>>> # ... or in pixel space with
>>> cviewer.jump_to_px(y=10, x=5)
>>> # get more information on available functionality with
>>> help(cviewer)
```

## Notes

The package name is `catmaid-catnap` (use this for e.g. `pip install`ing), to disambiguate this project from the unrelated [REST API testing utility](https://pypi.org/project/Catnap/) of the same name.
The module name is `catnap` (use this for e.g. `import`).
