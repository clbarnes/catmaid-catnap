# catnap

View [CATMAID](https://catmaid.org) data using [napari](https://napari.org), primarily for generating ground truth pixel label data for neuronal reconstructions.

`catnap` is optimised for datasets which fit comfortably into RAM - i.e. for dense labelling in a small volume.

For more complex tasks, consider

- [BigCAT](https://github.com/saalfeldlab/bigcat)
- [Paintera](https://github.com/saalfeldlab/paintera)
- [TrakEM2](https://imagej.net/TrakEM2)
- [Ilastik](https://www.ilastik.org/)

## Usage

1. Use `catnap-create` to convert hdf5/zarr/n5 image data, plus CATMAID skeleton annotations, into catnap's hdf5 format
2. Use `catnap` to create or edit pixel labels
3. Use `catnap-assess` to check the labels for false merges and splits

### catnap GUI

This is basically just a napari window.
At time of writing, this is not particularly well documented, although `Help -> Key bindings` is useful for keyboard shortcuts.

Make sure you select the `labels` layer before trying to edit.
Press `n` to pick the `n`ext unused label, or use the colour picker to use an existing label.
There is a button for whether labelling and paint filling should flow through onto different slices
(I recommend against it unless you're doing a merge you understand well).

Viewing in 3D (and related functions like rolling dimensions) is not supported.

More advanced features, including exporting labels, are available in the ipython console built into napari.
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

It is recommended that you export your labels regularly, because it's the only way to save your work.

### Command line

#### `catnap-create`: Data preparation

Create a file for use with catnap using an existing raw image dataset, fetching annotation data from CATMAID, and creating a volume for labels with seed labels around treenodes.

```_catnap_create
usage: catnap-create [-h] [-o OFFSET] [-r RESOLUTION] [-f] [-t]
                     [--label LABEL] [-s SEED_RADIUS] [--base-url BASE_URL]
                     [--project-id PROJECT_ID] [--token TOKEN]
                     [--auth-name AUTH_NAME] [--auth-pass AUTH_PASS]
                     [-c CREDENTIALS] [-v]
                     input output

positional arguments:
  input                 Path to HDF5 dataset containing raw data, in the form
                        '{file_path}:{dataset_path}'
  output                Path to HDF5 group to write raw, annotation, and label
                        data, in the form'{file_path}:{group_path}'. If the
                        group path is not given, it will default to the file's
                        root.

optional arguments:
  -h, --help            show this help message and exit
  -o OFFSET, --offset OFFSET
                        Offset, in world units, of the raw data's (0, 0, 0)
                        from the CATMAID project's (0, 0, 0), in the form
                        'z,y,x'. Will default to the raw dataset's 'offset'
                        attribute if applicable, or '0,0,0' otherwise
  -r RESOLUTION, --resolution RESOLUTION
                        Size, in word units, of voxels in the raw data, in the
                        form 'z,y,x'. Will default to the raw dataset's
                        'resolution' attribute if applicable, or '1,1,1'
                        otherwise
  -f, --force           Force usage of the given offset and arguments, even if
                        the dataset has its own which do not match
  -t, --transpose-attrs
                        Reverse offset and resolution attributes read from the
                        source (may be necessary in some N5 datasets)
  --label LABEL, -l LABEL
                        If there is existing label data, give it here in the
                        same format as for 'input'. Offset and resolution are
                        assumed to be identical to the raw (conflicting
                        attributes will raise an error).
  -s SEED_RADIUS, --seed-radius SEED_RADIUS
                        Radius of the label seed squares placed at each
                        treenode, in px
  -v, --verbose         Increase logging verbosity

catmaid connection details:
  --base-url BASE_URL   Base CATMAID URL to make requests to
  --project-id PROJECT_ID
  --token TOKEN         CATMAID user auth token
  --auth-name AUTH_NAME
                        Username for HTTP auth, if necessary
  --auth-pass AUTH_PASS
                        Password for HTTP auth, if necessary
  -c CREDENTIALS, --credentials CREDENTIALS
                        Path to JSON file containing credentials (command line
                        arguments will take precedence)
```

e.g.

```sh
catnap-create existing_data.hdf5:/raw catnap_format.hdf5 --credentials my_credentials.json --seed-radius=3
```

#### `catnap`: Label editing

Open a napari window viewing the pre-formatted data for label annotation.

```_catnap
usage: catnap [-h] [-v] [-l LABEL] input

positional arguments:
  input                 Path to HDF5 group containing catnap-formatted data,
                        in the form '{file_path}:{group_path}'. If the group
                        path is not given, it will default to the file's root.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Increase logging verbosity
  -l LABEL, --label LABEL
                        Path to HDF5 dataset containing label data (if it's
                        not in the expected place in the input HDF5), in the
                        form '{file_path}:{group_path}'. If the file path is
                        not given, uses the 'input' file.
```

e.g.

```sh
catnap catnap_format.hdf5
```

#### `catnap-assess`: Segmentation assessment

Write CSVs of false splits and merges.

```_catnap_assess
usage: catnap-assess [-h] [-v] [-m FALSE_MERGE] [-s FALSE_SPLIT] [-u UNTRACED]
                     [-r] [-l LABEL]
                     input

Merges are assessed before splits regardless of argument order.

positional arguments:
  input                 Path to HDF5 group containing catnap-formatted data,
                        in the form '{file_path}:{group_path}'. If the group
                        path is not given, it will default to the file's root.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Increase logging verbosity
  -m FALSE_MERGE, --false-merge FALSE_MERGE
                        Assess false merges and write to CSV file. If '-' is
                        given, write to stdout.
  -s FALSE_SPLIT, --false-split FALSE_SPLIT
                        Assess false splits and write to CSV file. If '-' is
                        given, write to stdout.
  -u UNTRACED, --untraced UNTRACED
                        Write labels of segments with no treenodes in them. If
                        '-' is given, write to stdout.
  -r, --relabel         Assign each connected component a new label. Useful to
                        assess whether there are skeletons which correctly
                        share labels around their treenodes, but those
                        labelled regions are not contiguous.
  -l LABEL, --label LABEL
                        Path to HDF5 dataset containing labels, in the form
                        '{file_path}:{group_path}'. Must have compatible
                        resolution and offset with 'input'.
```

e.g.

```sh
catnap-assess catnap_format.hdf5 --false-split splits.csv --false-merge merges.csv
```

See `catnap-assess --help` for more information.

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

## Notes

The package name is `catmaid-catnap` (use this for e.g. `pip install`ing), to disambiguate this project from the unrelated [REST API testing utility](https://pypi.org/project/Catnap/) of the same name.
The module name is `catnap` (use this for e.g. `import`).
