# octlabels

octlabels defines file formats for annotations of OCT (optical coherence
tomography) scans, and a Python library to read, write and process them.

## Why octlabels

OCT annotations are often stored as label maps: one PNG per B-scan in which
every pixel holds a class, the retinal layers as bands between boundaries, and
often the lesions in the same image. That is simple, but it loses information
that matters:

- **A lesion hides what is underneath.** Where a lesion is painted over the
  bands, it is no longer clear whether a boundary does not exist there or was
  simply covered by the lesion.
- **"Not annotated" cannot be stored.** A boundary that the grader could not
  see, a region outside the scan, a layer that does not exist: in a label map
  they all look like some other class.
- **Precision is lost.** Boundaries are rounded to whole pixels, and lesions
  that overlap (such as hyperreflective foci within fluid) cannot be stored at
  all, because a pixel holds only one class.

![A combined label map next to the same annotations in octlabels](assets/label-maps.svg)

Other ways of storing OCT annotations have their own problems. Invalid
positions are encoded differently in every dataset (NaN, 0, −1, a separate
mask, or not at all), and misreading them silently corrupts training and
evaluation. Model outputs lose their confidence and uncertainty as soon as
they are saved, because there is nowhere to put them. Vendor formats are
closed, DICOM is rarely used for layer boundaries in practice, MATLAB files
and pickled Python objects tie the data to one tool, and layer names and their
order are often implied by the position in an array instead of being stored.

octlabels stores layer boundaries as what they are, one height per A-scan, with
an explicit validity flag per A-scan and any further per-A-scan values next to
them. Lesions are stored separately as 2D masks, either mutually exclusive
(e.g. fluid compartments) or as overlays that may overlap anything (e.g.
hyperreflective foci). Every file carries the names of what it contains.

The files have further practical advantages:

- **Corrections are visible in version control.** Uncompressed layer files are
  text, so a git diff of a grader's correction shows exactly the A-scans that
  changed.
- **Labels can be shared without images.** The files are small and contain no
  image data, which simplifies exchanging annotations between groups.
- **They remain readable.** Plain YAML and PNG outlive the tools that wrote
  them, which matters for published datasets and long-running studies.

octlabels is meant for anyone who trains or evaluates segmentation models,
grades or corrects annotations, or publishes and exchanges OCT datasets.

## Design goals

The formats are designed to be:

- **Accessible B-scan by B-scan.** Every B-scan has its own file, so one
  B-scan can be read, checked or replaced without loading the whole volume. A
  volume is a folder of these files, or optionally one zip file of them.
- **Built on established standards, independent of this library.** octlabels
  does not invent a new file type: layers and regions are stored in standard
  YAML files, lesions in standard PNG files, and compression and archives use
  gzip, xz and zip. Any YAML parser or image library, in any language, can read
  them; this package is a convenience, not a requirement.
- **Readable by humans.** Layer files are plain text in which all values of an
  A-scan line up in one column, so invalid or suspicious positions are visible
  without any software. Lesion files are PNG images that any image viewer
  displays in color.
- **Easy to parse by machines.** Every layer row holds one number per A-scan
  (comma-separated, or a YAML list); every lesion pixel holds a lesion ID and
  overlay bits, with the lesion names stored as JSON text inside the PNG.
- **Platform independent.** Only UTF-8 text and standard PNG: no custom binary
  layouts or language-specific serialization. Every file is self-describing,
  with names stored next to the data.
- **As small as possible within these constraints.** Layer values are written
  with only the digits they need (plus the padding that aligns the columns),
  and files can be gzip or xz compressed; lesions are bit-packed into one 8-bit
  image per B-scan. On typical scans, a B-scan with 12 layers takes about
  3 KiB as `.yml.xz`, and its lesion masks 0.5–1.5 KiB as PNG.

## What can be stored

| Annotation | Per | Stored as | Python class |
|---|---|---|---|
| Layer boundary (height of a boundary, e.g. ILM) | A-scan | rows in a YAML file | `Layer` |
| Region (presence of a biomarker, e.g. drusen) | A-scan | rows in a YAML file | `AScanRegion` |
| Any property of a layer or region (confidence, uncertainty, your own) | A-scan | rows in a YAML file | properties of the above |
| Lesion mask (e.g. fluid, hyperreflective foci) | pixel | an 8-bit PNG | `Lesion` |

## Where to go next

- [Getting started](getting-started.md): installation and the most common calls.
- [File formats](formats/layers.md): the exact structure of the files, for
  reading them with or without this library.
- [User guide](guide/loading-and-saving.md): everything the library can do,
  including [custom properties](guide/custom-properties.md) and the
  [metadata keys](guide/metadata.md).
- [API reference](reference/octlabels.md): every class and function.
