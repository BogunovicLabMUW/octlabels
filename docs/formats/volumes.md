# Volumes

A volume (all B-scans of a scan) is a folder of B-scan files, or optionally a
single zip file of them.

## Folders

Files are named after the B-scan index with at least three digits:

```text
layers/  000.yml  001.yml  002.yml  ...
lesions/ 000.png  001.png  002.png  ...
```

The number in the file name is the B-scan index; layer files also store it as
`info.idx`. Files with other names are ignored, so a folder may contain notes
or images. Indices do not need to be contiguous: a volume may contain only the
annotated B-scans.

Layer files in a folder may be compressed individually (`000.yml.gz`,
`000.yml.xz`).

## Zip files

A volume can also be stored as one zip file containing the same files. The zip
compression can be Deflate (opens in any zip tool) or LZMA (smaller; needs a
tool such as 7-Zip on Windows). Inside a zip, files may be in a subfolder, so
zipping a folder of B-scan files produces a valid volume.

Because zip files allow reading a single entry, one B-scan of a zip volume can
be read without unpacking the others.

## Layers and lesions of one scan

Layers and lesions are separate volumes, conventionally side by side:

```text
patient_01/
  layers/    (or layers.zip)
  lesions/   (or lesions.zip)
```

B-scans match by index: `layers/007.yml` and `lesions/007.png` belong to the
same B-scan.
