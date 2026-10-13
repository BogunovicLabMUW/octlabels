# Layer files

A layer file holds the A-scan aligned annotations of one B-scan: layer
boundaries, regions and their properties. It is a UTF-8 YAML file named
`NNN.yml`, optionally compressed as `NNN.yml.gz` (gzip) or `NNN.yml.xz` (xz).

## Example

```yaml
info:
  width: 4
  height: 496
  idx: 0
  axial_resolution_um: 3.87

layers:
  - ILM:
    - values: 338,341,342.5,  0
    - confid:   1,  1,    1,  0
    - uncrtn:  12, 40,  999,  0
  - BM:
    - values: 401,402,  403,404
    - confid:   1,  1,    1,  1

regions:
  - DRUSEN:
    - values: 0,1,1,0
```

## Structure

The top level is a mapping with up to three keys, each optional:

| Key | Content |
|---|---|
| `info` | A mapping of metadata, see [Metadata](../guide/metadata.md). |
| `layers` | A list of layer boundaries. |
| `regions` | A list of regions (presence of a biomarker per A-scan). |

Readers should ignore other top-level keys (octlabels warns about them).

`layers` and `regions` are lists with one entry per annotation. Each entry is
a mapping with a single key, the annotation's name, whose value is a list of
**rows**; each row is a mapping with a single key:

```yaml
layers:
  - <name>:
    - <row key>: <row>
    - <row key>: <row>
```

The order of the entries is meaningful (for layers, typically from the inner
to the outer retina).

## Rows

Every row has exactly one entry per A-scan, so all rows of a file have the
same length, `info.width`.

| Row key | Meaning | Values |
|---|---|---|
| `values` | Layers: boundary height (row index, 0 at the top of the image). Regions: 1 if the biomarker is present, 0 otherwise. | Layers: numbers. Regions: 0 or 1. |
| `confid` | Confidence of each value; 0 means invalid. | Integers 0–255. |
| `uncrtn` | Normalized uncertainty. | Integers 0–999. |
| any other | A [custom property](../guide/custom-properties.md). | Numbers. |

`values` is always present. Layers always have `confid` (octlabels writes all
ones if none is given). Every other row is optional; an empty row (`- confid:`)
means the property is not stored.

## Row styles

A row is written in one of two styles; readers must accept both:

- **Comma-separated text** (default): `338,341,342.5,  0`. YAML reads it as a
  string; split it at the commas and strip spaces.
- **YAML list** (`yaml_arrays=True` in octlabels): `[338, 341, 342.5, 0]`.
  YAML reads it as a list of numbers. NaN and infinity are written as `.nan`,
  `.inf` and `-.inf`.

A row with a single A-scan may also appear as a single number.

Within one annotation, all cells of all rows are padded with spaces to the same
width, so the values of an A-scan line up in one column. The keys are padded
too when custom properties have longer names. Padding has no meaning.

Numbers are written with as few characters as possible: integers without a
decimal point, layer heights with at most two decimals, other floats in their
shortest exact form.

## Names and info values

Names and `info` values are YAML scalars. Strings are written without quotes
when YAML reads them back unchanged, and as double-quoted strings otherwise (for
example `"yes"`, `"1"` or `"scan #3"`).

## Reading a layer file without octlabels

The format needs nothing beyond a YAML parser:

```python
import numpy as np
import yaml

with open("patient_01/layers/000.yml") as file:
    data = yaml.safe_load(file)

for entry in data["layers"]:
    for name, rows in entry.items():
        row = {}
        for item in rows:
            row.update(item)
        text = row["values"]
        values = np.array(text.split(","), dtype=float) if isinstance(text, str) else np.array(text)
        print(name, values[:5])
```
