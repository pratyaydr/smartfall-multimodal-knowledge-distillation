# Data access and provenance

## Supplied evidence

The only attachment for this task was the Kaggle notebook. Its code references:

- Kaggle dataset slug: `pratyaydhar/smartfallmmdataset`.
- Local Kaggle root: `/kaggle/input/datasets/pratyaydhar/smartfallmmdataset/SmartFallMM-Dataset-main`.
- Kaggle dataset-version identifier in notebook metadata: `15123534`.
- Processed output: `/kaggle/working/smartfallmm_lighthart_final.npz`.

The last path names an output; those NPZ bytes were not supplied. The preserved
notebook metadata is the authoritative source for the numeric identifier; no
exact dataset revision or download URL is reconstructed from that ID.

## Obtain data

Use the [Kaggle mirror](https://www.kaggle.com/datasets/pratyaydhar/smartfallmmdataset)
if you have access to the source version. The live Kaggle page could not be
retrieved during this task, so its present availability and license fields are
unverified. An authenticated Kaggle CLI, if already set up locally, can download
the latest mirror to your machine:

```bash
kaggle datasets download -d pratyaydhar/smartfallmmdataset -p data/raw --unzip
```

This optional command requires the external Kaggle CLI and user-managed
credentials. It is not part of the package or tested installation. Never place
`kaggle.json`, tokens, raw recordings, or downloaded archives in Git.
The latest download is not guaranteed to reproduce the original version.

The [upstream repository](https://github.com/txst-cs-smartfall/SmartFallMM-Dataset)
is maintained by the SmartFall Group at Texas State University. Inspect access
conditions and the actual CSV format before use. The upstream README states
copyright retention and requests attribution; no permissive data-redistribution
license was identified in its visible root. Consequently, no dataset is bundled
with this repository and no redistribution permission is claimed.

## Exact inputs used by the notebook

Only skeleton CSVs with matching watch accelerometer files are processed. Trials
are parsed from `S<number>A<number>T<number>` filenames. The final three watch
columns become acceleration; the final 96 skeleton columns become `(32,3)`
coordinates. Activity number >9 produces label 1. The preprocessing pipeline
produces one NPZ with five numeric arrays. See `docs/SCIENTIFIC_AUDIT.md` for
header, row-trimming, sampling-rate, normalization, and alignment concerns.

No raw or processed sensor data is included. The synthetic smoke check creates
random test tensors and CSVs in a temporary directory; they are not participant
measurements. Re-run the preprocessing with the exact source data to check the
recorded 13,077-window count and the 23 reported unreadable CSV errors.

## Attribution

Acknowledge the SmartFall Group, Texas State University, when using the dataset.
Dataset creators and the published LightHART research should receive appropriate
citation when their data or methods underpin future research. This notebook's
class name is not evidence that it exactly reproduces the published LightHART
architecture. Its source lineage and borrowed-code licensing were not fully
established by the attachment.
