# Non-Textual Data Extraction — CBIR for Cars

Content-based image retrieval (CBIR) over a car-image dataset, built without
any deep learning. Two complementary descriptors are combined into a single
similarity score:

- **ORB keypoints** — local features, robust to viewpoint changes; matched
  with brute-force Hamming distance.
- **Double-histogram color analysis** — global color distribution in two
  representations, captures paint colour and overall tone.

Result: given a query car image, the system returns the most visually similar
images from the indexed dataset (used here on Stanford Cars).

## Pipeline

1. **Index** — `descriptorsGenerator.py` walks a folder, extracts and saves
   ORB descriptors per image.
2. **Query** — `CBIR.py` takes a query image, runs both ORB matching and
   colour-histogram comparison against the index, returns the top matches.

## Setup

```bash
pip install opencv-python numpy matplotlib
```

## Usage

```bash
# build the descriptor index
python Non_Textual_Data_Extraction_Project/descriptorsGenerator.py \
       --dataset path/to/cars --output path/to/descriptors

# query
python Non_Textual_Data_Extraction_Project/CBIR.py \
       --query path/to/query.jpg --dataset path/to/cars
```

Dataset used: [Stanford Cars on Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset).

## Layout

```
Non_Textual_Data_Extraction_Project/
  CBIR.py                       query-time matcher (ORB + colour histograms)
  descriptorsGenerator.py       offline index builder
  Result_example.png            sample output
  testsAndExperiments/          earlier experiments and ablations
Non-textual_data_extraction_Report.pdf       full report
Non-Textual_Data_Extraction_Presentation.pdf slide deck
```
