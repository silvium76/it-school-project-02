# IT School · Project 02 — two mini-projects

Two solved notebooks for the Machine Learning course. Every line of code has an English comment underneath, and every code cell is followed by a **"What this cell does"** note with the real results of the run. The course's own instructions stay in Romanian.

| Notebook | Topic | Result | Run it |
|---|---|---|---|
| [`CNNs_proiect_02.ipynb`](CNNs_proiect_02.ipynb) | Transfer learning: adapting a pre-trained CNN to classify 5 kinds of flowers | **86%** validation accuracy (fine-tuned ResNet18) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/silvium76/it-school-project-02/blob/main/CNNs_proiect_02.ipynb) |
| [`Forecasting_proiect_02.ipynb`](Forecasting_proiect_02.ipynb) | Anomaly detection on a time series: finding the failures of an industrial machine from its temperature | **4 of 4** real failure periods found, **0** false alarms | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/silvium76/it-school-project-02/blob/main/Forecasting_proiect_02.ipynb) |

Both notebooks download their data by themselves on the first run, so no upload is needed.

---

## 1 · Transfer learning on images (`CNNs_proiect_02.ipynb`)

A network pre-trained on ImageNet already recognises edges, shapes and textures. We replace only its last layer with a new one that has one output per class of our project, then train:

- **feature extraction:** all pre-trained layers frozen; only the new last layer learns (a few thousand weights);
- **fine-tuning:** every layer adapts, with a 10× smaller learning rate.

**Data:** 5 classes (daisy, dandelion, roses, sunflowers, tulips), **100 photos per class**, drawn with a fixed seed from TensorFlow's free `flower_photos` set and split **80 / 20**: 400 training and 100 validation images, with no photo in both. See [Images](#images) below for how to get them.

**Results** (3 epochs, same seed, CPU only):

| Model | Strategy | Parameters | Validation accuracy | Training time |
|---|---|---|---|---|
| ResNet18 | feature extraction | 11.2 M | 74% | 36 s |
| EfficientNet-B0 | feature extraction | 4.0 M | 80% | 44 s |
| MobileNetV3-Large | feature extraction | 4.2 M | 84% | 30 s |
| **ResNet18** | **fine-tuning** | 11.2 M | **86%** | 76 s |

- **Where it goes wrong:** the 5×5 confusion matrix of the ResNet18 baseline shows the mistakes follow visual similarity.
  - Sunflowers are found 95% of the time, but they also absorb daisies and dandelions, which share a round yellow centre.
  - Tulips and roses are confused both ways.
- **The misclassified photos** are mostly hard cases: insects covering the flower, extreme close-ups, dandelion seed heads.
- **Caveats:** all models were still improving after 3 epochs. With 100 validation images, 1 image = 1 point, so 84% vs 86% is within the noise.

### Images

The photos are **not stored in this repository** (the full set is 218 MB). Three ways to get them:

1. **Automatic (recommended):** open [`CNNs_proiect_02.ipynb`](CNNs_proiect_02.ipynb) (in Colab with the button above, or locally) and run the cells in order. The cell after *Pasul 1 · Setările proiectului* downloads the archive once and builds:

   ```text
   data/flori/
   ├── train/   daisy/ dandelion/ roses/ sunflowers/ tulips/   (80 photos each)
   └── val/     daisy/ dandelion/ roses/ sunflowers/ tulips/   (20 photos each)
   ```

2. **Direct download:** [**flower_photos.tgz** (218 MB, 3,670 photos)](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz). Dataset page: [tensorflow.org/datasets/catalog/tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers).

3. **By hand:** unpack the archive into a `data/` folder next to the notebook, giving `data/flower_photos/<class>/*.jpg`, then run the notebook. It skips the download and copies only the chosen photos.

[`flori_split.csv`](flori_split.csv) lists **exactly which 500 photos are used**, with their class, `train` / `val` split, photographer and Flickr link. The selection is reproducible (`random.Random(42)`), so the automatic and manual routes give the same folders.

To use your **own images** instead, put the path of a folder with `train/<class>/` and `val/<class>/` subfolders in `CALE_DATE` (the settings cell).

**Licence and credits:** all photos are licensed [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) by their Flickr authors. The authors of the 500 photos used (some appear in the notebook's figures) are listed in `flori_split.csv`; all 3,670 are credited in `LICENSE.txt` inside the archive.

## 2 · Machine-temperature anomalies (`Forecasting_proiect_02.ipynb`)

**Data:** the NAB (Numenta Anomaly Benchmark) `machine_temperature_system_failure` series: 22,695 measurements, one every 5 minutes, from December 2013 to February 2014, with 4 failure periods marked by the researchers.

**Method:** a daily mean temperature, compared with "normal" = the **median of the 7 previous days** (`rolling(7).median().shift(1)`). A day more than **20% colder** than normal raises an alert.

**Results:**

| Threshold | Days flagged | Real periods found | False alarms |
|---|---|---|---|
| 10% | 12 | 4 / 4 | 6 |
| **20%** | **6** | **4 / 4** | **0** |
| 40% | 1 | 1 / 4 | 0 |
| 20%, "same weekday last week" as normal | 5 | 3 / 4 | 1 |

- **Day by day:** TP 6, FN 2, FP 0, TN 65, so recall 0.75 and precision 1.0.
- The 2 missed days are the day after a failure, when the machine was already back to normal.

## How to run

- **Google Colab:** open a notebook and run the cells from top to bottom. A T4 GPU makes the CNN notebook much faster, but is not required.
- **Locally:** Python 3.11 with `torch`, `torchvision`, `pandas`, `matplotlib`, `numpy` and `jupyter`. Both notebooks download their data from the internet the first time.

The runs shown in the notebooks were made on a CPU (Docker, no GPU): the CNN notebook takes ~5 minutes including the download, the forecasting one a few seconds.
