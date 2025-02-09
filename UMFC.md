## How to Install
Our code is built upon the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). To adapt it to our needs, we made some modifications to its framework. Therefore, setting up the `dassl` environment is required before running our code. Simply follow the instructions described [here](https://github.com/GIT-LJc/Dassl.change/blob/main/README.md) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `UMFC/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run

We provide the running scripts in `scripts`, which allow you to reproduce the results on the NeurIPS'24 paper.

Make sure you change the path in `DATA` and run the commands under the main directory `UMFC/`.

### Transductive Learning

**DomainNet:** `bash scripts/main.sh domainnet_prototype UMFC_ensemble_summary 6 0`

**ImageNet-Variants:** `bash scripts/main.sh imagenet_ars_cluster UMFC_ensemble_summary 3 0`


### Unsupervised Calibration
**DomainNet:** `bash scripts/main.sh domainnet_prototype UMFC_ensemble_trainu_summary 6 0`

**ImageNet-Variants:** `bash scripts/main.sh imagenet_ars_cluster UMFC_ensemble_trainu_summary 3 0`


### Test-Time Calibration
**DomainNet:** `bash scripts/main.sh domainnet_ttc UMFC_TTC_Summary 6 0`

**ImageNet-Variants:** `bash scripts/main.sh imagenet_ars_cluster UMFC_TTC_Summary 3 0`


### Data Preprocessing
Obtain clustering labels based on visual features extracted by CLIP.

**DomainNet:** `bash scripts/preprocessed.sh domainnet_preprocessed ZeroshotCLIP 6 0 trainu(or val)`

**ImageNet-Variants:** `bash scripts/preprocessed.sh imagenet_ars_preprocessed ZeroshotCLIP 3 0 trainu(or val)`
