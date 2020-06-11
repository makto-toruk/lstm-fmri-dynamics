# Capturing brain dynamics: latent spatiotemporal patterns predict stimuli and individual differences 

This repository is the official implementation of [Capturing brain dynamics: latent spatiotemporal patterns predict stimuli and individual differences](https://arxiv.org/abs/2030.12345). 

> Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> CUDA version: 10.0.130 

Data: We employed Human Connectome Project (HCP) movie-watching data from the [Young Adult study](https://www.humanconnectome.org/study/hcp-young-adult). Access to HCP data requires [registration](https://db.humanconnectome.org) and agreeing to their [Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms). The required data is under
> Session Type: 7T fMRI; Movie Task fMRI 2mm/32k FIX-Denoised (Compact)

To preprocess this data (parcellation, standardization, etc.), run:

```
python preprocess.py --input-data <path_to_data> \
    --output-data data/roi_ts --roi 300 --net 7
```

## Training and Evaluation

To train the clip prediction model in the paper, run this command:

```
python clip_lstm.py --input-data data/roi_ts --roi 300 --net 7 \
    --k_hidden 150 --train_size 100
```

To train competing models (FF: feed-forward, TCN: temporal convolution neural network):
```
python clip_ff.py --input-data data/roi_ts --roi 300 --net 7 \
    --k_hidden 150 --k_layers 5 --train_size 100

python clip_tcn.py --input-data data/roi_ts --roi 300 --net 7 \
    --k_hidden 150 --k_wind 10 --train_size 100
```



---

To train the behavior/personality prediction model in the paper:
```
python bhv_lstm.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


## Results

Comparison of various models for clip prediction:

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

Results are compared in `clip_compare.ipynb`.

## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
