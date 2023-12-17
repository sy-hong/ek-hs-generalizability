Improving Cross-Domain Hate Speech Generalizability with Emotion Knowledge
==========================================================================

About
-----
We propose an emotion-integrated multitask hate speech (HS) generalization framework that utilizes emotion knowledge to strengthen cross-domain HS generalization. We investigate emotion corpora with varying emotion categorical scopes to determine the best corpus scope for supplying emotion knowledge to foster generalized HS detection. We further assess the relationship between using pre-trained language models adapted for HS and its effect on our emotion-enriched HS generalization model. Experimental results on six publicly available datasets from six online domains support that our emotion-enriched HS detection generalization method demonstrates consistent generalization improvement in cross-domain evaluation, increasing generalization performance up to 18.1% and average cross-domain performance up to 8.5%, according to the F1 measure.

For more details, please refer to our papaer:
> S.Y. Hong and S. Gauch, Improving Cross-Domain Hate Speech Generalizability with Emotion Knowledge, Pacific Asia Conference on Language, Information and Computation (PACLIC 37), 2023. [\[PDF\]] (https://arxiv.org/pdf/2311.14865.pdf) 

Requirements and Setup
----------------------
Stable: Python >=3.8 + PyTorch 1.13.1

Install [PyTorch](https://pytorch.org/)

Install requirements in ``requirements.txt``:
```  
    pip install -r requirements. txt 
``` 

Suggested Setup:
```
ek-hs-generalizability/
├─ configs/         # contain default configurations for models
├─ emo_data/        # csv files for emotion analysis data ([text], [label])
│  ├─ ekman/
│  │  ├─ train/
│  │  ├─ eval/
│  │  ├─ test/
│  ├─ goemotions/
│  │  ├─ train/
│  │  ├─ eval/
│  │  ├─ test/
├─ hs_data/         # csv files for hate speech data ([text], [label])
│  ├─ train/
│  ├─ eval/
│  ├─ test/
├─ log/             # log files
├─ models/          # baselines and multitask models
├─ stat/            # visualization/error analysis (auto-generated)
│  ├─ [HS training data name]/
│  │  ├─ [HS testing data name]/
│  │  │  ├─ [emotion category]/
├─ utils/           # for models, batch, loss, meta, visualization...
```

Run Models
----------
Sample baseline model:
```
python main.py train --model baseline --plm diptanu/fBERT --main_dataset [path_to_train_set] --dev_file [path_to_dev_set] --test_file [path_to_test_set] --train_data_name [training_set_name] --test_data_name [testing_set_name] --log --logfile [path_to_log_file]
```

> ```model```: running one of the baselines model

> ```plm```  : pre-trained language model (e.g. BERT, fBERT)

> ```train_data_name```: name of the dataset used for training

> ```testing_set_name```: name of the dataset used for testing

> ```log```  : storing the logfile

> ```logfile```: where the logfile will be saved


Sample emotion knowledge-enriched multitask HS generalization model:
```
python main.py train --model multitask --plm diptanu/fBERT --main_dataset [path_to_train_set] --dev_file [path_to_dev_set] --test_file [path_to_test_set] --aux_datasets [path_to_aux_train_set] [path_to_aux_dev_set] [path_to_aux_test_set] --train_data_name [training_set_name] --test_data_name [testing_set_name] --emo_num [number of emotion categories] --log --logfile [path_to_log_file]
```

> ```model```: running the multitask mode

> ```plm```  : pre-trained language model 

> ```train_data_name```: name of the dataset used for training

> ```testing_set_name```: name of the dataset used for testing

> ```emo_num```: emotion categories of the emotion corpus (e.g. 28 for GoEmotions or 6 for Ekman)

> ```log```  : storing the logfile

> ```logfile```: where the logfile will be saved


Citation
--------
```
@inproceedings{
    syhong2023a,
    title={Improving Cross-Domain Hate Speech Generalizability with Emotion Knowledge},
    author={Shi Yin Hong and Susan Gauch},
    booktitle={Pacific Asia Conference on Language, Information and Computation (PACLIC 37)},
    year={2023},
    url={https://arxiv.org/pdf/2311.14865.pdf}
}
```
