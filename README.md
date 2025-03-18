# **MSA Generation with Seqs2Seqs Pretraining:Advancing Protein Structure Predictions**

<img src="https://p.ipic.vip/9l6wrb.png" alt="method (1)" style="zoom: 25%;" />

#### [Paper](https://openreview.net/pdf/fd516f23b421f9d03d5b978b03eded9900f0a462.pdf)

## Pretrain

**All the commands are designed for slurm cluster, we use huggingface trainer to pretrain the model, more details could be find [here](https://huggingface.co/docs/transformers/main_classes/trainer)**

   1. Construct local binary dataset ( load training data from cluster is too slow, so it's better to  fisrt construct all your dataset to .bin file as shown in datasets )

      ```python
      python utils.py \
         --output_dir ./datasets/ \
         --random_src --src_seq_per_msa_l 5\
         --src_seq_per_msa_u 10 \
         --total_seq_per_msa 25 \
         --local_file_path  path_to_pretrained_dataset 
      ```

   2. install dependency libraries

      1. `conda create -n msagen python=3.10`
      2. `pip install -r requirements.txt`

   3. `bash run.sh`

## Inference

1. download [checkpoints](https://drive.google.com/file/d/12cYk3WZDX18j-9xwYK9uu2kaGjmLuowB/view) (According to the latest confidentiality policy, the model weights will not be made public.)
2. run inference by `bash scripts/inference.sh`

> Note: all inference code is in inference.py*

## Evaluation

| DATASET | MSA                                 | STRUCTURE                                                    |
| ------- | ----------------------------------- | ------------------------------------------------------------ |
| CASP15  | <https://zenodo.org/record/8126538> | [google drive](https://github.com/deepmind/alphafold/blob/main/docs/casp15_predictions.zip) |

### Alphafold2 Prediction

1. Please refer to [Alphafold2 GitHub](https://github.com/deepmind/alphafold) to learn more about set up af2.

2. We provide scripts to use alphafold2 to launch protein structure prediction by `bash scripts/run_af2`, one need to modify `msa directory`

### LDDT

  1. follow this document for lddt evaluation tool download <https://www.openstructure.org/>
  2. follow this document for <https://www.openstructure.org/docs/2.4/mol/alg/lddt/> usage

### Ensemble

Directly run following to get .json file of final results.

```python
python ensemble.py --predicted_pdb_root_dir ./af2/casp15/orphan/A1T3R1.5/
```

## :paperclip: Citation

```bibtex
@misc{zhang2023enhancing,
      title={Enhancing the Protein Tertiary Structure Prediction by Multiple Sequence Alignment Generation}, 
      author={Le Zhang and Jiayang Chen and Tao Shen and Yu Li and Siqi Sun},
      year={2023},
      eprint={2306.01824},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```

## :email: Contact

please let us know if you have further questions or comments, reach out to [le.zhang@mila.quebec]
