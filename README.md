# AURC
Accompanying repository of our AAAI-20 paper: "[Fine-Grained Argument Unit Recognition and Classification](https://aaai.org/Papers/AAAI/2020GB/AAAI-TrautmannD.7498.pdf)".

> ⚠️ The dataset was updated (cleaner parsing and encoding) and is not similar to the original one in the paper. 

> ⚠️ However, the count of the sentences and labels is the same.

> ⚠️ We will soon update the reported results.

### Setup

```
pip install -r requirements.txt
```

### Data Download

```
sh download.sh
```

### Annotations

The column `merged_segments` are the gold labels, created from the 5 annotators per sample.

Each entry is composed of 3 parts, resulted from the crowd annotation interface. 

Example `('false', '(21,22);(49,132);', 'con;pro;')`

The first part (`false` or `true`) indicates whether the annotator selected the field that no argument span can be detected. (`false` means there was an argument span selected)

The second part are the positions (in characters) of the argument segment(s) within the sentence in the form `(start_of_segment,length_of_segment)` separated with `;` if there are several segments.

The third part are the corresponding stance labels (`con` or `pro`) for the argument segment(s).


### Data Preparation

```
python3 preparation.py
```

### Trainining & Inference

```
sh run.sh
```

### Citation

If you find this dataset helpful, please cite our publication.

``` 
@inproceedings{trautmann2020fine,
  title = {Fine-Grained Argument Unit Recognition and Classification},
  author = {Dietrich Trautmann and
            Johannes Daxenberger and
            Christian Stab and
            Hinrich Schutze and
            Iryna Gurevych},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI} 2020},
  publisher = {{AAAI} Press},
  month = {2},
  year = {2020},
  url = {https://aaai.org/Papers/AAAI/2020GB/AAAI-TrautmannD.7498.pdf},
}
```
