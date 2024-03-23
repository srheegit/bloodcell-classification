# bloodcell-classification

CNN to classify blood cells

PIC 16B Winter 2024 Final Project by Alankrita Ghosh, Kelvin Luu, and Sam Rhee

Data was sourced from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells).

[Because Gradescope doesn't give the repo link: <https://github.com/luukelvin/bloodcell-classification>]

## Details
The notebook `bloodcell-classification-model-built.ipynb` is our first, custom-built model that implements
a variety of ML architectures. The second notebook `bloodcell-classification-tl.ipynb` contains our
transfer learning model with EfficientNetB0. 

We defined our own preprocessing and visualization functions in the `data_processing` module/directory.
The `saved_weights` directory contains the updated weights for the transfer learning model.

## References
[1] Kaggle. Blood cell images. <https://www.kaggle.com/datasets/paultimothymooney/blood-cells>. Accessed: 2024-03-11.

[2] Tan, M. and Le, Q. V. Efficientnet: Rethinking model scaling for convolutional neural networks. ICML, 2019a.
