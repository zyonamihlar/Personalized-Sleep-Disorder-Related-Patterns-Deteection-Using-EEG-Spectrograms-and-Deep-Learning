# Personalized EEG Sleep Analysis System

This repository contains the Python source code developed for the final year Biomedical Engineering project:

Personalized Sleep Disorder Detection Using EEG Spectrograms and Deep Learning.

## Project Overview

The system processes EEG sleep recordings, converts 30-second EEG epochs into STFT spectrograms, classifies sleep stages using a Convolutional Neural Network, performs personalized fine-tuning, generates preliminary sleep-pattern indicators, and displays results through a graphical user interface.

## Main Files

- config.py: Defines project paths, EEG settings, spectrogram parameters, training parameters, and sleep-stage labels.
- prepare_sleep_edf.py: Preprocesses Sleep-EDF recordings.
- prepare_isruc.py: Preprocesses ISRUC-SLEEP recordings.
- combine_datasets.py: Combines processed Sleep-EDF and ISRUC-SLEEP datasets.
- train_model.py: Trains the generic CNN model.
- personalize_model.py: Performs personalized fine-tuning.
- test_model.py: Evaluates generic and personalized models.
- app.py: Runs the graphical user interface.

## Dataset Notice

The raw EEG datasets, processed NumPy arrays, and trained model files are not included in this repository due to file size limitations and dataset ownership restrictions. The datasets can be accessed from their original public sources.

## Disclaimer

The system is a research prototype for automated EEG-based sleep-stage classification and preliminary sleep-pattern interpretation. It is not intended to replace clinical polysomnography or expert medical diagnosis.
