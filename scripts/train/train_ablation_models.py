#!/usr/bin/env python3
"""Compatibility module: OriginalLeWM wraps Unisize256 for metric head training scripts.

The metric head training/eval scripts import OriginalLeWM from this module.
Since Unisize256 has the same interface (encoder, embedding_projector, predictor),
OriginalLeWM is simply an alias.
"""
from scripts.train.train_dim256 import Unisize256 as OriginalLeWM
