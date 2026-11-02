"""Unit tests for mini_batch_trainer.py components."""
import pytest
import torch
from torch.utils.data import DataLoader
from mini_batch_trainer import StoryBatchDataset, build_minibatch_datasets


def test_story_batch_dataset():
    matrix = torch.arange(20).reshape(4, 5)  # 4 stories, seq_len=4
    ds = StoryBatchDataset(matrix, seq_len=4)
    assert len(ds) == 4
    x, y = ds[0]
    assert x.shape == (4,)
    assert y.shape == (4,)
    assert torch.equal(x, matrix[0, :4])
    assert torch.equal(y, matrix[0, 1:5])


def test_build_minibatch_datasets():
    stories = [
        "once upon a time there was a little girl named lily . she loved playing .",
        "one day a boy named timmy saw a dog in the park . they had fun .",
        "the dog barked and jumped high in the sky . it was happy .",
    ] * 5
    holdout = [
        "lily was very happy today . she went to see her friend .",
    ] * 2

    train_ds, ho_ds, itos, stoi = build_minibatch_datasets(
        stories, holdout, max_vocab_size=32, seq_len=10
    )
    assert len(train_ds) > 0
    assert len(ho_ds) > 0
    assert "lily" in stoi
    assert len(itos) <= 32

    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (4, 10)
    assert batch_y.shape == (4, 10)
