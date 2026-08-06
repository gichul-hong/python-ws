import json
import os

with open('iter12-gemini/train_convnext.py', 'r', encoding='utf-8') as f: code1 = f.read()
with open('iter12-gemini/train_scratch.py', 'r', encoding='utf-8') as f: code2 = f.read()
with open('iter12-gemini/ensemble_5fold.py', 'r', encoding='utf-8') as f: code3 = f.read()

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# DS2 Challenge Team 1 Final Code\n",
                "This notebook contains the complete pipeline for our final submission.\n",
                "1. ConvNeXt-Tiny (128px) 5-Fold Training & Inference\n",
                "2. Scratch CNN (48px) 5-Fold Training & Inference\n",
                "3. OOF Optimization & Ensemble"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code1.split('\n')]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code2.split('\n')]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code3.split('\n')]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('DS2_challenge_team1_final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook created.")
