# DNN Programming Assignment 2 — RNN, LSTM, GRU and Transformer

**Student:** Korada Sriya Kushal Kalyan  
**BITS ID:** 2026AJ05601

## Environment

- Python 3.10+
- PyTorch 2.x
- NumPy, pandas, Matplotlib and Basic Math, Date etc.
- scikit-learn is used for evaluation metrics in Problem 1.
- GPU: Kaggle GPU T4 for problem 2 and prayogashala VM CPU for problem 1.

Install the listed packages:

```bash
pip install -r requirements.txt
```

Record the exact versions used for your submission with:

```bash
python --version
python -c "import torch,numpy,pandas,matplotlib,sklearn; print('torch',torch.__version__); print('numpy',numpy.__version__); print('pandas',pandas.__version__); print('matplotlib',matplotlib.__version__); print('scikit-learn',sklearn.__version__); print('CUDA',torch.cuda.is_available())"
```

The repository's `requirements.txt` lists package names but does not pin version numbers. The exact versions therefore depend on the environment in which the notebooks are run; include the output above in your final environment record.

## Datasets

- **UCI HAR:** The Problem 1 notebook downloads and extracts the UCI Human Activity Recognition Using Smartphones dataset. Problem 2 reuses HAR.
- **Adding problem:** Generated in code; no external download is required.
- **Tiny Shakespeare:** The Problem 2 notebook downloads the character-level text dataset from Andrej Karpathy's `char-rnn` repository.

An internet connection is needed for the automatic dataset downloads. No pretrained weights are used.

## How to run

1. Extract the submitted project folder and open it in Jupyter Notebook, JupyterLab, Google Colab or Kaggle.
2. Select a GPU runtime when available and verify `torch.cuda.is_available()` is `True`.
3. Open `problem1/problem1.ipynb` and run all cells from top to bottom. This runs the HAR data pipeline, recurrent-cell checks, RNN/LSTM/GRU HAR comparison, adding experiments, gradient analysis, hidden-size sweep and the forget-gate-bias ablation.
4. Open `problem2/problem2.ipynb` and run all cells from top to bottom. This runs attention and encoder checks, HAR positional-encoding comparison, Tiny Shakespeare training and generation, head/mask ablations, and context-length benchmarks.
5. Check that the generated figures and `results.csv` are present and that notebook outputs are saved. The generated 500-character samples at temperatures 0.2, 0.8 and 1.5 are in `problem2/generated_samples.txt`.

**Path note:** The downloaded notebooks contain `/kaggle/working/...` output paths, particularly in Problem 2. They are directly aligned with a Kaggle Notebook environment. When using Colab or local Jupyter, those output paths must point to the current project/output directory before running. The assignment requires relative paths in the submitted notebooks; verify and resolve these environment-specific paths before final submission. Some later notebook cells reload intermediate CSV files created by earlier cells, so run the notebook sequentially rather than starting from an arbitrary cell.

## Experiment configuration and student variant

- Problem 1 recurrent models use hidden size **64**, batch size **64** for HAR and **128** for adding experiments, Adam with learning rate **1e-3**, and three seeds **0, 1, 2**. The main HAR and adding experiments use 30 epochs per run.
- Problem 1 hidden-size sweep: LSTM and GRU with hidden sizes **16, 32, 64, 128**, three seeds each.
- Problem 2 language model: context length **128**, model dimension **128**, **4** attention heads, feed-forward width **512**, **4** Transformer layers, dropout **0.1**, seed **42**, batch size **64**, learning rate **3e-4**, and **2,000** training steps. Head-count and causal-mask ablations use the same main language-model setup.
- **Individualisation:** BITS ID 2026AJ05601 ends in an odd digit (**1**). The assigned ablation is therefore the LSTM forget-gate-bias comparison, initialising the forget-gate bias to **0** versus **1** at sequence length **T = 200**. Both settings are included in the recorded results.

## Runtime and known limitations

The assignment estimates roughly 10–15 minutes for the Problem 2 language model on a T4. Extrapolating the recorded per-epoch timings for the completed Problem 1 runs gives about 5.5 hours for the main HAR, adding, hidden-size and forget-bias runs, before notebook setup, gradient/curve reruns and Problem 2 experiments. **Plan approximately 6–7 hours for a full end-to-end run on a T4; this is an estimate, not a measured single-session T4 runtime.** Actual time depends on GPU, package versions and runtime availability. The saved `results.csv` and figures are the results included with this downloaded repository.

The repository does not pin dependency versions, and its Problem 2 notebook uses Kaggle-specific absolute output paths. Confirm the environment, path portability, saved outputs and generated artifacts before submission. No additional known execution failures are documented here.
