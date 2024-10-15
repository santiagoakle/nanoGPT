
Please take a moment to read this text in full. It may save you time.

This code is an unofficial implementation of the Normalized Transformer as described in the paper:

"nGPT: Normalized Transformer with Representation Learning on the Hypersphere"
by Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, and Boris Ginsburg.
https://arxiv.org/abs/2410.01131

This codebase is built upon the MIT-licensed project nanoGPT by Andrej Karpathy.
We strongly recommend becoming familiar with nanoGPT first, as it has numerous resolved and open issues. Once you understand nanoGPT, using this project will be much easier.

The main modification in this project is in model.py, where we implement both the original and the normalized Transformers. The train.py implements the normalization procedure. The architecture follows the one described in the paper, with the exception of the vocabulary size. You will need to use nanoGPT to generate the data folder using OpenWebText.

The code reproduces our original experiments, which were conducted using NVIDIA's internal libraries. Specifically, we observe the reported 4x speedup for a sequence length of 1k and a 10x speedup for a sequence length of 4k. We tested only the 0.5B model, as the paper suggests that the speedup numbers for the 0.5B and 1B models are very similar. The speedup factor also depends on the training budget; the longer the run, the greater the speedup, meaning the reported numbers can be exceeded.

One main observable difference is that the baseline GPT in nanoGPT diverges with larger learning rates. However, this does not occur with our internal code. As a result, the optimal learning rates differ slightly. I have also included a MATLAB file to reproduce the attached figure which contains the final validation loss values for various hyperparameter settings. These results may vary depending on the GPU used. In our case, all experiments were conducted on 64 GPUs in parallel. 

Please note that this is an unofficial release, and I do not plan to modify the code unless a major bug is found. Like nanoGPT, it is beneficial to keep the code stable so that it can serve as a consistent reference implementation.

This implementation is not optimized for memory or compute performance. The paper suggests that nGPT can be simplified in various ways, sometimes without any loss in performance.

Many thanks to Andrej Karpathy, whose nanoGPT library helped me gain a better understanding of how Transformers work.



