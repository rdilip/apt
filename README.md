# Adaptive Protein Tokenization

This repository contains weights and models for our paper Adaptive Protein Tokenization. We introduce a new form of protein structure tokenization, where each token provides global, higher frequency information about a proteins structure. This departs from the existing paradigm where each token represents a local neighborhood in a protein chain.

<details>

<summary><h3>APT gone bananas</h3></summary>
<img width="1376" height="768" alt="Gemini_Generated_Image_ucikl2ucikl2ucik" src="https://github.com/user-attachments/assets/bba0b6b3-3d19-4dd3-88ee-044cbcecbb10" />
</details>

<p align="center">
  <strong>
    <a href="https://arxiv.org/abs/2602.06418">arXiv</a> •
    <a href="https://apt.rohitdilip.com/">Demo</a> •
    <a href="https://www.rohitdilip.com/research.html?post=apt">Blog Post</a>
  </strong>
</p>

## Quickstart
1. Create a Python environment (3.10+ recommended).
2. Install the package: `pip install -e .`
3. Verify checkpoint loading:
   - `python scripts/verify_checkpoints.py`
4. Example usage:
```python
from apt.models import APTLanguageModel, APTTokenizer

tokenizer = APTTokenizer.from_pretrained()
model = APTLanguageModel.from_pretrained()
```
5. Run the bundled PDB example:
   - `python scripts/run_example.py`
6. Sample a protein:
   - `python scripts/sample_generate.py`

## Checkpoints
| Name | Description | Download link |
| --- | --- | --- |
| tokenizer128.pt | Tokenizer weights trained on a maximum of 128 tokens. | [Google Drive](https://drive.google.com/file/d/1XnAs2lU4TBhgwD23th-reki8KHjUMR_G/view?usp=drive_link) |
| lm128.pt | Language model trained on tokens from tokenizer128.pt. | [Google Drive](https://drive.google.com/file/d/1tvD0q2nolfV8KZ47ktCiBZTNl5BYo489/view?usp=drive_link) |
| lm128_cond.pt | Language model trained with CATH-A level conditioning (coming soon!). |  |

Missing files are downloaded automatically from the Hugging Face Hub.


## Acknowledgments
We use code from several other works: [Kanzi](https://github.com/rdilip/kanzi), [Proteina](https://github.com/NVIDIA-Digital-Bio/proteina), and [TorchCFM](https://github.com/atong01/conditional-flow-matching). If you have questions, reach out to `rdilip@caltech.edu`. 

```
@article{dilip2026adaptive,
  title={Adaptive Protein Tokenization},
  author={Dilip, Rohit and Varshney, Ayush and Van Valen, David},
  journal={arXiv preprint arXiv:2602.06418},
  year={2026}
}
```
