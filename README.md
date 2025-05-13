# wSignGen: Word-Conditioned 3D American Sign Language Motion Generation


[EMNLP 2024 Paper Link](https://aclanthology.org/2024.findings-emnlp.584)


The official PyTorch implementation of the paper [**"Word-Conditioned 3D American Sign Language Motion Generation"**](https://aclanthology.org/2024.findings-emnlp.584/).

Please visit our [**webpage**](https://dongludeeplearning.github.io/wSignGen.html) for more details.



#### Bibtex
If you find this code useful in your research, please cite:

```
@inproceedings{dong-etal-2024-word,
    title = "Word-Conditioned 3{D} {A}merican {S}ign {L}anguage Motion Generation",
    author = "Dong, Lu  and Wang, Xiao  and Nwogu, Ifeoma",
    editor = "Al-Onaizan, Yaser  and  Bansal, Mohit  and Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.584/",
    doi = "10.18653/v1/2024.findings-emnlp.584",
    pages = "9993--9999"
}
```


## Getting started

This code was tested on `"Ubuntu 20.04.5 LTS` and requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.

Setup conda env:
```shell
conda env create -f environment.yml
conda activate wsigngen
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:

<details>
  <summary><b>Text to Motion</b></summary>

```bash
bash prepare/download_smplx_files.sh
```
</details>


### 2. Get data


Check the follow link to get the data:
[wSignGen dataset google drive](https://drive.google.com/drive/folders/1coa5-DuI3foXNyKAeu28bL3tZ0J4XG9r?usp=sharing) 



### 3. Download the pretrained models

Download the model(s) you wish to use, then unzip and place them in `./save/`. 



## Sign Motion Synthesis


### Generate from test set prompts

```shell
python -m sample.generate --model_path ./save/wlasl100_ckpt_final/model000400000.pt --num_samples 10 --num_repetitions 3
```

### Generate from your text file

```shell
python -m sample.generate --model_path ./save/wlasl100_ckpt_final/model000400000.pt --input_text ./assets/sign_words.txt
```

### Generate a single prompt

```shell
python -m sample.generate --model_path ./save/wlasl100_ckpt_final/model000400000.pt --text_prompt "paper"
```



**You may also define:**
* `--device` id.
* `--seed` to sample different prompts.
* `--motion_length` (text-to-motion only) in seconds (maximum is 9.8[sec]).

**Running those will get you:**

* `results.npy` file with text prompts and xyz positions of the generated animation
* `sample##_rep##.mp4` - a stick figure animation for each generated motion.


You can stop here, or render the SMPL mesh using the following script.

### Render SMPLX mesh

To create SMPL mesh per frame run:

```shell
python -m visualize.render_mesh --input_path /path/to/mp4/stick/figure/file
```

**This script outputs:**
* `sample##_rep##_smplx_params.npy` - SMPLX parameters (thetas, root translations, vertices and faces)
* `sample##_rep##_obj` - Mesh per frame in `.obj` format.

**Notes:**
* The `.obj` can be integrated into Blender/Maya/3DS-MAX and rendered using them.
* This script is running [SMPLify](https://smplify.is.tue.mpg.de/) and needs GPU as well (can be specified with the `--device` flag).
* **Important** - Do not change the original `.mp4` path before running the script.

**Notes for 3d makers:**
* You have two ways to animate the sequence:
  1. Use the [SMPLX add-on](https://github.com/Meshcapade/SMPL_blender_addon) and the theta parameters saved to `sample##_rep##_smplx_params.npy` (we always use beta=0 and the gender-neutral model).
  1. A more straightforward way is using the mesh data itself. All meshes have the same topology (SMPLX), so you just need to keyframe vertex locations. 
     Since the OBJs are not preserving vertices order, we also save this data to the `sample##_rep##_smpl_params.npy` file for your convenience.



## Train your own wSignGen


```shell
bash run_train_mdm100.sh
```


## Evaluate


```shell
python -m eval.eval_wsigngen --model ./save/wlasl100_ckpt_final/model000400000.pt --eval_mode full --batch_size 128 
```


## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including MDM, CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
