# wSignGen: Word-Conditioned 3D American Sign Language Motion Generation


[EMNLP 2024 Paper Link](https://aclanthology.org/2024.findings-emnlp.584)


The official PyTorch implementation of the paper [**"Word-Conditioned 3D American Sign Language Motion Generation"**](https://aclanthology.org/2024.findings-emnlp.584/).

Please visit our [**webpage**](https://dongludeeplearning.github.io/wSignGen.html) for more details.



#### Bibtex
If you find this code useful in your research or use our dataset, please cite:

```
@inproceedings{dong2024word,
  title={Word-Conditioned 3D American Sign Language Motion Generation},
  author={Dong, Lu and Wang, Xiao and Nwogu, Ifeoma},
  year={2024},
  organization={Association for Computational Linguistics}
}


@inproceedings{dong2024signavatar,
  title={Signavatar: Sign language 3d motion reconstruction and generation},
  author={Dong, Lu and Chaudhary, Lipisha and Xu, Fei and Wang, Xiao and Lary, Mason and Nwogu, Ifeoma},
  booktitle={2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG)},
  pages={1--10},
  year={2024},
  organization={IEEE}
}
```
## Getting started

This code was tested on `"Ubuntu 20.04.5 LTS"` and requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

## 1. Setup environment

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

## Download SMPLX Models and Datasets

```bash
Required files in body_models/smplx/:
 - SMPLX_NEUTRAL.npz files 
 - kin_pose53_smplx.pkl 

# Download SMPLX_NEUTRAL from: https://smpl-x.is.tue.mpg.de/
# kin_pose53_smplx.pkl file have been updated
```

## 2. Get data and pretrained models

Check the follow link to get the dataset:
[wSignGen dataset google drive](https://drive.google.com/drive/folders/1pncvvaxr1UXPBg6ewG225wrPoGEZbTd1?usp=sharing) 


Download the pre-trained diffusion model, then unzip and place them in `./save/`. 
[wSignGen diffusion pre-trained google drive](https://drive.google.com/drive/folders/1ytuImcAKg78WdnPo5NQRbbP6_ns1hooW?usp=sharing) 

The recognition model (STGCN checkpoint) have been uploaded in assets/actionrecognition/

If you want to know more details about the STGCN model training, please check the the following [link:](https://github.com/dongludeeplearning/SignAvatar) 

## 3. Sign Motion Synthesis

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



## 5. Train your own wSignGen


```shell
bash run_train_mdm100.sh
```


## 6. Evaluate


```shell
python -m eval.eval_wsigngen --model ./save/wlasl100_ckpt_final/model000400000.pt --eval_mode full --batch_size 128 
```


## License
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

Note that our code depends on other libraries, including MDM, CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
