# LOGO: A Long-Form Video Dataset for Group Action Quality Assessment

Created by Shiyi Zhang, Wenxun Dai, Sujia Wang, Xiangwei Shen, Jiwen Lu, Jie Zhou, Yansong Tang

This repository contains the LOGO dataset and PyTorch implementation for GOAT. (CVPR 2023)

[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_LOGO_A_Long-Form_Video_Dataset_for_Group_Action_Quality_Assessment_CVPR_2023_paper.pdf) [[Dataset]](https://pan.baidu.com/s/1GNi_ZcbSq6oi2SEX_iuFwA?pwd=v329) (extract number: v329) 

## Dataset

### TODO

- [x] Release the dataset
- [x] The code of GOAT
- [ ] Pretrained features for LOGO

### Lexicon

We construct a fine-grained video dataset organized by temporal structure, which contains action and formation manual annotations. Herein, we design the labeling system with professional artistic swimming athletes to construct a lexicon for annotation, considering FINA rules and the actual scenario of the competitions. In the *Technical* event, the group size is eight people, the video length is $170±15s$, and the actions include *Upper*, *Lower*, *Float*, *None*, *Acrobatic*, *Cadence*, and five *Required Elements*. Each competition cycle needs to complete five *Required Elements*, at least two *Acrobatic* movements, and at least one *Cadence* action. In the *Free* events, there are 8 people, the video length is $240±15s$, and the actions include *Upper*, *Lower*, *Float*, *None*, *Acrobatic*, *Cadence*, and *Free* elements. When performing *Required*, *Upper*, *Lower*, and *Float*, the athletes form neat polygons.

### Annotation

Given an RGB artistic swimming video, the annotator utilizes our defined lexicon to label each frame with its action and formation. We accomplish the 25fps frame-wise **action annotation** stage utilizing the [COIN Annotation Toolbox](https://github.com/coin-dataset/annotation-tool) and the 1fps frame-wise **formation labels** using [Labelme](https://github.com/wkentaro/labelme). Specifically, we set strict rules defining the boundaries between artistic swimming sequences and the formation marking position and employ eight workers with prior knowledge in the artistic swimming domain to label the dataset frame by frame following the rules. The annotation results of one worker are checked and adjusted by another, which ensures annotation results are double-checked. 

The annotation information is saved in [[Baidu Drive]](https://pan.baidu.com/s/1UwlGzCeq_UjY0GbOnaHXxw?pwd=ojgf) (extract number: ojgf)

The annotation information contained in `anno dict.pkl` for each sample is:

| List Num. | Type   | Description                       | Example                       |
| --------- | ------ | --------------------------------- | ----------------------------- |
| `0`       | string | Event type.                       | 'tech'                        |
| `1`       | float  | The score of the video.           | 90.25                         |
| `2`       | float  | /                                 | /                             |
| `3`       | list   | End frame of the action instance. | [76, 141, 187, 246, 263, ···] |
| `4`       | list   | Action type of each frame.        | [12, 12, 12, 12, 12, ···]     |

### Statistics

The LOGO dataset consists of 200 video samples from 26 events with 204.2s average duration and above 11h total duration, covering 3 annotation types, 12 action types, and 17 formation types.

### Download

- Video_Frames:  [[Baidu Drive]](https://pan.baidu.com/s/1GNi_ZcbSq6oi2SEX_iuFwA?pwd=v329) (extract number: v329) 
- Annotations and Split: [[Baidu Drive]](https://pan.baidu.com/s/1UwlGzCeq_UjY0GbOnaHXxw?pwd=ojgf) (extract number: ojgf)

## Code for Group-aware Attention (GOAT)

### Requirement

- torch_videovision

```
pip install git+https://github.com/hassony2/torch_videovision
```

### Data Preperation

- The prepared dataset ([[Baidu Drive]](https://pan.baidu.com/s/1GNi_ZcbSq6oi2SEX_iuFwA?pwd=v329) (extract number: v329) ) and annotations ([[Baidu Drive]](https://pan.baidu.com/s/1UwlGzCeq_UjY0GbOnaHXxw?pwd=ojgf) (extract number: ojgf)) are already provided in this repo. 

- The data structure should be:

```
$DATASET_ROOT
├── LOGO
|  ├── WorldChampionship2019_free_final
|     ├── 0
|        ├── 00000.jpg
|        ...
|        └── 06249.jpg
|     ...
|     └── 11
|        ├── 00000.jpg
|        ...
|        └── 06249.jpg
|  ...
|  └── WorldChampionship2022_free_final
|     ├── 0
|     ...
|     └── 7 
└──
```

### Pretrain Model

The Kinetics pretrained I3D downloaded from the reposity [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth)

```
model_rgb.pth
```

### Training

```
srun -p mm_human --job-name co --quotatype=reserved --ntasks-per-node=1 --ntasks=1 --gres=gpu:0 --cpus-per-task=24 bash ./scripts/train.sh MTL res --use_goat=1 
```

**Contact:** [shiyi-zh19@mails.tsinghua.edu.cn](mailto:shiyi-zh19@mails.tsinghua.edu.cn)
