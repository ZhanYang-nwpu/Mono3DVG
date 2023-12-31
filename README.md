# <img src="images/logo_Mono3DVG.png" height="40"> Mono3DVG: 3D Visual Grounding in Monocular Images
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

##### Author: Yang Zhan, Yuan Yuan, Zhitong Xiong
This is the official repository for paper **"Mono3DVG: 3D Visual Grounding in Monocular Images"**. [paper](https://arxiv.org/pdf/2312.08022.pdf)

The paper has been accepted by **AAAI 2024** 🎉.

**School of Artificial Intelligence, OPtics, and ElectroNics (iOPEN), Northwestern Polytechnical University**
## Please share a <font color='orange'>STAR ⭐</font> if this project does help


## 📢 Latest Updates
- **Dec-09-2023**: Mono3DVG paper is accepted by AAAI2024. 🔥🔥
- **Dec-29-2023**: Mono3DRefer dataset is released. 🔥🔥
- 📦 Codebase and models coming soon! 🚀
---

## 💬 Mono3DVG: 3D Visual Grounding in Monocular Images
We introduce a novel task of 3D visual grounding in monocular RGB images using descriptions with appearance and geometry information, termed Mono3DVG.
Mono3DVG aims to localize the true 3D extent of referred objects in an image using language descriptions with geometry information.
<div align="center">
  <img src="images/task.png"/>
</div>

## 🌋 Dataset
Download our Mono3DRefer dataset. We build the first dataset for Mono3DVG, termed Mono3DRefer, which can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1ICBv0SRbRIUnl_z8DVuH8lz7KQt580EI?usp=drive_link). The download link is available below:
```
https://drive.google.com/drive/folders/1ICBv0SRbRIUnl_z8DVuH8lz7KQt580EI?usp=drive_link
```

<div align="center">
  <img src="images/5-dataset.png"/>
</div>


## <img src="images/logo_Mono3DVG.png" height="20"> Mono3DVG-TR: Architecture

Mono3DVG-TR is the **first end-to-end transformer-based network** for monocular 3D visual grounding. 
<div align="center">
  <img src="images/model.png"/>
</div>


## 👁️ Visualization

### 1. Qualitative results from baseline methods and our Mono3DVG-TR.

<div align="center">
  <img src="images/1-results.png"/>
</div>
Fig.1 Blue, green, and red boxes denote the ground truth, prediction with IoU higher than 0.5, and prediction with IoU lower than 0.5, respectively.

### 2. Qualitative results of our Mono3DVG-TR.
<div align="center">
  <img src="images/results_maps.png"/>
</div>
Fig.2 Visualization of ’000152.png’ image’s localization results, the depth predictor’s depth maps, and the text-guided
adapter’s attention score maps for our Mono3DVG-TR.

### 3. Qualitative results in the ’unique’(top) and ’multiple’(bottom) subsets.
<div align="center">
  <img src="images/2-results.png"/>
</div>
<div align="center">
  <img src="images/2-results2.png"/>
</div>
Fig.3 The gray block is the traditional query without specific geometry information.

### 4. Qualitative results in the ’near’, ’medium’, and ’far’ subsets.
<div align="center">
  <img src="images/3-results.png"/>
</div>
<div align="center">
  <img src="images/3-results2.png"/>
</div>
Fig.4 The gray block is the traditional query without specific geometry information.

### 5. Qualitative results in the ’easy’, ’moderate’, and ’hard’ subsets.
<div align="center">
  <img src="images/4-results.png"/>
</div>
<div align="center">
  <img src="images/4-results2.png"/>
</div>
Fig.5 The gray block is the traditional query without specific geometry information.


## 🔍 Results

### 1. Comparison with baselines.

<div align="center">
  <img src="images/result_table1.png"/>
</div>

### 2. Results for ’near’-’medium’-’far’ subsets and ’easy’-’moderate’-’hard’ subsets.
<div align="center">
  <img src="images/result_table2.png"/>
</div>



## 📜 Citation
```bibtex
@misc{zhan2023mono3dvg,
      title={Mono3DVG: 3D Visual Grounding in Monocular Images}, 
      author={Yang Zhan and Yuan Yuan and Zhitong Xiong},
      year={2023},
      eprint={2312.08022},
      archivePrefix={arXiv}
}
```

## 🙏 Acknowledgement
Our code is based on (ICCV 2023)[MonoDETR](https://github.com/ZrrSkywalker/MonoDETR). We sincerely appreciate their contributions and authors for releasing source codes. I would like to thank Xiong zhitong and Yuan yuan for helping the manuscript. I also thank the School of Artificial Intelligence, OPtics, and ElectroNics (iOPEN), Northwestern Polytechnical University for supporting this work.

## 🤖 Contact
If you have any questions about this project, please feel free to contact zhanyangnwpu@gmail.com.
