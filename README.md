
<br />
<p align="center">
  <h1 align="center">Towards Semantic Equivalence of Tokenization in Multimodal LLM</h1>
  <p align="center">
    <a href="https://chocowu.github.io"><strong>Shengqiong Wu</strong></a>
    ·
    <a href="https://haofei.vip/"><strong>Hao Fei</strong></a>
    ·
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=xp_rICcAAAAJ&hl=zh-CN"><strong>Jiayi Ji</strong></a>
    ·
    <br/>
    <a href="https://personal.ntu.edu.sg/hanwangzhang/"><strong>Hanwang Zhang</strong></a>
    ·
    <a href="https://www.chuatatseng.com/"><strong>Tat-seng Chua</strong></a>
    ·
    <a href="https://yanshuicheng.info/"><strong>Shuicheng Yan</strong></a>
  </p>
  <p align="center" margin="0 auto">
    <small>National University of Singapore · Skywork AI, Singapore · 
    <br/> Nanyang Technological University</small>
  </p>
   <p align="center">
    <small><small>Work is done as an intern in Skywork AI, Hao Fei is the corresponding author.</small></small>
  </p>
  
  <p align="center">
    <a href='https://arxiv.org/abs/2406.05127'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
    <a href='https://chocowu.github.io/SeTok-web/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'> </a>
    <!-- <a href='https://huggingface.co/LXT/OMG_Seg' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Huggingface%20Model-8A2BE2' alt='Project Page'> </a> -->
    <!-- <a href="https://huggingface.co/spaces/LXT/OMG_Seg">
    <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-App-blue' alt='HuggingFace Model'> </a> -->
  </p>
<br />

![avatar](./assets/intro.jpeg)


### Abstract

Multimodal Large Language Models (MLLMs) have demonstrated exceptional capabilities in processing vision-language tasks. One of the crux of MLLMs lies in vision tokenization, which involves efficiently transforming input visual signals into feature representations that are most beneficial for LLMs. However, existing vision tokenizers, essential for semantic alignment between vision and language, remain problematic. Existing methods aggressively fragment visual input, corrupting the visual semantic integrity. To address this, this paper proposes a novel dynamic `Semantic-Equivalent Vision Tokenizer` (**SeTok**), which groups visual features into semantic units via a dynamic clustering algorithm, flexibly determining the number of tokens based on image complexity. The resulting vision tokens effectively preserve semantic integrity and capture both low-frequency and high-frequency visual features. The proposed MLLM (**Setokim**) equipped with SeTok significantly demonstrates superior performance across various tasks, as evidenced by our experimental results.

![avatar](./assets/framework.jpeg)


## Citation

If you use **SeTok** in your project, please kindly cite:

```bibtex
@articles{Wu2024setok,
  title={Towards Semantic Equivalence of Tokenization in Multimodal LLM},
  author={Shengqiong Wu, Hao Fei, Hanwang Zhang, Tat-Seng Chua, Shuicheng Yan},
  journal = {CoRR},
  year={2024}
}
```