# Discriminative Multi-modality Speech Recognition
In this paper, we propose a two-stage speech recognition model. In the first stage, the target voice is separated from background noises with help from the corresponding visual information of lip movements, making the model ‘listen' clearly. At the second stage, the audio modality combines visual modality again to better understand the speech by a MSR sub-network, further improving the recognition rate.
## Paper
[Paper(Arxiv)](http://arxiv.org/abs/2005.05592)
## Preparation
First of all, clone the code
```
git clone https://github.com/JackSyu/AE-MSR.git
```
Then, create a folder:
```
cd AE-MSR && mkdir data
```
### Requirement
```
Python 3.5
Tensorflow 1.12.0.
CUDA 9.0 or higher. 
MATLAB (optionally）
```
### Data preprocessing
LRS3:<br>
Download the data.<br>
Extract the video frames and crop lip area.<br>
```
cd preprocessing
python dataset_tfrecord_trainval.py
```
## Training & Testing
We train the audio enhancement sub-network and the MSR sub-network separately.
```
python Train_Audio_Visual_Speech_Enhancement.py
python Train_Audio_Visual_Speech_Recognition.py
```
Then we freeze the AE sub-network and complete the subsequent joint training.
```
python Train_AE_MSR.py
Python Test_AE_MSR.py
```
## Citation
If you find our code useful, please consider citing:
```
@article{xu2020dmsr,
  title={Discriminative Multi-modality Speech Recognition},
  author={Xu, Bo and Lu, Cheng and Guo, Yandong and Jacob Wang},
  journal={arXiv preprint arXiv:2005.05592},
  year={2020}
}
```
