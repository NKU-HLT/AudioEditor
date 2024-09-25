# AudioEditor: A Training-Free Diffusion-Based Audio Editing Framework

[Demo Page](https://kiri0824.github.io/AudioEditor-demo-page/) | [Arxiv Paper](https://arxiv.org/pdf/2409.12466)

Diffusion-based text-to-audio (TTA) generation has made substantial progress, leveraging Latent Diffusion Model (LDM) to produce high-quality, diverse and instruction-relevant audios. However, beyond generation, the task of audio editing remains equally important but has received comparatively little attention. Audio editing tasks face two primary challenges: executing precise edits and preserving the unedited sections. While workflows based on LDMs have effectively addressed these challenges in the field of image processing, similar approaches have been scarcely applied to audio editing. In this paper, we introduce AudioEditor, a training-free audio editing framework built on the pretrained diffusion-based TTA model. AudioEditor incorporates Null-text Inversion and EOT-Suppression methods, enabling the model to preserve original audio features while executing accurate edits. Comprehensive objective and subjective experiments validate the effectiveness of AudioEditor in delivering high-quality audio edits.

<div align="center">
  <img src="docs/workflow.png" alt="Workflow" width="750"/>
</div>

## 🚀 Features

- **Pre-trained Model: Auffusion**  
  We use the pre-trained model Auffusion for audio editing tasks.  
  [Auffusion Repository](https://github.com/happylittlecat2333/Auffusion/tree/main) | [Model Download Link](https://huggingface.co/auffusion/auffusion)
- **Null-text Inversion**: Ensures preservation of unedited audio portions during the editing process.
- **EOT-suppression**: Enhance the model's ability to preserve original audio
features and improve editing capabilities.
- **Support multiple audio editing operations**: Add, Delete and Replace.
- **Easy integration with other TTA models**: Plug and play with existing TTA diffuser-based models.

## 📀 Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/YuuhangJia/AudioEditor.git
    cd AudioEditor
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up additional environment variables or configurations (if any):

    ```bash
    export YOUR_ENV_VAR=your_value
    ```

## ⭐ Usage
We will release the code for the main methods proposed in our paper after its acceptance immediately!  

### 1️⃣ Delete


<img src="docs/delete.png" alt="Delete/Add Workflow" width="500"/>

To run the deletion on an audio with a simple example, use the following command:
```bash
python main.py  --prompt "After a gunshot, there was a burst of dog barking" \
                --audio_path "audio_examples/input_audios/After a gunshot, there was a burst of dog barking.wav" \
                --token_indices "[[10,11]]" \
                --alpha "[1.,]" --cross_retain_steps "[.2,]"
```

### 2️⃣ Replace

<img src="docs/replacement.png" alt="Delete/Add Workflow" width="500"/>

To run the Replacement on an audio with a simple example, use the following command:
```bash
python main.py  --prompt "After a thunder, there was a burst of dog barking" \
                --audio_path "audio_examples/input_audios/After a gunshot, there was a burst of dog barking.wav" \
                --token_indices "[[3]]" \
                --alpha "[-0.001,]" --cross_retain_steps "[.2,]"
```

### 3️⃣ Add
<img src="docs/add.png" alt="Delete/Add Workflow" width="500"/>

```bash
python main.py  --prompt "A woman is giving a speech amid applause" \
                --audio_path "audio_examples/input_audios/A woman is giving a speech.wav" \
                --token_indices "[[7,8]]" \
                --alpha "[-0.001,]" --cross_retain_steps "[.2,]"
```
## 📐 Quantitative comparison
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>

<body>
    <table border="1" style="width: 100%;margin: 0 auto;">
        <caption><strong>Objective Evaluation Results</strong></caption>
        <thead>
            <tr>
                <td style="text-align: center;" rowspan="2"><strong>Edit_Models</strong></td>
                <td style="text-align: center;" rowspan="2"><strong>Edit_Type</strong></td>
                <td style="text-align: center;" colspan="2"><strong>Overall Quality</strong></td>
                <td style="text-align: center;" colspan="3"><strong>Similarity with (Regenerated_wavs)</strong></td>
                <td style="text-align: center;" colspan="3"><strong>Similarity with (Original_wavs)</strong></td>
            </tr>
            <tr>
                <td style="text-align: center;" ><em><strong>Clap↑</strong></em></td>
                <td style="text-align: center;" ><em><strong>IS ↑</strong></em></td>
                <td style="text-align: center;" ><em><strong>FD ↓</strong></em></td>
                <td style="text-align: center;" ><em><strong>FAD ↓</strong></em></td>
                <td style="text-align: center;" ><em><strong>KL ↓</strong></em></td>
                <td style="text-align: center;" ><em><strong>FD ↓</strong></em></td>
                <td style="text-align: center;" ><em><strong>FAD ↓</strong></em></td>
                <td style="text-align: center;" ><em><strong>KL ↓</strong></em></td>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="text-align: center;"  rowspan="4"><strong>Original_wavs</strong></td>
                <td style="text-align: center;" >add</td>
                <td style="text-align: center;" >51.4%</td>
                <td style="text-align: center;" >5.64</td>
                <td style="text-align: center;" >44.71</td>
                <td style="text-align: center;" >5.28</td>
                <td style="text-align: center;" >1.78</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
            </tr>
            <tr>
                <td style="text-align: center;" >delete</td>
                <td style="text-align: center;" >51.5%</td>
                <td style="text-align: center;" >4.26</td>
                <td style="text-align: center;" >51.82</td>
                <td style="text-align: center;" >6.16</td>
                <td style="text-align: center;" >1.85</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
            </tr>
            <tr>
                <td style="text-align: center;" >replace</td>
                <td style="text-align: center;" >41.6%</td>
                <td style="text-align: center;" >4.41</td>
                <td style="text-align: center;" >69.92</td>
                <td style="text-align: center;" >7.88</td>
                <td style="text-align: center;" >4.56</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
            </tr>
            <tr>
                <td style="text-align: center;" ><em><strong>Average</strong></em></td>
                <td style="text-align: center;" >48.2%</td>
                <td style="text-align: center;" >4.77</td>
                <td style="text-align: center;" >55.48</td>
                <td style="text-align: center;" >6.45</td>
                <td style="text-align: center;" >2.73</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
            </tr>
            <tr>
                <td style="text-align: center;"  rowspan="4"><strong>Regenerated_wavs</strong></td>
                <td style="text-align: center;" >add</td>
                <td style="text-align: center;" >59.7%</td>
                <td style="text-align: center;" >5.96</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >44.71</td>
                <td style="text-align: center;" >5.28</td>
                <td style="text-align: center;" >1.36</td>
            </tr>
            <tr>
                <td style="text-align: center;" >delete</td>
                <td style="text-align: center;" >59.1%</td>
                <td style="text-align: center;" >4.47</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >51.82</td>
                <td style="text-align: center;" >6.16</td>
                <td style="text-align: center;" >2.39</td>
            </tr>
            <tr>
                <td style="text-align: center;" >replace</td>
                <td style="text-align: center;" >58.9%</td>
                <td style="text-align: center;" >5.13</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >69.92</td>
                <td style="text-align: center;" >7.88</td>
                <td style="text-align: center;" >4.09</td>
            </tr>
            <tr>
                <td style="text-align: center;" ><em><strong>Average</strong></em></td>
                <td style="text-align: center;" ><strong>59.2%</strong></td>
                <td style="text-align: center;" >5.19</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >-</td>
                <td style="text-align: center;" >55.48</td>
                <td style="text-align: center;" >6.45</td>
                <td style="text-align: center;" >2.61</td>
            </tr>
            <tr>
                <td style="text-align: center;"  rowspan="4"><strong>SDEdit(baseline)</strong></td>
                <td style="text-align: center;" >add</td>
                <td style="text-align: center;" >58.4%</td>
                <td style="text-align: center;" >6.36</td>
                <td style="text-align: center;" >27.89</td>
                <td style="text-align: center;" >2.74</td>
                <td style="text-align: center;" >0.79</td>
                <td style="text-align: center;" >36.74</td>
                <td style="text-align: center;" >3.08</td>
                <td style="text-align: center;" >1.08</td>
            </tr>
            <tr>
                <td style="text-align: center;" >delete</td>
                <td style="text-align: center;" >53.3%</td>
                <td style="text-align: center;" >5.31</td>
                <td style="text-align: center;" >55.12</td>
                <td style="text-align: center;" >6.65</td>
                <td style="text-align: center;" >1.78</td>
                <td style="text-align: center;" >40.43</td>
                <td style="text-align: center;" >6.95</td>
                <td style="text-align: center;" >0.88</td>
            </tr>
            <tr>
                <td style="text-align: center;" >replace</td>
                <td style="text-align: center;" >58.6%</td>
                <td style="text-align: center;" >4.99</td>
                <td style="text-align: center;" >29.76</td>
                <td style="text-align: center;" >3.24</td>
                <td style="text-align: center;" >0.80</td>
                <td style="text-align: center;" >55.21</td>
                <td style="text-align: center;" >7.00</td>
                <td style="text-align: center;" >3.40</td>
            </tr>
            <tr>
                <td style="text-align: center;" ><em><strong>Average</strong></em></td>
                <td style="text-align: center;" >56.8%</td>
                <td style="text-align: center;" ><strong>5.55</strong></td>
                <td style="text-align: center;" ><strong>37.59</strong></td>
                <td style="text-align: center;" >4.21*</td>
                <td style="text-align: center;" >1.12*</td>
                <td style="text-align: center;" >44.13*</td>
                <td style="text-align: center;" >5.68*</td>
                <td style="text-align: center;" ><strong>1.79</strong></td>
            </tr>
            <tr>
                <td style="text-align: center;"  rowspan="4"><strong>AudioEditor(ours)</strong></td>
                <td style="text-align: center;" >add</td>
                <td style="text-align: center;" >59.4%</td>
                <td style="text-align: center;" >6.16</td>
                <td style="text-align: center;" >27.83</td>
                <td style="text-align: center;" >2.41</td>
                <td style="text-align: center;" >0.85</td>
                <td style="text-align: center;" >40.00</td>
                <td style="text-align: center;" >3.52</td>
                <td style="text-align: center;" >1.27</td>
            </tr>
            <tr>
                <td style="text-align: center;" >delete</td>
                <td style="text-align: center;" >54.1%</td>
                <td style="text-align: center;" >4.75</td>
                <td style="text-align: center;" >52.56</td>
                <td style="text-align: center;" >5.02</td>
                <td style="text-align: center;" >1.54</td>
                <td style="text-align: center;" >37.16</td>
                <td style="text-align: center;" >4.91</td>
                <td style="text-align: center;" >1.05</td>
            </tr>
            <tr>
                <td style="text-align: center;" >replace</td>
                <td style="text-align: center;" >58.1%</td>
                <td style="text-align: center;" >5.14</td>
                <td style="text-align: center;" >28.80</td>
                <td style="text-align: center;" >3.34</td>
                <td style="text-align: center;" >0.79</td>
                <td style="text-align: center;" >59.46</td>
                <td style="text-align: center;" >7.52</td>
                <td style="text-align: center;" >3.73</td>
            </tr>
            <tr>
                <td style="text-align: center;" ><em><strong>Average</strong></em></td>
                <td style="text-align: center;" >57.6%*</td>
                <td style="text-align: center;" >5.19*</td>
                <td style="text-align: center;" >37.63*</td>
                <td style="text-align: center;" ><strong>3.27</strong></td>
                <td style="text-align: center;" ><strong>1.07</strong></td>
                <td style="text-align: center;" ><strong>43.48</strong></td>
                <td style="text-align: center;" ><strong>4.95</strong></td>
                <td style="text-align: center;" >1.93*</td>
            </tr>
        </tbody>
    </table>
</body>
* indicates a suboptimal value, which may represent more desirable than optimal one in certain metrics.

</html>

## 🤝🏻 Contact
Should you have any questions, please contact 2120240729@mail.nankai.edu.cn

## 📚 Citation
Coming soon.

## 🐍 License
The code in this repository is licensed under the MIT License for academic and other non-commercial uses.

[//]: # (For commercial use of the code and models, separate commercial licensing is available. Please contact authors.)

![visitors](https://visitor-badge.laobi.icu/badge?page_id=sen-mao/SuppressEOT)

## 🙏 Acknowledgment:
This code is based on the [P2P, Null-text](https://github.com/google/prompt-to-prompt) , [SuppressEOT](https://github.com/sen-mao/SuppressEOT) and [Auffusion](https://github.com/happylittlecat2333/Auffusion) repositories. 
