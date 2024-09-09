# AudioEditor: A Training-Free Diffusion-Based Audio Editing Framework

[Demo Page](todo)

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
