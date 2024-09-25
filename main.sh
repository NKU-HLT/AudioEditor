export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1

CUDA_VISIBLE_DEVICES='0'

### delete
python main.py  --prompt "After a gunshot, there was a burst of dog barking" \
                --audio_path "audio_examples/input_audios/After a gunshot, there was a burst of dog barking.wav" \
                --token_indices "[[10,11]]" \
                --alpha "[1.,]" --cross_retain_steps "[.2,]"

### replace
python main.py  --prompt "After a thunder, there was a burst of dog barking" \
                --audio_path "audio_examples/input_audios/After a gunshot, there was a burst of dog barking.wav" \
                --token_indices "[[3]]" \
                --alpha "[-0.001,]" --cross_retain_steps "[.2,]"

### add
python main.py  --prompt "A woman is giving a speech amid applause" \
                --audio_path "audio_examples/input_audios/A woman is giving a speech.wav" \
                --token_indices "[[7,8]]" \
                --alpha "[-0.001,]" --cross_retain_steps "[.2,]"