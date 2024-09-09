import torch
from msclap import CLAP

# Load model (Choose between versions '2022' or '2023')
# The model weight will be downloaded automatically if `model_fp` is not specified
clap_model = CLAP(version = '2023', model_fp = 'CLAP_weights_2023.pth', use_cuda = False)

# Extract text embeddings
text_embeddings = clap_model.get_text_embeddings(["A baby and a woman laugh"])

# Extract audio embeddings
audio_embeddings = clap_model.get_audio_embeddings(["A baby and a woman laugh.wav"])

# cosine_similarity = torch.nn.functional.cosine_similarity(audio_embeddings, text_embeddings)
# print("Cosine Similarity:", cosine_similarity.item())