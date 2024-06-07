import torch
import json
from one_peace.models import from_pretrained
from tqdm import tqdm
import pandas as pd
import os

id2label = json.load(open("/Users/jaeyoungshin/Desktop/dcase2024/ONE-PEACE/assets/vggsound_id2label.json"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = from_pretrained(
#   "ONE-PEACE_VGGSound",
    model_name_or_path="/Users/jaeyoungshin/Desktop/dcase2024/checkpoints/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt",
    model_type="one_peace_retrieval",
    device=device,
    dtype="float32"
)

# process audio
# audio_list = ["/Users/jaeyoungshin/Desktop/dcase2024/ONE-PEACE/assets/cow.flac", "/Users/jaeyoungshin/Desktop/dcase2024/ONE-PEACE/assets/dog.flac"]
# metadata = "/Volumes/One_Touch/dcase2024/retreival/evaluation_dataset/retrieval_audio_metadata.csv"
# df = pd.read_csv(metadata)
# audio_files = df['file_name'].tolist()

audio_dir = "/Volumes/One_Touch/dcase2024/retreival/evaluation_dataset/test"
audio_files = os.listdir(audio_dir)[:50] + ['people_sitting_by_fire.wav']
# audio_files = ["Tool Box.wav",  "people_sitting_by_fire.wav", "Outside UU FSW 080701.wav","3overtones.wav", "1_ambiente inicio.wav", "2_frogs_qinghua.wav", "02-BikeDemo_Chimes_Speak.wav", "5_Navy-Blue-Angels-jets_CLIP.wav", "5. Apartment ambience inside room.wav"]
audio_list = [os.path.join(audio_dir, x) for x in audio_files if not x.startswith('._')]
# print('audio_list:',audio_list)
src_audios, audio_padding_masks = model.process_audio(audio_list)

# Define the text query
# text_query = ['The person is rummaging through the pans while looking for something.']
text_query = ['Children and Adults talk with each other positively and laugh.']
# Process text to obtain text features
text_tokens = model.process_text(text_query)


if __name__ == '__main__':
    with torch.no_grad():
        # Extract audio features
        text_features = model.extract_text_features(text_tokens)
        audio_features = model.extract_audio_features(src_audios, audio_padding_masks)
        
        # Compute similarity scores between text features and each audio
        similarity_scores = audio_features @ text_features.T
        similarity_scores = similarity_scores.squeeze()

        # Retrieve audio files based on highest similarity scores
        top_audio_indices = torch.topk(similarity_scores, k=10).indices  # Adjust k based on how many top files you want
        print('top_audio_indices:',top_audio_indices)

        top_audio_indices = top_audio_indices.cpu().numpy().tolist()  # Convert to a list of indices


    # Print out the top k relevant audio files
    for idx in top_audio_indices:
        print('Relevant audio file:', audio_list[idx])
