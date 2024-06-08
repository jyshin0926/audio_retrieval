import torch
import json
from one_peace.models import from_pretrained
from tqdm import tqdm
import pandas as pd
import os

# Initialize device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = from_pretrained(
    # model_name_or_path="../checkpoints/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt",
    model_name_or_path="/root/jaeyoung/checkpoints/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt",
    model_type="one_peace_retrieval",
    device=device,
    dtype="float32"
)

# Load captions and prepare audio files
# captions_path = "/Volumes/One_Touch/dcase2024/retreival/evaluation_dataset/retrieval_captions.csv"
# audio_dir = "/Volumes/One_Touch/dcase2024/retreival/evaluation_dataset/test"
captions_path = "/root/jaeyoung/data/retrieval_captions.csv"
audio_dir = "/root/jaeyoung/data/test"

df = pd.read_csv(captions_path)
text_queries = df['caption'].tolist()
audio_files = os.listdir(audio_dir)
audio_list = [os.path.join(audio_dir, x) for x in audio_files if not x.startswith('._')]
src_audios, audio_padding_masks = model.process_audio(audio_list)

# Process all text queries to obtain text features
text_tokens = model.process_text(text_queries)

# Prepare results dataframe
results_df = pd.DataFrame(columns=['caption', 'fname_1', 'fname_2', 'fname_3', 'fname_4', 'fname_5', 'fname_6', 'fname_7', 'fname_8', 'fname_9', 'fname_10'])

if __name__ == '__main__':
    with torch.no_grad():
        # Extract text and audio features
        text_features = model.extract_text_features(text_tokens)
        audio_features = model.extract_audio_features(src_audios, audio_padding_masks)
        
        # Compute similarity scores between all text features and all audio features
        similarity_scores = torch.matmul(audio_features, text_features.T)
        
        # For each text query, find the top matching audio files
        for text_idx, single_text_features in enumerate(text_features):
            single_similarity_scores = similarity_scores[:, text_idx]
            top_audio_indices = torch.topk(single_similarity_scores, k=10).indices
            top_audio_indices = top_audio_indices.cpu().numpy().tolist()
            
            # Retrieve the filenames of the top matching audio files
            top_files = [audio_list[idx] for idx in top_audio_indices]
            results_df.loc[len(results_df)] = [text_queries[text_idx]] + top_files

# Save results to CSV
results_csv_path = '/root/jaeyoung/submission/onepeace_retrieval_results.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = from_pretrained(
# #   "ONE-PEACE_VGGSound",
#     model_name_or_path="/Users/jaeyoungshin/Desktop/dcase2024/checkpoints/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt",
#     model_type="one_peace_retrieval",
#     device=device,
#     dtype="float32"
# )

# # process audio
# captions = "/Volumes/One_Touch/dcase2024/retreival/evaluation_dataset/retrieval_captions.csv"

# df = pd.read_csv(captions)
# text_queries = df['caption'].tolist()

# audio_files = os.listdir(audio_dir)
# audio_list = [os.path.join(audio_dir, x) for x in audio_files if not x.startswith('._')]
# src_audios, audio_padding_masks = model.process_audio(audio_list)

# # Define the text query
# # text_query = ['The person is rummaging through the pans while looking for something.']
# text_query = ['Children and Adults talk with each other positively and laugh.']
# # Process text to obtain text features
# # text_tokens = model.process_text(text_query)
# text_tokens = model.process_text(text_queries)



# if __name__ == '__main__':
#     with torch.no_grad():
#         # Extract audio features
#         text_features = model.extract_text_features(text_tokens)
#         audio_features = model.extract_audio_features(src_audios, audio_padding_masks)
        
#         # Compute similarity scores between text features and each audio
#         similarity_scores = audio_features @ text_features.T
#         similarity_scores = similarity_scores.squeeze()

#         # Retrieve audio files based on highest similarity scores
#         top_audio_indices = torch.topk(similarity_scores, k=10).indices  # Adjust k based on how many top files you want
#         print('top_audio_indices:',top_audio_indices)

#         top_audio_indices = top_audio_indices.cpu().numpy().tolist()  # Convert to a list of indices


#     # Print out the top k relevant audio files
#     for idx in top_audio_indices:
#         print('Relevant audio file:', audio_list[idx])
