import torch
import json
from one_peace.models import from_pretrained
from tqdm import tqdm
import pandas as pd
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'

# Initialize device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = from_pretrained(
    model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt",
    model_type="one_peace_retrieval",
    device=device,
    dtype="float16"
)

# Load captions and prepare audio files
captions_path = "/workspace/jaeyoung/evaluation_dataset/retrieval_captions.csv"
audio_dir = "/workspace/jaeyoung/evaluation_dataset/test"
df = pd.read_csv(captions_path)
text_queries = df['caption'].tolist()
audio_files = os.listdir(audio_dir)
audio_list = [os.path.join(audio_dir, x) for x in audio_files if not x.startswith('._')]

# Prepare results dataframe
results_df = pd.DataFrame(columns=['caption', 'fname_1', 'fname_2', 'fname_3', 'fname_4', 'fname_5', 'fname_6', 'fname_7', 'fname_8', 'fname_9', 'fname_10'])

# Batch processing parameters
audio_batch_size = 10  # Adjust based on your GPU capacity
text_batch_size = 50   # Adjust based on your GPU capacity

if __name__ == '__main__':
    with torch.no_grad():
        # Process audio in batches
        all_audio_features = []
        for i in range(0, len(audio_list), audio_batch_size):
            batch_audio_list = audio_list[i:i + audio_batch_size]
            src_audios, audio_padding_masks = model.process_audio(batch_audio_list)
            batch_audio_features = model.extract_audio_features(src_audios, audio_padding_masks)
            all_audio_features.append(batch_audio_features)
        all_audio_features = torch.cat(all_audio_features, dim=0)

        # Process text in batches
        all_text_features = []
        for i in range(0, len(text_queries), text_batch_size):
            batch_text_queries = text_queries[i:i + text_batch_size]
            text_tokens = model.process_text(batch_text_queries)
            batch_text_features = model.extract_text_features(text_tokens)
            all_text_features.append(batch_text_features)
        all_text_features = torch.cat(all_text_features, dim=0)

        # Compute similarity scores between all text features and all audio features
        similarity_scores = torch.matmul(all_audio_features, all_text_features.T)

        # Retrieve top matching audio files for each text query
        for text_idx, single_text_features in tqdm(enumerate(all_text_features)):
            single_similarity_scores = similarity_scores[:, text_idx]
            top_audio_indices = torch.topk(single_similarity_scores, k=10).indices
            top_audio_indices = top_audio_indices.cpu().numpy().tolist()
            top_files = [audio_list[idx] for idx in top_audio_indices]
            results_df.loc[len(results_df)] = [text_queries[text_idx]] + top_files

        # Save results to CSV
        results_csv_path = '/workspace/jaeyoung/dcase2024_retrieval/submission/onepeace_retrieval_results.csv'
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to {results_csv_path}")
