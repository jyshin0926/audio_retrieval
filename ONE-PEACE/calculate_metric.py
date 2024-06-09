import torch
import pandas as pd
import os
from tqdm import tqdm
from one_peace.models import from_pretrained

# Initialize device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = from_pretrained(
    model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt",
    model_type="one_peace_retrieval",
    device=device,
    dtype="float16"
)

# Load captions and prepare audio files
captions_path = "/workspace/jaeyoung/evaluation_dataset/clotho/clotho_captions_evaluation.csv"
audio_dir = "/workspace/jaeyoung/evaluation_dataset/clotho/evaluation"
df = pd.read_csv(captions_path)
text_queries = df['caption_1'].tolist()
audio_files = os.listdir(audio_dir)
audio_list = [os.path.join(audio_dir, x) for x in audio_files if not x.startswith('._')]

# Prepare a list to store metrics for each caption
results = []

if __name__ == '__main__':
    with torch.no_grad():
        # Your existing code for processing audio and text batches

        # Compute similarity scores
        similarity_scores = torch.matmul(all_audio_features, all_text_features.T)

        # Retrieve and calculate metrics
        for text_idx, single_text_features in tqdm(enumerate(all_text_features)):
            true_index = audio_list.index(text_queries[text_idx])  # Assuming the correct match is the corresponding index
            single_similarity_scores = similarity_scores[:, text_idx]
            top_k_indices = torch.topk(single_similarity_scores, k=10).indices

            r_1 = 1 if true_index in top_k_indices[:1] else 0
            r_5 = 1 if true_index in top_k_indices[:5] else 0
            r_10 = 1 if true_index in top_k_indices[:10] else 0

            # Calculate AP for this query
            ranks = torch.where(top_k_indices == true_index)[0]
            ap = 1.0 / (ranks[0].item() + 1) if len(ranks) > 0 else 0

            # Append results
            results.append({
                "caption": text_queries[text_idx],
                "R@1": r_1,
                "R@5": r_5,
                "R@10": r_10,
                "mAP": ap
            })

        # Create DataFrame from results and save to CSV
        results_df = pd.DataFrame(results)
        results_csv_path = '/workspace/jaeyoung/dcase2024_retrieval/submission/onepeace_retrieval_metrics.csv'
        results_df.to_csv(results_csv_path, index=False)
        print(f"Metrics saved to {results_csv_path}")
