from collections import OrderedDict
from transformers import RobertaModel
# from one_peace.model import OnePeaceModel
import torch
from one_peace.models import OnePeaceRetrievalModel
from one_peace.models.one_peace.one_peace_retrieval import OnePeaceRetrievalConfig
import torch.nn as nn
from fairseq.data import Dictionary
import os

# RoBERTa 모델 로드
roberta_model = RobertaModel.from_pretrained("roberta-base")
onepeace_model = torch.load("/workspace/jaeyoung/checkpoints/onepeace_pretrained_chkpoint/one-peace.pt")


# Define mapping function
def map_roberta_to_onepeace(roberta_params):
    mapping = {
        "embeddings.word_embeddings.weight": "encoder_wrapper.text_adapter.embed_tokens.weight",
        "embeddings.position_embeddings.weight": "encoder_wrapper.text_adapter.embed_positions.weight",
        "embeddings.token_type_embeddings.weight": "encoder_wrapper.text_adapter.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight": "encoder_wrapper.text_adapter.LayerNorm.weight",
        "embeddings.LayerNorm.bias": "encoder_wrapper.text_adapter.LayerNorm.bias",
        # Add more mappings as needed for all layers
    }
    
    converted_params = OrderedDict()
    for key, value in mapping.items():
        if key in roberta_params:
            converted_params[value] = roberta_params[key]
    return converted_params

# Ensure dimensions match
def match_dimensions(tensor, target_tensor):
    batch_size, current_feature_size = tensor.shape
    bsz2, target_feature_size = target_tensor.shape
    
    if current_feature_size == target_feature_size:
        return tensor  # If already matched, return as is
    else:
        # Create a Linear layer to transform feature dimension
        linear_layer = nn.Linear(current_feature_size, target_feature_size)
        # Apply the linear transformation
        transformed_tensor = linear_layer(tensor)

        # full_dataset_tensor = torch.randn(50264, current_feature_size)
        # transformed_full_dataset = linear_layer(full_dataset_tensor)
        return transformed_tensor


# Merge parameters
def merge_parameters(onepeace_params, mapped_params, alpha=0.5):
    combined_params = OrderedDict()
    for key in onepeace_params:
        if key in mapped_params:
            # target_shape = onepeace_params[key].shape
            matched_param = match_dimensions(mapped_params[key][:50264], onepeace_params[key])
            combined_params[key] = alpha * matched_param + (1 - alpha) * onepeace_params[key]
        else:
            combined_params[key] = onepeace_params[key]
    return combined_params


if __name__=='__main__':
    roberta_params = roberta_model.state_dict()
    onepeace_params = onepeace_model['model']
    mapped_params = map_roberta_to_onepeace(roberta_params)
    combined_params = merge_parameters(onepeace_params, mapped_params)

    # Verify the combined parameters
    print(combined_params.keys())
    dictionary = Dictionary.load(os.path.join('/workspace/jaeyoung/dcase2024_retrieval/ONE-PEACE/one_peace/utils/BPE', "dict.txt"))

    onepeace_model = OnePeaceRetrievalModel(OnePeaceRetrievalConfig, src_dict=dictionary, head_type='al')

    # Update ONE-PEACE model with combined parameters
    onepeace_model.load_state_dict(combined_params, strict=False)

    # Print success message
    print("ONE-PEACE model's language branch successfully initialized with combined parameters.")


