from collections import OrderedDict
from transformers import RobertaModel
# from one_peace.model import OnePeaceModel
import torch
from one_peace.models import OnePeaceRetrievalModel, OnePeaceBaseModel, OnePeacePretrainModel
from one_peace.models.one_peace.one_peace_retrieval import OnePeaceRetrievalConfig
from one_peace.models.one_peace.one_peace_pretrain import OnePeacePretrainConfig
from one_peace.data.pretrain_data.audio_text_pretrain_dataset import AudioTextPretrainDataset

import torch.nn as nn
from fairseq.data import Dictionary
import os

# RoBERTa 모델 로드
roberta_model = RobertaModel.from_pretrained("roberta-base")
onepeace_model = torch.load("/workspace/jaeyoung/checkpoints/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt")


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

import torch

# Function to clone the initial state of model parameters
def clone_model_state(model):
    return {name: param.clone() for name, param in model.state_dict().items()}


# Function to compare and print changes in the model parameters
def print_updated_params(model, initial_state):
    for name, param in model.state_dict().items():
        initial_param = initial_state[name]
        # Using 'torch.equal' to check if two tensors are the same
        if not torch.equal(param, initial_param):
            print(f"Parameter updated: {name}")




if __name__=='__main__':
    roberta_params = roberta_model.state_dict()
    onepeace_params = onepeace_model['model']
    onepeace_config = onepeace_model['cfg']  # Safely load config if it exists

    mapped_params = map_roberta_to_onepeace(roberta_params)
    combined_params = merge_parameters(onepeace_params, mapped_params)

    # Verify the combined parameters
    print(combined_params.keys())
    dictionary = Dictionary.load(os.path.join('/workspace/jaeyoung/dcase2024_retrieval/ONE-PEACE/one_peace/utils/BPE', "dict.txt"))

    # onepeace_model = OnePeaceRetrievalModel(OnePeaceRetrievalConfig, src_dict=dictionary, head_type='al')

    # # Update ONE-PEACE model with combined parameters
    # onepeace_model.load_state_dict(combined_params, strict=False)
    model_path = '/workspace/jaeyoung/checkpoints/onepeace_pretrained_chkpoint/retrieval_onepeace_roberta_ensemble.pt'
    # torch.save({'model':onepeace_model.state_dict(), 'cfg':onepeace_config}, model_path)
    torch.save({'model':combined_params, 'cfg':onepeace_config}, model_path)

    # Print success message
    print("ONE-PEACE model's language branch successfully initialized with combined parameters.")



    # # Assuming `onepeace_model` is your initial model
    # initial_state = clone_model_state(onepeace_model)

    # # Update the model's parameters (assuming this is done somewhere in your code)
    # onepeace_model.load_state_dict(combined_params, strict=False)

    # # Call the function to print updated parameters
    # print_updated_params(onepeace_model, initial_state)

