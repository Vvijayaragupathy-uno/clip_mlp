"""
Prompt processing utilities for MedCLIP
Simplified version for toxoplasmosis classification
"""

import torch
from transformers import AutoTokenizer
from . import constants


def process_class_prompts(cls_prompts, n_prompt=5):
    """
    Process class prompts into tokenized inputs.
    
    Args:
        cls_prompts: Dictionary mapping class names to lists of prompt sentences
                    e.g., {'healthy': ['normal eye', 'no disease'], ...}
        n_prompt: Number of prompts to use per class (default: 5)
    
    Returns:
        Dictionary with tokenized prompt inputs
    """
    tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
    tokenizer.model_max_length = 77
    
    # Flatten all prompts
    all_prompts = []
    prompt_class_map = []  # Track which class each prompt belongs to
    
    for class_idx, (class_name, prompts) in enumerate(cls_prompts.items()):
        # Take up to n_prompt prompts per class
        selected_prompts = prompts[:n_prompt] if len(prompts) >= n_prompt else prompts
        all_prompts.extend(selected_prompts)
        prompt_class_map.extend([class_idx] * len(selected_prompts))
    
    # Tokenize all prompts
    prompt_inputs = tokenizer(
        all_prompts,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': prompt_inputs['input_ids'],
        'attention_mask': prompt_inputs['attention_mask'],
        'class_map': torch.tensor(prompt_class_map)
    }


def process_class_prompts_for_tuning(cls_prompts, n_context=16, class_specific_context=False):
    """
    Process class prompts for prompt tuning.
    
    Args:
        cls_prompts: Dictionary mapping class names to lists of prompt sentences
        n_context: Number of context tokens for prompt tuning
        class_specific_context: Whether to use class-specific context
    
    Returns:
        Dictionary with prompt inputs for tuning
    """
    tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
    tokenizer.model_max_length = 77
    
    # For prompt tuning, we add learnable context tokens
    # This is a simplified version - just tokenize the prompts
    all_prompts = []
    prompt_class_map = []
    
    for class_idx, (class_name, prompts) in enumerate(cls_prompts.items()):
        # Add context placeholder tokens
        context_tokens = " ".join(["[CONTEXT]"] * n_context)
        
        for prompt in prompts:
            # Prepend context tokens to each prompt
            contextualized_prompt = f"{context_tokens} {prompt}"
            all_prompts.append(contextualized_prompt)
            prompt_class_map.append(class_idx)
    
    # Tokenize
    prompt_inputs = tokenizer(
        all_prompts,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': prompt_inputs['input_ids'],
        'attention_mask': prompt_inputs['attention_mask'],
        'class_map': torch.tensor(prompt_class_map),
        'n_context': n_context
    }


def generate_chexpert_class_prompts(n=None):
    """
    Generate class prompts for CheXpert dataset.
    
    Args:
        n: Number of prompts per class (optional)
    
    Returns:
        Dictionary mapping class names to prompt lists
    """
    # Default CheXpert prompts
    chexpert_prompts = {
        'No Finding': [
            'normal chest x-ray',
            'no acute disease',
            'clear lungs',
            'normal heart size',
            'no abnormality detected'
        ],
        'Enlarged Cardiomediastinum': [
            'enlarged cardiomediastinum',
            'widened mediastinum',
            'increased cardiac silhouette'
        ],
        'Cardiomegaly': [
            'enlarged heart',
            'cardiomegaly present',
            'increased heart size'
        ],
        'Lung Opacity': [
            'lung opacity',
            'pulmonary opacity',
            'abnormal lung density'
        ],
        'Lung Lesion': [
            'lung lesion',
            'pulmonary lesion',
            'focal lung abnormality'
        ],
        'Edema': [
            'pulmonary edema',
            'lung edema',
            'fluid in lungs'
        ],
        'Consolidation': [
            'lung consolidation',
            'pulmonary consolidation',
            'airspace consolidation'
        ],
        'Pneumonia': [
            'pneumonia',
            'lung infection',
            'pulmonary infection'
        ],
        'Atelectasis': [
            'atelectasis',
            'lung collapse',
            'partial lung collapse'
        ],
        'Pneumothorax': [
            'pneumothorax',
            'collapsed lung',
            'air in pleural space'
        ],
        'Pleural Effusion': [
            'pleural effusion',
            'fluid around lung',
            'pleural fluid'
        ],
        'Pleural Other': [
            'pleural abnormality',
            'pleural disease',
            'other pleural finding'
        ],
        'Fracture': [
            'fracture',
            'bone fracture',
            'rib fracture'
        ],
        'Support Devices': [
            'support devices present',
            'medical devices visible',
            'tubes and lines present'
        ]
    }
    
    if n is not None:
        # Limit to n prompts per class
        chexpert_prompts = {
            k: v[:n] for k, v in chexpert_prompts.items()
        }
    
    return chexpert_prompts
