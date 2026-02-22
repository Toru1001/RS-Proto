from huggingface_hub import hf_hub_download
import torch
from torch.cuda import device
from torch.nn import functional as F
import torch.nn as nn
import json
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from time import time
import psutil
import os

# Model Architecture Classes
class TemporalCNN(nn.Module):
    def __init__(self, hidden_size=768, num_filters=128, kernel_sizes=(2, 3, 4), dropout=0.1, dilation_base=2):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.dilation_base = dilation_base
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, k, dilation=dilation_base ** i, padding=0)
            for i, k in enumerate(kernel_sizes)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = num_filters * len(kernel_sizes)

    def _causal_padding(self, x, kernel_size, dilation):
        padding = (kernel_size - 1) * dilation
        return F.pad(x, (padding, 0))

    def forward(self, x, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        x = x * mask
        x = x.transpose(1, 2)
        feats = []
        for i, conv in enumerate(self.convs):
            kernel_size = self.kernel_sizes[i]
            dilation = self.dilation_base ** i
            x_padded = self._causal_padding(x, kernel_size, dilation)
            c = F.relu(conv(x_padded))
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)
            feats.append(p)
        out = torch.cat(feats, dim=1)
        return self.dropout(out)

class MultiScaleAttentionCNN(nn.Module):
        def __init__(self, hidden_size=768, num_filters=128, kernel_sizes=(2, 3, 4), dropout=0.3):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Conv1d(hidden_size, num_filters, k) for k in kernel_sizes
            ])
            self.attention_fc = nn.Linear(num_filters, 1)
            self.dropout = nn.Dropout(dropout)
            self.out_dim = num_filters * len(kernel_sizes)
    
        def forward(self, x, mask):
            x = x.transpose(1, 2)
            feats = []
            for conv in self.convs:
                h = F.relu(conv(x))
                h = h.transpose(1, 2)
                attn = self.attention_fc(h).squeeze(-1)
                attn = attn.masked_fill(mask[:, :attn.size(1)] == 0, -1e9)
                alpha = F.softmax(attn, dim=1)
                pooled = torch.sum(h * alpha.unsqueeze(-1), dim=1)
                feats.append(pooled)
            out = torch.cat(feats, dim=1)
            return self.dropout(out)

class ProjectionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, x):
        return self.layers(x)

class GumbelTokenSelector(nn.Module):
        def __init__(self, hidden_size, tau=1.0):
            super().__init__()
            self.tau = tau
            self.proj = nn.Linear(hidden_size * 2, 1)
    
        def forward(self, token_embeddings, cls_embedding, training=True):
            B, L, H = token_embeddings.size()
            cls_exp = cls_embedding.unsqueeze(1).expand(-1, L, -1)
            x = torch.cat([token_embeddings, cls_exp], dim=-1)
            logits = self.proj(x).squeeze(-1)
    
            if training:
                probs = F.gumbel_softmax(
                    torch.stack([logits, torch.zeros_like(logits)], dim=-1),
                    tau=self.tau,
                    hard=False
                )[..., 0]
            else:
                probs = torch.sigmoid(logits)
            return probs, logits

class BaseShield(nn.Module):
    """
    Simple base model that concatenates HateBERT and rationale BERT CLS embeddings
    """
    def __init__(self, hatebert_model, additional_model, projection_mlp, device='cpu',
                 freeze_additional_model=True):
        super().__init__()
        self.hatebert_model = hatebert_model
        self.additional_model = additional_model
        self.projection_mlp = projection_mlp
        self.device = device
        
        if freeze_additional_model:
            for param in self.additional_model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, additional_input_ids, additional_attention_mask):
        hatebert_outputs = self.hatebert_model(input_ids=input_ids, attention_mask=attention_mask)
        hatebert_embeddings = hatebert_outputs.last_hidden_state[:, 0, :]
        hatebert_embeddings = torch.nn.LayerNorm(hatebert_embeddings.size()[1:]).to(self.device)(hatebert_embeddings.to(self.device)).to(self.device)

        additional_outputs = self.additional_model(input_ids=additional_input_ids, attention_mask=additional_attention_mask)
        additional_embeddings = additional_outputs.last_hidden_state[:, 0, :]
        additional_embeddings = torch.nn.LayerNorm(additional_embeddings.size()[1:]).to(self.device)(additional_embeddings.to(self.device)).to(self.device)

        concatenated_embeddings = torch.cat((hatebert_embeddings, additional_embeddings), dim=1).to(self.device)
        projected_embeddings = self.projection_mlp(concatenated_embeddings).to(self.device)

        # Return 4 values to match ConcatModel interface (rationale_probs, selector_logits, attentions are None)
        return projected_embeddings 

class ConcatModel(nn.Module):
    def __init__(self, hatebert_model, additional_model, temporal_cnn, msa_cnn, selector, projection_mlp, freeze_additional_model=True, freeze_hatebert=True):
        super().__init__()
        self.hatebert_model = hatebert_model
        self.additional_model = additional_model
        self.temporal_cnn = temporal_cnn
        self.msa_cnn = msa_cnn
        self.selector = selector
        self.projection_mlp = projection_mlp

        if freeze_additional_model:
            for p in self.additional_model.parameters():
                p.requires_grad = False
        if freeze_hatebert:
            for p in self.hatebert_model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, additional_input_ids, additional_attention_mask):
        hate_outputs = self.hatebert_model(input_ids=input_ids, attention_mask=attention_mask)
        seq_emb = hate_outputs.last_hidden_state
        cls_emb = seq_emb[:, 0, :]
        
        token_probs, token_logits = self.selector(seq_emb, cls_emb, self.training)
        temporal_feat = self.temporal_cnn(seq_emb, attention_mask)
        
        weights = token_probs.unsqueeze(-1)
        H_r = (seq_emb * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
        
        with torch.no_grad():
            add_outputs = self.additional_model(input_ids=additional_input_ids, attention_mask=additional_attention_mask)
            add_seq = add_outputs.last_hidden_state
        
        msa_feat = self.msa_cnn(add_seq, additional_attention_mask)
        concat = torch.cat([cls_emb, temporal_feat, msa_feat, H_r], dim=1)
        logits = self.projection_mlp(concat)
        return logits, token_probs, token_logits, hate_outputs.attentions if hasattr(hate_outputs, "attentions") else None

def load_model_from_hf(model_type="altered"):
    """
    Load model from Hugging Face Hub
    
    Args:
        model_type: Either "altered" or "base" to choose which model to load
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    repo_id = "seffyehl/BetterShield"
    # repo_type = "e5912f6e8c34a10629cfd5a7971ac71ac76d0e9d"
    
    # Choose model and config files based on model_type
    if model_type.lower() == "altered":
        model_filename = "AlteredShield.pth"
        config_filename = "alter_config.json"
    elif model_type.lower() == "base":
        model_filename = "BaseShield.pth"
        config_filename = "base_config.json"
    else:
        raise ValueError(f"model_type must be 'altered' or 'base', got '{model_type}'")
    
    # Download files
    model_path = hf_hub_download(
        repo_id=repo_id,
        # revision=repo_type,
        filename=model_filename
    )
    
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        # revision=repo_type
    )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle nested config structure (base model uses model_config, altered uses flat structure)
    if 'model_config' in config:
        model_config = config['model_config']
        training_config = config.get('training_config', {})
    else:
        model_config = config
        training_config = config
    
    # Initialize base models
    hatebert_model = AutoModel.from_pretrained(model_config['hatebert_model'])
    rationale_model = AutoModel.from_pretrained(model_config['rationale_model'])
    
    tokenizer_hatebert = AutoTokenizer.from_pretrained(model_config['hatebert_model'])
    tokenizer_rationale = AutoTokenizer.from_pretrained(model_config['rationale_model'])
    
    # Rebuild architecture based on model type
    H = hatebert_model.config.hidden_size
    max_length = training_config.get('max_length', 128)
    
    if model_type.lower() == "base":
        # Base Shield: Simple concatenation model
        # Input: 768 (HateBERT CLS) + 768 (Rationale BERT CLS) = 1536
        proj_input_dim = H * 2  # 1536
        # The saved model uses 512, not what's in projection_config
        adapter_dim = 512  # hardcoded to match saved weights
        projection_mlp = ProjectionMLP(input_size=proj_input_dim, hidden_size=adapter_dim, 
                                      num_labels=2)
        
        model = BaseShield(
            hatebert_model=hatebert_model,
            additional_model=rationale_model,
            projection_mlp=projection_mlp,
            freeze_additional_model=True, 
            device=device
        ).to(device)
    else:
        temporal_cnn = TemporalCNN(hidden_size=768, num_filters=128, kernel_sizes=(2, 3, 4)).to(device)
        msa_cnn = MultiScaleAttentionCNN(hidden_size=768, num_filters=128, kernel_sizes=(2, 3, 4)).to(device)
        selector = GumbelTokenSelector(hidden_size=768, tau=1.0).to(device)
        projection_mlp = ProjectionMLP(input_size=temporal_cnn.out_dim + msa_cnn.out_dim + 768 * 2, hidden_size=512, num_labels=2).to(device)
        model = ConcatModel(
            hatebert_model=hatebert_model, 
            additional_model=rationale_model, 
            temporal_cnn=temporal_cnn, 
            msa_cnn=msa_cnn, 
            selector=selector,
            projection_mlp=projection_mlp, 
            freeze_additional_model=True, 
            freeze_hatebert=True).to(device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Dataset: {checkpoint.get('dataset', 'unknown')}, Seed: {checkpoint.get('seed', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)
    
    # Create a unified config dict with max_length at top level for compatibility
    unified_config = config.copy()
    if 'max_length' not in unified_config and 'training_config' in config:
        unified_config['max_length'] = training_config.get('max_length', 128)
    
    return model, tokenizer_hatebert, tokenizer_rationale, unified_config, device

def predict_text(text, rationale, model, tokenizer_hatebert, tokenizer_rationale, 
                 device='cpu', max_length=128, model_type="altered"):
    """
    Predict hate speech for a given text and rationale
    
    Args:
        text: Input text to classify
        rationale: Rationale/explanation text
        model: Loaded model
        tokenizer_hatebert: HateBERT tokenizer
        tokenizer_rationale: Rationale model tokenizer
        device: 'cpu' or 'cuda'
        max_length: Maximum sequence length
        model_type: Either "altered" or "base" to determine how to process inputs
    
    Returns:
        prediction: 0 or 1
        probability: Confidence score
        rationale_scores: Token-level rationale scores
    """
    model.eval()
    
    # Tokenize inputs
    inputs_main = tokenizer_hatebert(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    inputs_rationale = tokenizer_rationale(
        rationale if rationale else text,  # Use text if no rationale provided
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = inputs_main['input_ids'].to(device)
    attention_mask = inputs_main['attention_mask'].to(device)
    add_input_ids = inputs_rationale['input_ids'].to(device)
    add_attention_mask = inputs_rationale['attention_mask'].to(device)
    
    # Inference
    if model_type.lower() == "base":
        with torch.no_grad():
            logits = model(
                input_ids, 
                attention_mask, 
                add_input_ids, 
                add_attention_mask
            )
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            prediction = logits.argmax(dim=1).item()
            confidence = probs[0, prediction].item()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy(),
            'rationale_scores': None,  # Base model does not produce token-level rationale scores
            'tokens': tokenizer_hatebert.convert_ids_to_tokens(input_ids[0])
        }
    
    with torch.no_grad():
        logits, rationale_probs, selector_logits, _ = model(
            input_ids, 
            attention_mask, 
            add_input_ids, 
            add_attention_mask
        )
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': probs[0].cpu().numpy(),
        'rationale_scores': rationale_probs[0].cpu().numpy(),
        'tokens': tokenizer_hatebert.convert_ids_to_tokens(input_ids[0])
    }

def predict_hatespeech_from_file(text_list, rationale_list, true_label, model, tokenizer_hatebert, tokenizer_rationale, config, device, model_type="altered"):
    """
    Predict hate speech for text read from a file
    
    Args:
        text_list: List of input texts to classify
        rationale_list: List of rationale/explanation texts
        true_label: True label for evaluation
        model: Loaded model
        tokenizer_hatebert: HateBERT tokenizer
        tokenizer_rationale: Rationale tokenizer
        config: Model configuration
        device: Device to run on
    Returns:
        f1_score: F1 score for the predictions
        accuracy: Accuracy for the predictions
        precision: Precision for the predictions
        recall: Recall for the predictions
        confusion_matrix: Confusion matrix as a 2D list
        cpu_usage: CPU usage during prediction
        memory_usage: Memory usage during prediction
        runtime: Total runtime for predictions
    """
    predictions = []
    cpu_percent_list = []
    memory_percent_list = []

    process = psutil.Process(os.getpid())
    start_time = time()
    for idx, (text, rationale) in enumerate(zip(text_list, rationale_list)):
        result = predict_text(
            text=text,
            rationale=rationale,
            model=model,
            tokenizer_hatebert=tokenizer_hatebert,
            tokenizer_rationale=tokenizer_rationale,
            device=device,
            max_length=config.get('max_length', 128),
            model_type=model_type
        )
        predictions.append(result['prediction'])
        # Log resource usage every 10th sample and at end to reduce overhead
        if idx % 10 == 0 or idx == len(text_list) - 1:
            cpu_percent_list.append(process.cpu_percent())
            memory_percent_list.append(process.memory_info().rss / 1024 / 1024)

    end_time = time()
    runtime = end_time - start_time
    # Calculate metrics
    f1 = f1_score(true_label, predictions, zero_division=0)
    accuracy = accuracy_score(true_label, predictions)
    precision = precision_score(true_label, predictions, zero_division=0)
    recall = recall_score(true_label, predictions, zero_division=0)
    cm = confusion_matrix(true_label, predictions).tolist()
    
    avg_cpu = sum(cpu_percent_list) / len(cpu_percent_list) if cpu_percent_list else 0
    avg_memory = sum(memory_percent_list) / len(memory_percent_list) if memory_percent_list else 0  
    peak_memory = max(memory_percent_list) if memory_percent_list else 0
    peak_cpu = max(cpu_percent_list) if cpu_percent_list else 0

    return {
        'f1_score': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'cpu_usage': avg_cpu,
        'memory_usage': avg_memory,
        'peak_cpu_usage': peak_cpu,
        'peak_memory_usage': peak_memory,
        'runtime': runtime
    }


def predict_hatespeech(text, rationale, model, tokenizer_hatebert, tokenizer_rationale, config, device, model_type="altered"):
    """
    Predict hate speech for given text
    
    Args:
        text: Input text to classify
        rationale: Optional rationale text
        model: Loaded model
        tokenizer_hatebert: HateBERT tokenizer
        tokenizer_rationale: Rationale tokenizer
        config: Model configuration
        device: Device to run on
    
    Returns:
        Dictionary with prediction results
    """
    # Get prediction
    result = predict_text(
        text=text,
        rationale=rationale,
        model=model,
        tokenizer_hatebert=tokenizer_hatebert,
        tokenizer_rationale=tokenizer_rationale,
        device=device,
        max_length=config.get('max_length', 128),
        model_type=model_type
    )
    
    return result

def predict_hatespeech_from_file_mock():
    """
    Mock function for predict_hatespeech_from_file that returns hardcoded data for testing
    
    Args:
        text_list: List of input texts to classify (not used in mock)
        rationale_list: List of rationale/explanation texts (not used in mock)
        true_label: True label for evaluation (not used in mock)
        model: Loaded model (not used in mock)
        tokenizer_hatebert: HateBERT tokenizer (not used in mock)
        tokenizer_rationale: Rationale tokenizer (not used in mock)
        config: Model configuration (not used in mock)
        device: Device to run on (not used in mock)
    Returns:
        Dictionary with hardcoded metrics for testing
    """
    # Hardcoded predictions matching the number of samples
    predictions = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    true_labels = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    
    # Hardcoded resource usage metrics
    cpu_percent_list = [25.3, 28.1, 26.5, 27.2, 26.8, 27.9, 25.5, 28.3, 26.2, 27.1]
    memory_percent_list = [145.3, 152.1, 148.5, 151.2, 149.8, 153.2, 146.5, 154.3, 150.2, 152.1]
    
    f1 = f1_score(true_labels, predictions, zero_division=0)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions).tolist()
    
    avg_cpu = sum(cpu_percent_list) / len(cpu_percent_list) if cpu_percent_list else 0
    avg_memory = sum(memory_percent_list) / len(memory_percent_list) if memory_percent_list else 0
    peak_memory = max(memory_percent_list) if memory_percent_list else 0
    peak_cpu = max(cpu_percent_list) if cpu_percent_list else 0
    
    # Hardcoded runtime
    runtime = 12.543
    
    return {
        'f1_score': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'cpu_usage': avg_cpu,
        'memory_usage': avg_memory,
        'peak_cpu_usage': peak_cpu,
        'peak_memory_usage': peak_memory,
        'runtime': runtime,
        'predictions': predictions  # Added for visibility
    }

def predict_text_mock(text, max_length=128):
    import numpy as np

    # Simple whitespace tokenization for mock output
    raw_tokens = (text or "").split()
    mock_tokens = raw_tokens[:max_length]

    # Build a simple attention mask (1 for tokens)
    attention_mask = [1] * len(mock_tokens)

    # Generate random rationale scores matching token count
    mock_rationale_scores = np.random.rand(len(mock_tokens)).astype(np.float32)
    
    # Randomized probabilities [class_0, class_1]
    # Class 0 = not hate speech, Class 1 = hate speech
    mock_probabilities = np.random.rand(2).astype(np.float32)
    mock_probabilities = mock_probabilities / mock_probabilities.sum()
    
    # Prediction (argmax of probabilities)
    mock_prediction = int(np.argmax(mock_probabilities))  # Class 1: hate speech
    
    # Confidence score
    mock_confidence = float(np.max(mock_probabilities))
    
    return {
        'prediction': mock_prediction,
        'confidence': mock_confidence,
        'probabilities': mock_probabilities,
        'rationale_scores': mock_rationale_scores,
        'tokens': mock_tokens,
        'attention_mask': attention_mask
    }