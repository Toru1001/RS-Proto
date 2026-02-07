from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import json
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from time import time
import psutil
import os

# Model Architecture Classes
class TemporalCNN(nn.Module):
    def __init__(self, input_dim=768, num_filters=128, kernel_sizes=(2,3,4,5,6,7), dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        # Output size is num_filters * num_kernels * 2 (max + mean pooling)
        self.output_size = num_filters * len(kernel_sizes) * 2
        
    def forward(self, x, mask=None):
        x = x.transpose(1, 2)  # (B, H, L)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))  # (B, num_filters, L')
            # Both max and mean pooling
            max_pool = torch.max(c, dim=2)[0]  # (B, num_filters)
            mean_pool = torch.mean(c, dim=2)   # (B, num_filters)
            conv_outs.append(max_pool)
            conv_outs.append(mean_pool)
        out = torch.cat(conv_outs, dim=1)  # (B, num_filters * len(kernel_sizes) * 2)
        out = self.dropout(out)
        return out

class MultiScaleAttentionCNN(nn.Module):
    def __init__(self, hidden_size=768, num_filters=128, kernel_sizes=(2,3,4,5,6,7), dropout=0.3):
        super().__init__()
        # Convolution layers
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, k) for k in kernel_sizes
        ])
        # Attention layers - output 1 value per filter for attention weighting
        self.attn = nn.ModuleList([
            nn.Linear(num_filters, 1) for _ in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_size = num_filters * len(kernel_sizes)
        
    def forward(self, x, mask=None):
        x = x.transpose(1, 2)  # (B, H, L)
        conv_outs = []
        for conv, attn in zip(self.convs, self.attn):
            c = torch.relu(conv(x))  # (B, num_filters, L')
            c_t = c.transpose(1, 2)  # (B, L', num_filters)
            # Apply attention to get weights
            w = attn(c_t)  # (B, L', 1)
            w = torch.softmax(w, dim=1)  # attention weights
            # Weighted sum pooling
            pooled = (c_t * w).sum(dim=1)  # (B, num_filters)
            conv_outs.append(pooled)
        out = torch.cat(conv_outs, dim=1)  # (B, num_filters * len(kernel_sizes))
        out = self.dropout(out)
        return out

class ProjectionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, x):
        return self.layers(x)

class BaseShield(nn.Module):
    """
    Simple base model that concatenates HateBERT and rationale BERT CLS embeddings
    """
    def __init__(self, hatebert_model, additional_model, projection_mlp, hidden_size=768,
                 freeze_additional_model=True):
        super().__init__()
        self.hatebert_model = hatebert_model
        self.additional_model = additional_model
        self.projection_mlp = projection_mlp
        self.hidden_size = hidden_size
        
        if freeze_additional_model:
            for param in self.additional_model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, additional_input_ids, additional_attention_mask,
                return_attentions=False):
        # Main text through HateBERT - get CLS token only
        hatebert_out = self.hatebert_model(input_ids=input_ids, attention_mask=attention_mask,
                                           output_attentions=return_attentions, return_dict=True)
        hatebert_cls = hatebert_out.last_hidden_state[:, 0, :]  # (B, 768)
        
        # Rationale text through frozen BERT - get CLS token only
        with torch.no_grad():
            add_out = self.additional_model(input_ids=additional_input_ids,
                                           attention_mask=additional_attention_mask,
                                           return_dict=True)
            rationale_cls = add_out.last_hidden_state[:, 0, :]  # (B, 768)
        
        # Concatenate CLS embeddings: (B, 1536)
        concat_emb = torch.cat((hatebert_cls, rationale_cls), dim=1)
        
        # Classification
        logits = self.projection_mlp(concat_emb)
        
        # Return dummy rationale_probs and selector_logits for compatibility with app
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        dummy_rationale_probs = torch.zeros(batch_size, seq_len, device=input_ids.device)
        dummy_selector_logits = torch.zeros(batch_size, seq_len, device=input_ids.device)
        
        attns = hatebert_out.attentions if (return_attentions and hasattr(hatebert_out, "attentions")) else None
        return logits, dummy_rationale_probs, dummy_selector_logits, attns


class ConcatModelWithRationale(nn.Module):
    def __init__(self, hatebert_model, additional_model, projection_mlp, hidden_size=768,
                 gumbel_temp=0.5, freeze_additional_model=True, cnn_num_filters=128,
                 cnn_kernel_sizes=(2,3,4), cnn_dropout=0.3):
        super().__init__()
        self.hatebert_model = hatebert_model
        self.additional_model = additional_model
        self.projection_mlp = projection_mlp
        self.gumbel_temp = gumbel_temp
        self.hidden_size = hidden_size
        
        if freeze_additional_model:
            for param in self.additional_model.parameters():
                param.requires_grad = False
        
        self.selector = nn.Linear(hidden_size, 1)
        self.temporal_cnn = TemporalCNN(input_dim=hidden_size, num_filters=cnn_num_filters,
                                        kernel_sizes=cnn_kernel_sizes, dropout=cnn_dropout)
        self.temporal_out_dim = cnn_num_filters * len(cnn_kernel_sizes) * 2
        self.msa_cnn = MultiScaleAttentionCNN(hidden_size=hidden_size, num_filters=cnn_num_filters,
                                              kernel_sizes=cnn_kernel_sizes, dropout=cnn_dropout)
        self.msa_out_dim = self.msa_cnn.output_size
    
    def gumbel_sigmoid_sample(self, logits):
        noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        y = logits + noise
        return torch.sigmoid(y / self.gumbel_temp)
    
    def forward(self, input_ids, attention_mask, additional_input_ids, additional_attention_mask,
                return_attentions=False):
        hatebert_out = self.hatebert_model(input_ids=input_ids, attention_mask=attention_mask,
                                           output_attentions=return_attentions, return_dict=True)
        hatebert_emb = hatebert_out.last_hidden_state
        cls_emb = hatebert_emb[:, 0, :]
        
        with torch.no_grad():
            add_out = self.additional_model(input_ids=additional_input_ids,
                                           attention_mask=additional_attention_mask,
                                           return_dict=True)
            rationale_emb = add_out.last_hidden_state
        
        selector_logits = self.selector(hatebert_emb).squeeze(-1)
        rationale_probs = self.gumbel_sigmoid_sample(selector_logits)
        rationale_probs = rationale_probs * attention_mask.float().to(rationale_probs.device)
        
        masked_hidden = hatebert_emb * rationale_probs.unsqueeze(-1)
        denom = rationale_probs.sum(1).unsqueeze(-1).clamp_min(1e-6)
        pooled_rationale = masked_hidden.sum(1) / denom
        
        temporal_features = self.temporal_cnn(hatebert_emb, attention_mask)
        rationale_features = self.msa_cnn(rationale_emb, additional_attention_mask)
        
        concat_emb = torch.cat((cls_emb, temporal_features, rationale_features, pooled_rationale), dim=1)
        logits = self.projection_mlp(concat_emb)
        
        attns = hatebert_out.attentions if (return_attentions and hasattr(hatebert_out, "attentions")) else None
        return logits, rationale_probs, selector_logits, attns

def load_model_from_hf(model_type="altered"):
    """
    Load model from Hugging Face Hub
    
    Args:
        model_type: Either "altered" or "base" to choose which model to load
    """
    
    repo_id = "seffyehl/BetterShield"
    repo_type = "e5912f6e8c34a10629cfd5a7971ac71ac76d0e9d"
    
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
        revision=repo_type,
        filename=model_filename
    )
    
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        revision=repo_type
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
                                      num_labels=2, dropout=0.0)
        
        model = BaseShield(
            hatebert_model=hatebert_model,
            additional_model=rationale_model,
            projection_mlp=projection_mlp,
            hidden_size=H,
            freeze_additional_model=True
        )
    else:
        # Altered Shield: Complex model with CNN and attention
        cnn_num_filters = model_config.get('cnn_num_filters', 128)
        # Use extended kernel sizes to match saved model
        cnn_kernel_sizes = (2, 3, 4, 5, 6, 7)
        adapter_dim = model_config.get('adapter_dim', 128)
        cnn_dropout = model_config.get('cnn_dropout', 0.3)
        
        # Calculate dimensions
        # TemporalCNN: num_filters * len(kernel_sizes) * 2 (max + mean pooling)
        temporal_out_dim = cnn_num_filters * len(cnn_kernel_sizes) * 2
        # MultiScaleAttentionCNN: num_filters * len(kernel_sizes)
        msa_out_dim = cnn_num_filters * len(cnn_kernel_sizes)
        # Total: CLS (768) + TemporalCNN + MSA + pooled_rationale (768)
        proj_input_dim = H + temporal_out_dim + msa_out_dim + H
        projection_mlp = ProjectionMLP(input_size=proj_input_dim, hidden_size=adapter_dim, 
                                      num_labels=2, dropout=0.0)
        
        model = ConcatModelWithRationale(
            hatebert_model=hatebert_model,
            additional_model=rationale_model,
            projection_mlp=projection_mlp,
            hidden_size=H,
            freeze_additional_model=True,
            cnn_num_filters=cnn_num_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_dropout=cnn_dropout
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create a unified config dict with max_length at top level for compatibility
    unified_config = config.copy()
    if 'max_length' not in unified_config and 'training_config' in config:
        unified_config['max_length'] = training_config.get('max_length', 128)
    
    return model, tokenizer_hatebert, tokenizer_rationale, unified_config, device

def predict_text(text, rationale, model, tokenizer_hatebert, tokenizer_rationale, 
                 device='cpu', max_length=128):
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

def predict_hatespeech_from_file(text_list, rationale_list, true_label, model, tokenizer_hatebert, tokenizer_rationale, config, device):
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
            max_length=config.get('max_length', 128)
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


def predict_hatespeech(text, rationale, model, tokenizer_hatebert, tokenizer_rationale, config, device):
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
        max_length=config.get('max_length', 128)
    )
    
    return result