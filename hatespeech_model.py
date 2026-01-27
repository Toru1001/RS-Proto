from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import json
from transformers import AutoModel, AutoTokenizer

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

class SimpleProjectionMLP(nn.Module):
    """Simpler 2-layer MLP from combined-baseline notebook"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

class SimpleConcatModel(nn.Module):
    """
    Simple concatenation model from combined-baseline notebook.
    Uses dynamic LayerNorm on embeddings before concatenation.
    This matches the original training notebook architecture.
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
        device = input_ids.device
        
        # Forward pass through the HateBERT model
        hatebert_outputs = self.hatebert_model(input_ids=input_ids, attention_mask=attention_mask,
                                               output_attentions=return_attentions, return_dict=True)
        hatebert_embeddings = hatebert_outputs.last_hidden_state[:, 0, :]  # CLS token
        # Dynamic LayerNorm (matching training notebook)
        hatebert_embeddings = torch.nn.LayerNorm(hatebert_embeddings.size()[1:]).to(device)(hatebert_embeddings)
        
        # Forward pass through the Additional Model (frozen)
        with torch.no_grad():
            additional_outputs = self.additional_model(input_ids=additional_input_ids, 
                                                       attention_mask=additional_attention_mask,
                                                       return_dict=True)
            additional_embeddings = additional_outputs.last_hidden_state[:, 0, :]  # CLS token
            # Dynamic LayerNorm (matching training notebook)
            additional_embeddings = torch.nn.LayerNorm(additional_embeddings.size()[1:]).to(device)(additional_embeddings)
        
        # Concatenate the embeddings
        concatenated_embeddings = torch.cat((hatebert_embeddings, additional_embeddings), dim=1)
        
        # Project concatenated embeddings
        logits = self.projection_mlp(concatenated_embeddings)
        
        # Return dummy outputs for compatibility with app
        batch_size, seq_len = input_ids.size()
        dummy_rationale_probs = torch.zeros(batch_size, seq_len, device=device)
        dummy_selector_logits = torch.zeros(batch_size, seq_len, device=device)
        
        attns = hatebert_outputs.attentions if (return_attentions and hasattr(hatebert_outputs, "attentions")) else None
        return logits, dummy_rationale_probs, dummy_selector_logits, attns

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

def load_model_from_hf(model_type="altered", local_checkpoint_path=None):
    """
    Load model from Hugging Face Hub or local checkpoint
    
    Args:
        model_type: "altered", "base", or "simple" to choose which model to load
        local_checkpoint_path: Path to local .pth file (overrides model_type with auto-detection)
    """
    
    repo_id = "seffyehl/BetterShield"
    
    # Handle local checkpoint with auto-detection
    if local_checkpoint_path:
        print(f"Loading local checkpoint: {local_checkpoint_path}")
        checkpoint = torch.load(local_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Auto-detect model type from checkpoint structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Check for model type by looking at keys and shapes
            has_selector = any('selector' in k for k in state_dict.keys())
            has_cnn = any('temporal_cnn' in k or 'msa_cnn' in k for k in state_dict.keys())
            
            # Check projection layer shape to distinguish simple vs base
            proj_key = 'projection_mlp.layers.0.weight'
            if proj_key in state_dict:
                proj_shape = state_dict[proj_key].shape
                # Simple: [512, 1536] (SimpleProjectionMLP)
                # Base: [512, 1536] (ProjectionMLP with same dims, but different class)
                # Altered: [128, 3840] (ProjectionMLP with different dims)
                is_simple_shape = (proj_shape[0] == 512 and proj_shape[1] == 1536)
                is_altered_shape = (proj_shape[0] == 128 and proj_shape[1] == 3840)
            else:
                is_simple_shape = False
                is_altered_shape = False
            
            if has_selector and has_cnn:
                detected_type = "altered"
                print("✓ Detected: Altered Shield (with CNN and selector)")
            elif is_simple_shape and not has_cnn and not has_selector:
                # Could be either simple or base - they have same architecture
                # Use model_type as hint, default to simple if it was requested
                if model_type == "simple":
                    detected_type = "simple"
                    print("✓ Detected: Simple Concat Model (using dynamic LayerNorm)")
                else:
                    detected_type = "base"
                    print("✓ Detected: Base Shield (simple concatenation)")
            else:
                detected_type = "base"
                print("✓ Detected: Base Shield (simple concatenation)")
            
            # Override model_type based on detection
            if model_type != detected_type:
                print(f"⚠️  Overriding model_type '{model_type}' → '{detected_type}' based on checkpoint")
                model_type = detected_type
        
        # Create minimal config for local model
        config = {
            'model_config': {
                'hatebert_model': 'GroNLP/HateBERT',
                'rationale_model': 'bert-base-uncased',
            },
            'training_config': {
                'max_length': 512
            }
        }
        model_path = local_checkpoint_path
        
    else:
        # Download from HuggingFace
        if model_type.lower() == "altered":
            model_filename = "AlteredShield.pth"
            config_filename = "alter_config.json"
        elif model_type.lower() == "base":
            model_filename = "BaseShield.pth"
            config_filename = "base_config.json"
        elif model_type.lower() == "simple":
            # Default to local checkpoint if no path provided
            local_checkpoint_path = ".models/concat_model_reddit_85-15_epochs3_seed42.pth"
            return load_model_from_hf(model_type="simple", local_checkpoint_path=local_checkpoint_path)
        else:
            raise ValueError(f"model_type must be 'altered', 'base', or 'simple', got '{model_type}'")
        
        model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
        config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load checkpoint (weights_only=False needed for older checkpoints)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
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
    
    if model_type.lower() == "simple":
        # Simple Concat Model: From combined-baseline notebook with dynamic LayerNorm
        # Input: 768 (HateBERT CLS) + 768 (BERT CLS) = 1536
        projection_mlp = SimpleProjectionMLP(input_size=1536, output_size=512)
        
        model = SimpleConcatModel(
            hatebert_model=hatebert_model,
            additional_model=rationale_model,
            projection_mlp=projection_mlp,
            hidden_size=H,
            freeze_additional_model=True
        )
    elif model_type.lower() == "base":
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
    
    # Load state dict - handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # HuggingFace models store the state dict directly
        model.load_state_dict(checkpoint)
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