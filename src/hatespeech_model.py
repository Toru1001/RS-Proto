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
    """
    Temporal CNN applied across the sequence (time) dimension.
    Input: sequence_embeddings (B, L, H), attention_mask (B, L)
    Output: pooled vector (B, output_dim) where output_dim = num_filters * len(kernel_sizes) * 2
            (we concatenate max-pooled and mean-pooled features for each kernel size)
    """
    def __init__(self, input_dim=768, num_filters=256, kernel_sizes=(2, 3, 4), dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        # Convs expect (B, C_in, L) where C_in = input_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_embeddings, attention_mask=None):
        """
        sequence_embeddings: (B, L, H)
        attention_mask: (B, L) with 1 for valid tokens, 0 for padding
        returns: (B, num_filters * len(kernel_sizes) * 2)  # max + mean pooled per conv
        """
        # transpose to (B, H, L)
        x = sequence_embeddings.transpose(1, 2).contiguous()  # (B, H, L)

        pooled_outputs = []
        for conv in self.convs:
            conv_out = conv(x)                # (B, num_filters, L_out)
            conv_out = F.relu(conv_out)
            L_out = conv_out.size(2)

            if attention_mask is not None:
                # resize mask to match L_out
                mask = attention_mask.float()
                if mask.size(1) != L_out:
                    mask = F.interpolate(mask.unsqueeze(1), size=L_out, mode='nearest').squeeze(1)
                mask = mask.unsqueeze(1).to(conv_out.device)  # (B,1,L_out)

                # max pool with masking
                neg_inf = torch.finfo(conv_out.dtype).min / 2
                max_masked = torch.where(mask.bool(), conv_out, neg_inf * torch.ones_like(conv_out))
                max_pooled = torch.max(max_masked, dim=2)[0]  # (B, num_filters)

                # mean pool with masking
                sum_masked = (conv_out * mask).sum(dim=2)    # (B, num_filters)
                denom = mask.sum(dim=2).clamp_min(1e-6)     # (B,1)
                mean_pooled = sum_masked / denom            # (B, num_filters)
            else:
                max_pooled = torch.max(conv_out, dim=2)[0]
                mean_pooled = conv_out.mean(dim=2)

            pooled_outputs.append(max_pooled)
            pooled_outputs.append(mean_pooled)

        out = torch.cat(pooled_outputs, dim=1)  # (B, num_filters * len(kernel_sizes) * 2)
        out = self.dropout(out)
        return out


class MultiScaleAttentionCNN(nn.Module):
    def __init__(self, hidden_size=768, num_filters=64, kernel_sizes=(2, 3, 4, 5, 6, 7), dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList()
        self.pads = nn.ModuleList()

        for k in self.kernel_sizes:
            pad_left = (k - 1) // 2
            pad_right = k - 1 - pad_left
            self.pads.append(nn.ConstantPad1d((pad_left, pad_right), 0.0))
            self.convs.append(nn.Conv1d(hidden_size, num_filters, kernel_size=k, padding=0))

        self.attn = nn.ModuleList([nn.Linear(num_filters, 1) for _ in self.kernel_sizes])
        self.output_size = num_filters * len(self.kernel_sizes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, mask):
        """
        hidden_states: (B, L, H)
        mask: (B, L)
        """
        x = hidden_states.transpose(1, 2)  # (B, H, L)
        attn_mask = mask.unsqueeze(1).float()

        conv_outs = []

        for pad, conv, att in zip(self.pads, self.convs, self.attn):
            padded = pad(x)      # (B, H, L)
            c = conv(padded)     # (B, F, L)
            c = F.relu(c)
            c = c * attn_mask

            c_t = c.transpose(1, 2)    # (B, L, F)
            w = att(c_t)               # (B, L, 1)
            w = w.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            w = F.softmax(w, dim=1)

            pooled = (c_t * w).sum(dim=1)   # (B, F)
            conv_outs.append(pooled)

        out = torch.cat(conv_outs, dim=1)   # (B, F * K)
        return self.dropout(out)


class ProjectionMLP(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_labels=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, x):
        return self.layers(x)


class ConcatModelWithRationale(nn.Module):
    def __init__(self,
                 hatebert_model,
                 additional_model,
                 projection_mlp,
                 hidden_size=768,
                 gumbel_temp=0.5,
                 freeze_additional_model=True,
                 cnn_num_filters=64,
                 cnn_kernel_sizes=(2, 3, 4, 5, 6, 7),
                 cnn_dropout=0.0):
        super().__init__()
        self.hatebert_model = hatebert_model
        self.additional_model = additional_model
        self.projection_mlp = projection_mlp
        self.gumbel_temp = gumbel_temp
        self.hidden_size = hidden_size

        if freeze_additional_model:
            for param in self.additional_model.parameters():
                param.requires_grad = False

        # selector head (per-token logits)
        self.selector = nn.Linear(hidden_size, 1)

        # Temporal CNN over HateBERT embeddings (main text)
        self.temporal_cnn = TemporalCNN(input_dim=hidden_size,
                                        num_filters=cnn_num_filters,
                                        kernel_sizes=cnn_kernel_sizes,
                                        dropout=cnn_dropout)
        self.temporal_out_dim = cnn_num_filters * len(cnn_kernel_sizes) * 2

        # MultiScaleAttentionCNN over rationale embeddings (frozen BERT)
        self.msa_cnn = MultiScaleAttentionCNN(hidden_size=hidden_size,
                                              num_filters=cnn_num_filters,
                                              kernel_sizes=cnn_kernel_sizes,
                                              dropout=cnn_dropout)
        self.msa_out_dim = self.msa_cnn.output_size

    def gumbel_sigmoid_sample(self, logits):
        noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        y = logits + noise
        return torch.sigmoid(y / self.gumbel_temp)

    def forward(self, input_ids, attention_mask, additional_input_ids, additional_attention_mask, return_attentions=False):
        # Main text through HateBERT
        hatebert_out = self.hatebert_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           output_attentions=return_attentions,
                                           return_dict=True)
        hatebert_emb = hatebert_out.last_hidden_state   # (B, L, H)
        cls_emb = hatebert_emb[:, 0, :]                 # (B, H)

        # Rationale text through frozen BERT
        with torch.no_grad():
            add_out = self.additional_model(input_ids=additional_input_ids,
                                            attention_mask=additional_attention_mask,
                                            return_dict=True)
            rationale_emb = add_out.last_hidden_state   # (B, L, H)

        # selector logits & Gumbel-Sigmoid sampling on HateBERT
        selector_logits = self.selector(hatebert_emb).squeeze(-1)  # (B, L)
        rationale_probs = self.gumbel_sigmoid_sample(selector_logits)  # (B, L)
        rationale_probs = rationale_probs * attention_mask.float().to(rationale_probs.device)

        # pooled rationale summary
        masked_hidden = hatebert_emb * rationale_probs.unsqueeze(-1)
        denom = rationale_probs.sum(1).unsqueeze(-1).clamp_min(1e-6)
        pooled_rationale = masked_hidden.sum(1) / denom  # (B, H)

        # CNN branches
        temporal_features = self.temporal_cnn(hatebert_emb, attention_mask)           # (B, temporal_out_dim)
        rationale_features = self.msa_cnn(rationale_emb, additional_attention_mask)   # (B, msa_out_dim)

        # concat CLS + CNN features + pooled rationale
        concat_emb = torch.cat((cls_emb, temporal_features, rationale_features, pooled_rationale), dim=1)

        logits = self.projection_mlp(concat_emb)

        attns = hatebert_out.attentions if (return_attentions and hasattr(hatebert_out, "attentions")) else None
        return logits, rationale_probs, selector_logits, attns


class BaseShield(nn.Module):
    """
    Simple base model that concatenates HateBERT and rationale BERT CLS embeddings
    and projects to label logits via a small MLP.
    """
    def __init__(self, hatebert_model, additional_model, projection_mlp, device='cpu', freeze_additional_model=True):
        super().__init__()
        self.hatebert_model = hatebert_model
        self.additional_model = additional_model
        self.projection_mlp = projection_mlp
        self.device = device

        if freeze_additional_model:
            for param in self.additional_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, additional_input_ids, additional_attention_mask):
        hatebert_outputs = self.hatebert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hatebert_embeddings = hatebert_outputs.last_hidden_state[:, 0, :]

        additional_outputs = self.additional_model(input_ids=additional_input_ids, attention_mask=additional_attention_mask, return_dict=True)
        additional_embeddings = additional_outputs.last_hidden_state[:, 0, :]

        concatenated_embeddings = torch.cat((hatebert_embeddings, additional_embeddings), dim=1)
        logits = self.projection_mlp(concatenated_embeddings)
        return logits

def load_model_from_hf(model_type="altered"):
    """
    Load model from Hugging Face Hub
    
    Args:
        model_type: Either "altered" or "base" to choose which model to load
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    repo_id = "seffyehl/BetterShield"
    
    # Choose model and config files based on model_type
    if model_type.lower() == "altered":
        model_filename = "AlteredShield.pth"
        config_filename = "alter_config.json"
    elif model_type.lower() == "base":
        model_filename = "BaselineShield.pth"
        config_filename = "base_config.json"
    else:
        raise ValueError(f"model_type must be 'altered' or 'base', got '{model_type}'")
    
    # Download files
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename
    )
    
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
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
    
    # Rebuild architecture based on model type using training_config values when available
    H = hatebert_model.config.hidden_size
    max_length = training_config.get('max_length', 128)
    
    if model_type.lower() == "base":
        proj_input_dim = H * 2
        projection_mlp = ProjectionMLP(input_size=proj_input_dim, hidden_size=adapter_dim, num_labels=num_labels)
        model = BaseShield(
            hatebert_model=hatebert_model,
            additional_model=rationale_model,
            projection_mlp=projection_mlp,
            freeze_additional_model=freeze_rationale,
            device=device
        ).to(device)
    else:
        # For altered model, let ConcatModelWithRationale initialize its own CNN modules
        # The CNN modules are created inside __init__, so we just need to create the model
        # and then load the state dict
        
        # First, create a dummy projection_mlp - we'll replace it after calculating dims
        # Actually, we need to calculate dims first to create the correct projection_mlp
        
        # Calculate dimensions based on inferred parameters
        temporal_out_dim = cnn_num_filters * len(cnn_kernel_sizes) * 2
        msa_out_dim = cnn_num_filters * len(cnn_kernel_sizes)
        proj_input_dim = H + temporal_out_dim + msa_out_dim + H
        
        projection_mlp = ProjectionMLP(input_size=proj_input_dim, hidden_size=adapter_dim, num_labels=num_labels)
        
        model = ConcatModelWithRationale(
            hatebert_model=hatebert_model,
            additional_model=rationale_model,
            projection_mlp=projection_mlp,
            hidden_size=H,
            freeze_additional_model=freeze_rationale,
            cnn_num_filters=cnn_num_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_dropout=cnn_dropout
        ).to(device)
    
    # Load state dict with strict checking and error reporting
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict_to_load = checkpoint['model_state_dict']
    else:
        state_dict_to_load = checkpoint
    
    # Check for missing and unexpected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict_to_load.keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys:
        print(f"WARNING: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"WARNING: Unexpected keys in checkpoint: {unexpected_keys}")
    
    # Load with strict=False to handle any minor mismatches, but log warnings
    incompatible_keys = model.load_state_dict(state_dict_to_load, strict=True)
    
    if incompatible_keys.missing_keys:
        print(f"Missing keys after load: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        print(f"Unexpected keys after load: {incompatible_keys.unexpected_keys}")
    
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Dataset: {checkpoint.get('dataset', 'unknown')}, Seed: {checkpoint.get('seed', 'unknown')}")
    
    # CRITICAL: Set to eval mode and ensure no gradient computation
    model.eval()
    
    # Disable dropout explicitly by setting training mode to False for all modules
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.p = 0  # Set dropout probability to 0
    
    model = model.to(device)
    
    # Verify model is in eval mode
    print(f"Model training mode: {model.training}")
    print(f"Dropout layers found: {sum(1 for _ in model.modules() if isinstance(_, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)))}")
    
    # Create a unified config dict with max_length at top level for compatibility
    unified_config = config.copy()
    if 'max_length' not in unified_config and 'training_config' in config:
        unified_config['max_length'] = training_config.get('max_length', 128)
    
    return model, tokenizer_hatebert, tokenizer_rationale, unified_config, device


def combined_loss(logits, labels, rationale_probs, selector_logits, rationale_mask=None, attns=None, attn_weight=0.0, rationale_weight=1.0):
    cls_loss = F.cross_entropy(logits, labels)

    # supervise selector logits with BCE-with-logits against rationale mask (if available)
    if rationale_mask is not None:
        selector_loss = F.binary_cross_entropy_with_logits(selector_logits, rationale_mask.to(selector_logits.device))
    else:
        selector_loss = torch.tensor(0.0, device=cls_loss.device)

    # optional attention alignment loss (disabled by default)
    attn_loss = torch.tensor(0.0, device=cls_loss.device)
    if attns is not None and attn_weight > 0.0:
        try:
            last_attn = attns[-1]  # (B, H, L, L)
            attn_mass = last_attn.mean(1).mean(1)  # (B, L)
            attn_loss = F.mse_loss(attn_mass, rationale_mask.to(attn_mass.device))
        except Exception:
            attn_loss = torch.tensor(0.0, device=cls_loss.device)

    total_loss = cls_loss + rationale_weight * selector_loss + attn_weight * attn_loss
    return total_loss, cls_loss.item(), selector_loss.item(), attn_loss.item()


def predict_text(text, rationale, model, tokenizer_hatebert, tokenizer_rationale, 
                 device='cpu', max_length=128, model_type="altered"):
    # Ensure model is in eval mode (defensive programming)
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
    
    # Inference with no gradient computation
    with torch.no_grad():
        if model_type.lower() == "base":
            logits = model(
                input_ids, 
                attention_mask, 
                add_input_ids, 
                add_attention_mask
            )
        else:
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
        'tokens': tokenizer_hatebert.convert_ids_to_tokens(input_ids[0])
    }
    
    if model_type.lower() != "base":
        result['rationale_scores'] = rationale_probs[0].cpu().numpy() if 'rationale_probs' in locals() else None
    else:
        result['rationale_scores'] = None
    
    return result


def predict_hatespeech_from_file(
    text_list,
    rationale_list,
    true_label,
    model,
    tokenizer_hatebert,
    tokenizer_rationale,
    config,
    device,
    model_type="altered"
):

    print(f"\nStarting inference for model: {type(model).__name__}")

    predictions = []
    all_probs = []
    cpu_percent_list = []
    memory_percent_list = []

    process = psutil.Process(os.getpid())

    # 🔥 GPU synchronization BEFORE timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 🔥 Optional warmup (prevents first-batch timing bias)
    with torch.no_grad():
        _ = predict_text(
            text=text_list[0],
            rationale=rationale_list[0],
            model=model,
            tokenizer_hatebert=tokenizer_hatebert,
            tokenizer_rationale=tokenizer_rationale,
            device=device,
            max_length=config.get('max_length', 128),
            model_type=model_type
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ⏱ Start timer AFTER warmup
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
        all_probs.append(result['probabilities'])

        # Reduce monitoring overhead
        if idx % 10 == 0 or idx == len(text_list) - 1:
            cpu_percent_list.append(process.cpu_percent())
            memory_percent_list.append(process.memory_info().rss / 1024 / 1024)

    # 🔥 GPU synchronization BEFORE stopping timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time()
    runtime = end_time - start_time

    print(f"Inference completed for {type(model).__name__}")
    print(f"Total runtime: {runtime:.4f} seconds")

    # ---------------- Metrics ----------------
    all_probs = np.array(all_probs)

    print(f"Probability Mean: {all_probs.mean(axis=0)}")
    print(f"Probability Std: {all_probs.std(axis=0)}")
    print(f"Prediction distribution: {np.bincount(predictions, minlength=2)}")

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
        'model_name': type(model).__name__,   # 👈 makes logs clearer
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
        'all_probabilities': all_probs.tolist()
    }


def predict_hatespeech(text, rationale, model, tokenizer_hatebert, tokenizer_rationale, config, device, model_type="altered"):
    """
    Predict hate speech for given text
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
