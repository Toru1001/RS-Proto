"""
Test script for loading and using the ConcatModel from combined-baseline notebook
This script verifies that your trained model checkpoint works correctly.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import os

# ==================== MODEL ARCHITECTURE ====================
class ProjectionMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, 2)
        )
    
    def forward(self, x):
        return self.layers(x)


class ConcatModel(nn.Module):
    """
    Simple concatenation model from combined-baseline notebook.
    Uses dynamic LayerNorm on embeddings before concatenation.
    """
    def __init__(self, hatebert_model, additional_model, projection_mlp, freeze_additional_model=True):
        super().__init__()
        self.hatebert_model = hatebert_model
        self.additional_model = additional_model
        self.projection_mlp = projection_mlp
        
        if freeze_additional_model:
            for param in self.additional_model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, additional_input_ids, additional_attention_mask):
        device = input_ids.device
        
        # Forward pass through the HateBERT model
        hatebert_outputs = self.hatebert_model(input_ids=input_ids, attention_mask=attention_mask)
        hatebert_embeddings = hatebert_outputs.last_hidden_state[:, 0, :]  # CLS token
        hatebert_embeddings = torch.nn.LayerNorm(hatebert_embeddings.size()[1:]).to(device)(hatebert_embeddings)
        
        # Forward pass through the Additional Model
        additional_outputs = self.additional_model(input_ids=additional_input_ids, 
                                                   attention_mask=additional_attention_mask)
        additional_embeddings = additional_outputs.last_hidden_state[:, 0, :]  # CLS token
        additional_embeddings = torch.nn.LayerNorm(additional_embeddings.size()[1:]).to(device)(additional_embeddings)
        
        # Concatenate the embeddings
        concatenated_embeddings = torch.cat((hatebert_embeddings, additional_embeddings), dim=1)
        
        # Project concatenated embeddings
        projected_embeddings = self.projection_mlp(concatenated_embeddings)
        
        return projected_embeddings


# ==================== LOAD MODEL FUNCTION ====================
def load_concat_model(checkpoint_path, device='cpu'):
    """
    Load the ConcatModel from checkpoint
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: 'cpu' or 'cuda'
    
    Returns:
        model, tokenizer_hatebert, tokenizer_bert, checkpoint_info
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint (weights_only=False needed for older checkpoints with numpy arrays)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Initialize base models
    print("Loading HateBERT and BERT models...")
    hatebert_model = BertModel.from_pretrained("GroNLP/HateBERT")
    additional_model = BertModel.from_pretrained("bert-base-uncased")
    projection_mlp = ProjectionMLP(input_size=1536, output_size=512)
    
    # Create model
    model = ConcatModel(
        hatebert_model=hatebert_model,
        additional_model=additional_model,
        projection_mlp=projection_mlp,
        freeze_additional_model=True
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load tokenizers
    tokenizer_hatebert = BertTokenizer.from_pretrained("GroNLP/HateBERT")
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Extract checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'dataset': checkpoint.get('dataset', 'unknown'),
        'freeze': checkpoint.get('freeze', 'unknown'),
        'seed': checkpoint.get('seed', 'unknown'),
        'history': checkpoint.get('history', {})
    }
    
    print(f"✅ Model loaded successfully!")
    print(f"   Dataset: {checkpoint_info['dataset']}")
    print(f"   Epochs: {checkpoint_info['epoch']}")
    print(f"   Seed: {checkpoint_info['seed']}")
    
    return model, tokenizer_hatebert, tokenizer_bert, checkpoint_info


# ==================== PREDICTION FUNCTION ====================
def predict(model, tokenizer_hatebert, tokenizer_bert, text, rationale=None, 
            max_length=512, device='cpu'):
    """
    Predict hate speech for given text
    
    Args:
        model: Loaded ConcatModel
        tokenizer_hatebert: HateBERT tokenizer
        tokenizer_bert: BERT tokenizer
        text: Input text to classify
        rationale: Optional rationale text (if None, uses the input text)
        max_length: Maximum sequence length
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with prediction results
    """
    model.eval()
    
    # Use text as rationale if not provided
    if rationale is None:
        rationale = text
    
    # Tokenize inputs
    encoding = tokenizer_hatebert(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    additional_encoding = tokenizer_bert(
        rationale,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    add_input_ids = additional_encoding['input_ids'].to(device)
    add_attention_mask = additional_encoding['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            additional_input_ids=add_input_ids,
            additional_attention_mask=add_attention_mask
        )
        
        # Get probabilities
        probs = torch.softmax(outputs, dim=1)
        prediction = outputs.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
    
    return {
        'prediction': prediction,
        'label': 'Hate Speech' if prediction == 1 else 'Not Hate Speech',
        'confidence': confidence,
        'probabilities': {
            'not_hate': probs[0, 0].item(),
            'hate': probs[0, 1].item()
        }
    }


# ==================== MAIN TEST SCRIPT ====================
def main():
    # Configuration
    CHECKPOINT_PATH = ".models/concat_model_reddit_85-15_epochs3_seed42.pth"
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ ERROR: Checkpoint file not found!")
        print(f"   Looking for: {CHECKPOINT_PATH}")
        print("\n📝 Please update CHECKPOINT_PATH in the script to point to your .pth file")
        print("   Expected format: concat_model_{dataset}_dataof8515_epochs{num}_seed{seed}.pth")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("-" * 60)
    
    # Load model
    model, tokenizer_hatebert, tokenizer_bert, checkpoint_info = load_concat_model(
        CHECKPOINT_PATH, 
        device=device
    )
    
    print("-" * 60)
    print("\n🧪 Running test predictions...\n")
    
    # Test cases
    test_cases = [
        {
            "text": "This is a normal message",
            "rationale": "This appears to be a neutral, non-offensive statement"
        },
        {
            "text": "I hate you and everything you stand for!",
            "rationale": "This contains aggressive language and personal attacks"
        },
        {
            "text": "Great work everyone, keep it up!",
            "rationale": "This is positive encouragement"
        },
        {
            "text": "You're worthless trash",
            "rationale": "This is derogatory and insulting language"
        }
    ]
    
    # Run predictions
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"  Text: '{test_case['text']}'")
        
        result = predict(
            model=model,
            tokenizer_hatebert=tokenizer_hatebert,
            tokenizer_bert=tokenizer_bert,
            text=test_case['text'],
            rationale=test_case['rationale'],
            device=device
        )
        
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities: Not Hate={result['probabilities']['not_hate']:.4f}, "
              f"Hate={result['probabilities']['hate']:.4f}")
        print()
    
    print("-" * 60)
    print("✅ All tests completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Verify predictions look reasonable")
    print("   2. Integrate this model into hatespeech_model.py")
    print("   3. Update app.py to include this model as an option")


if __name__ == "__main__":
    main()
