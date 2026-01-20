# 🛡️ Hate Speech Detection Streamlit App

A professional web application for detecting hate speech using advanced NLP with explainable AI.

## Features

- **Real-time Hate Speech Detection**: Classify text as hate speech or not
- **Explainable AI**: See which words influenced the prediction
- **Token Importance Visualization**: Color-coded highlighting of important tokens
- **Probability Distribution**: Visual representation of model confidence
- **Professional UI**: Clean, modern interface with interactive elements

## Installation

1. Install the required packages:

```bash
uv sync
```

## Running the Application

Run the Streamlit app with:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Enter Text**: Type or paste text into the main input area
2. **Optional Context**: Provide additional context or rationale (optional)
3. **Analyze**: Click the "🔍 Analyze Text" button
4. **View Results**: 
   - See the classification (Hate Speech or Not Hate Speech)
   - View confidence scores and probability distribution
   - Explore token importance visualization
   - Check which words influenced the decision

## Model Information

- **Architecture**: HateBERT + Rationale BERT + Multi-Scale CNN + Attention
- **Model Repository**: [seffyehl/BetterShield](https://huggingface.co/seffyehl/BetterShield)
- **Training Details**: 
  - Batch Size: 8
  - Learning Rate: 1e-5
  - Weight Decay: 0.05
  - Best Validation Loss: 0.27

## Files

- `app.py` - Main Streamlit application
- `hatespeech_model.py` - Model loading and prediction functions
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Troubleshooting

### Model Loading Issues

If the model fails to load:
- Check your internet connection (model downloads from Hugging Face)
- Ensure you have enough disk space (~500MB for model files)
- The first run will take longer as it downloads the model

### Memory Issues

If you encounter memory errors:
- The model requires approximately 2GB of RAM
- Close other applications to free up memory
- Use CPU mode if GPU memory is limited

## Configuration

You can modify settings in the sidebar:
- **Show Token Importance**: Toggle token highlighting
- **Show Probability Distribution**: Toggle probability chart
- **Show Technical Details**: View raw model outputs

## Examples

Try the built-in examples:
- **Hate Speech Example**: Clear example of offensive content
- **Not Hate Speech Example**: Disagreement expressed respectfully
- **Borderline Example**: Strong criticism without hate

## Credits

Model trained using best practices:
- Early stopping to prevent overfitting
- Batch size optimization (8 vs 16)
- Proper regularization (weight decay, dropout)
- Extensive hyperparameter tuning

## License

MIT License
