# 🛡️ Hate Speech Detection Streamlit App

A professional web application for detecting hate speech using advanced NLP with explainable AI.

## Features

- **Real-time Hate Speech Detection**: Classify text as hate speech or not
- **Batch File Processing**: Upload CSV files to analyze multiple texts at once (up to 200 rows)
- **Explainable AI**: See which words influenced the prediction
- **Token Importance Visualization**: Color-coded highlighting of important tokens
- **Probability Distribution**: Visual representation of model confidence
- **Performance Metrics**: View F1 score, accuracy, precision, recall, and confusion matrix for batch processing
- **Resource Monitoring**: Track CPU and memory usage during batch predictions
- **Professional UI**: Clean, modern interface with interactive elements

## Installation

1. Install the required packages:

```bash
uv sync
```

## Running the Application

Run the Streamlit app with:

```bash
uv run main.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### Single Text Analysis

1. **Enter Text**: Type or paste text into the main input area
2. **Optional Context**: Provide additional context or rationale (optional)
3. **Analyze**: Click the "🔍 Analyze Text" button
4. **View Results**: 
   - See the classification (Hate Speech or Not Hate Speech)
   - View confidence scores and probability distribution
   - Explore token importance visualization
   - Check which words influenced the decision

### Batch File Analysis

1. **Enable File Upload**: Check the "Enable File Upload" option in the sidebar
2. **Upload CSV File**: Click "Browse files" and select your CSV file
   - Required columns: `text`, `CF_Rationales`, `label`
   - Maximum recommended: 200 rows
3. **Preview Data**: Review the file statistics and preview
4. **Analyze**: Click the "🔍 Analyze Text" button
5. **View Results**:
   - Classification metrics (F1 score, accuracy, precision, recall)
   - Confusion matrix heatmap
   - CPU and memory usage statistics
   - Processing time and performance summary

### CSV File Format

Your CSV file should contain the following columns:
- `text`: The text to analyze for hate speech
- `CF_Rationales`: Contextual rationale or explanation (can be empty)
- `label`: Ground truth label (0 = not hate speech, 1 = hate speech)

Example:
```csv
text,CF_Rationales,label
"This is a sample text",Some context here,0
"Another example text",More context,1
```

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
- **Enable File Upload**: Toggle between single text and batch file processing
- **Show Token Importance**: Toggle token highlighting
- **Show Probability Distribution**: Toggle probability chart
- **Show Technical Details**: View raw model outputs

## Performance Optimizations

The application includes several optimizations for efficient batch processing:
- **Selective column loading**: Only loads required CSV columns to reduce memory usage
- **Optimized resource monitoring**: Samples CPU/memory every 10th prediction instead of every prediction
- **No blocking delays**: Removed sleep intervals from performance tracking
- **Memory efficient**: Processes up to 200 rows with minimal memory overhead (~15-20MB reduction)

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
