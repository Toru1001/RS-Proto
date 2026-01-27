"""Final integration test"""
from hatespeech_model import load_model_from_hf, predict_hatespeech

print('Testing final integration...')

# Test Simple model
model, tok1, tok2, cfg, dev = load_model_from_hf('simple')
result = predict_hatespeech('This is hate speech trash', 'Negative language', model, tok1, tok2, cfg, dev)
print(f'\nSimple Model - Prediction: {result["prediction"]} (confidence: {result["confidence"]:.2%})')

# Test Base model  
model2, tok1_2, tok2_2, cfg2, dev2 = load_model_from_hf('base')
result2 = predict_hatespeech('This is hate speech trash', 'Negative language', model2, tok1_2, tok2_2, cfg2, dev2)
print(f'Base Model - Prediction: {result2["prediction"]} (confidence: {result2["confidence"]:.2%})')

print('\n✅ Everything works! Both models are integrated.')
