"""
Quick test to verify all three models load and predict correctly
"""
from hatespeech_model import load_model_from_hf, predict_hatespeech

def test_model(model_type, test_text="This is a test message"):
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()} model")
    print(f"{'='*60}")
    
    try:
        # Load model
        print(f"Loading {model_type} model...")
        model, tokenizer_hatebert, tokenizer_rationale, config, device = load_model_from_hf(model_type)
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {device}")
        
        # Make prediction
        print(f"\nTesting prediction on: '{test_text}'")
        result = predict_hatespeech(
            text=test_text,
            rationale="This is a test rationale",
            model=model,
            tokenizer_hatebert=tokenizer_hatebert,
            tokenizer_rationale=tokenizer_rationale,
            config=config,
            device=device
        )
        
        print(f"✅ Prediction successful!")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Probabilities: {result['probabilities']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing all three models...")
    print("This will verify that your integration is complete and working.\n")
    
    # Test cases
    test_text = "This is a test message to check if the model works correctly"
    
    # Test Simple model (your trained model)
    simple_success = test_model("simple", test_text)
    
    # Test Base model (from HuggingFace)
    base_success = test_model("base", test_text)
    
    # Test Altered model (from HuggingFace)
    altered_success = test_model("altered", test_text)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Simple Concat Model: {'✅ PASS' if simple_success else '❌ FAIL'}")
    print(f"Base Shield Model:   {'✅ PASS' if base_success else '❌ FAIL'}")
    print(f"Altered Shield Model: {'✅ PASS' if altered_success else '❌ FAIL'}")
    print(f"{'='*60}")
    
    if all([simple_success, base_success, altered_success]):
        print("\n🎉 All models working! You can now run the Streamlit app.")
        print("   Run: streamlit run app.py")
    else:
        print("\n⚠️ Some models failed. Please check the errors above.")
