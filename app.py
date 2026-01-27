import streamlit as st
from hatespeech_model import predict_hatespeech, load_model_from_hf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="🛡️ Hate Speech Detector",
    page_icon="🛡️",
    layout="wide"
)

# Cached model loading function
@st.cache_resource
def load_cached_model(model_type="altered", local_checkpoint_path=None):
    """Load and cache the model"""
    return load_model_from_hf(model_type=model_type, local_checkpoint_path=local_checkpoint_path)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .hate-speech {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .not-hate-speech {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛡️ Hate Speech Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced NLP model with explainable AI for detecting hate speech</div>', unsafe_allow_html=True)

# Model selection
col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    model_type = st.radio(
        "Select Model:",
        ["Altered Shield (Advanced)", "Base Shield (Simple)", "Simple Concat (Baseline)"],
        horizontal=True,
        help="Altered Shield: Full architecture with CNNs and attention (may have loading issues). Base Shield: HuggingFace baseline. Simple Concat: Your trained baseline model."
    )
    
if "Altered" in model_type:
    model_choice = "altered"
elif "Base" in model_type:
    model_choice = "base"
else:
    model_choice = "simple"

# Load model with spinner
with st.spinner('🔄 Loading model... This may take a moment on first run.'):
    try:
        model, tokenizer_hatebert, tokenizer_rationale, config, device = load_cached_model(model_choice)
        st.success(f'✅ {model_type} loaded successfully!')
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**Device:** {device.upper()}")
    st.markdown(f"**Max Length:** {config.get('max_length', 128)}")
    st.markdown(f"**CNN Filters:** {config.get('cnn_num_filters', 128)}")
    
    st.divider()
    
    show_rationale_viz = st.checkbox("Show Token Importance", value=True)
    show_probabilities = st.checkbox("Show Probability Distribution", value=True)
    show_details = st.checkbox("Show Technical Details", value=False)
    
    st.divider()
    st.subheader("💡 About")
    st.markdown("""
    This model uses:
    - **HateBERT** for hate speech understanding
    - **Multi-Scale CNN** for feature extraction
    - **Attention mechanisms** for interpretability
    """)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Text")
    user_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste text here to check for hate speech...",
        height=150,
        help="Enter any text and the model will classify it as hate speech or not"
    )
    
    optional_rationale = st.text_area(
        "Optional: Provide context or rationale (leave empty to use main text):",
        placeholder="Why might this be hate speech? (optional)",
        height=80
    )

with col2:
    st.subheader("📊 Quick Stats")
    if user_input:
        word_count = len(user_input.split())
        char_count = len(user_input)
        st.metric("Words", word_count)
        st.metric("Characters", char_count)
    else:
        st.info("Enter text to see statistics")

# Classification button
classify_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

if classify_button:
    if user_input.strip():
        with st.spinner('🔄 Analyzing text...'):
            # Get prediction
            start = time.time()
            result = predict_hatespeech(
                text=user_input,
                rationale=optional_rationale if optional_rationale else None,
                model=model,
                tokenizer_hatebert=tokenizer_hatebert,
                tokenizer_rationale=tokenizer_rationale,
                config=config,
                device=device
            )
            end = time.time()
            
            # Extract results
            prediction = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']
            rationale_scores = result['rationale_scores']
            tokens = result['tokens']
            processing_time = end - start
            
            # Display results
            st.divider()
            st.header("📈 Analysis Results")
            
            # Prediction box
            if prediction == 1:
                st.markdown(f'<div class="prediction-box hate-speech">🚨 HATE SPEECH DETECTED</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box not-hate-speech">✅ NOT HATE SPEECH</div>', 
                           unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence", f"{confidence:.1%}")
            with col2:
                st.metric("Not Hate Speech", f"{probabilities[0]:.1%}")
            with col3:
                st.metric("Hate Speech", f"{probabilities[1]:.1%}")
            with col4:
                st.metric("Processing Time", f"{processing_time:.3f}s")
            
            # Probability distribution chart
            if show_probabilities:
                st.subheader("📊 Probability Distribution")
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Not Hate Speech', 'Hate Speech'],
                        y=probabilities,
                        marker_color=['#66bb6a', '#ef5350'],
                        text=[f"{p:.1%}" for p in probabilities],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    yaxis_title="Probability",
                    yaxis_range=[0, 1],
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Token importance visualization
            if show_rationale_viz:
                st.subheader("🔍 Token Importance Analysis")
                st.caption("Highlighted words show which parts of the text influenced the prediction")
                
                # Filter out special tokens and create visualization
                token_importance = []
                html_output = "<div style='font-size: 18px; line-height: 2.5; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>"
                
                for token, score in zip(tokens, rationale_scores):
                    if token not in ['[CLS]', '[SEP]', '[PAD]']:
                        # Clean token
                        display_token = token.replace('##', '')
                        token_importance.append({'Token': display_token, 'Importance': score})
                        
                        # Color intensity based on score
                        alpha = min(score * 1.5, 1.0)  # Scale up visibility
                        if prediction == 1:  # Hate speech
                            color = f"rgba(239, 83, 80, {alpha:.2f})"
                        else:  # Not hate speech
                            color = f"rgba(102, 187, 106, {alpha:.2f})"
                        
                        html_output += f"<span style='background-color: {color}; padding: 4px 8px; margin: 2px; border-radius: 5px; display: inline-block;'>{display_token}</span> "
                
                html_output += "</div>"
                st.markdown(html_output, unsafe_allow_html=True)
                
                if prediction == 1:
                    st.caption("🔴 Darker red = Higher importance for hate speech detection")
                else:
                    st.caption("🟢 Darker green = Higher importance for non-hate speech classification")
                
                # Top important tokens
                st.subheader("📋 Top Important Tokens")
                df_importance = pd.DataFrame(token_importance)
                df_importance = df_importance.sort_values('Importance', ascending=False).head(10)
                df_importance['Importance'] = df_importance['Importance'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(
                    df_importance,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Technical details
            if show_details:
                st.subheader("🔧 Technical Details")
                with st.expander("View Model Outputs"):
                    st.json({
                        'prediction': int(prediction),
                        'confidence': float(confidence),
                        'probability_not_hate': float(probabilities[0]),
                        'probability_hate': float(probabilities[1]),
                        'num_tokens': len([t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]),
                        'device': device,
                        'model_config': {
                            'max_length': config.get('max_length'),
                            'cnn_filters': config.get('cnn_num_filters'),
                        }
                    })
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Examples section
st.divider()
st.subheader("💡 Try Example Texts")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Example: Hate Speech", use_container_width=True):
        st.session_state.example_text = "You people are worthless and should leave this country!"
        st.rerun()

with col2:
    if st.button("Example: Not Hate Speech", use_container_width=True):
        st.session_state.example_text = "I disagree with your opinion, but I respect your right to express it."
        st.rerun()

with col3:
    if st.button("Example: Borderline", use_container_width=True):
        st.session_state.example_text = "This policy is terrible and will hurt everyone involved."
        st.rerun()

if 'example_text' in st.session_state:
    st.info(f"**Example loaded:** {st.session_state.example_text}")
    st.caption("↑ Copy this text to the input box above and click 'Analyze Text'")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><b>Hate Speech Detection Model with Rationale Extraction</b></p>
        <p>Powered by HateBERT + Multi-Scale CNN + Attention Mechanisms</p>
        <p>Model trained with advanced regularization and early stopping for optimal performance</p>
    </div>
""", unsafe_allow_html=True)
