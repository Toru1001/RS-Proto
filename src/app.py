import streamlit as st
from hatespeech_model import predict_hatespeech, load_model_from_hf, predict_hatespeech_from_file, get_rationale_from_mistral, preprocess_rationale_mistral
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time


is_file_uploader_visible = False

# Page configuration
st.set_page_config(
    page_title="🛡️ Hate Speech Detector",
    page_icon="🛡️",
    layout="wide"
)

# Cached model loading function
@st.cache_resource
def load_cached_model(model_type="altered"):
    """Load and cache the model"""
    return load_model_from_hf(model_type=model_type)

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
st.markdown('<div class="sub-header">Comparing Base vs Enhanced models with explainable AI for detecting hate speech</div>', unsafe_allow_html=True)

# Load both models with spinner
with st.spinner('🔄 Loading models... This may take a moment on first run.'):
    try:
        base_model, base_tokenizer_hatebert, base_tokenizer_rationale, base_config, base_device = load_cached_model("base")
        enhanced_model, enhanced_tokenizer_hatebert, enhanced_tokenizer_rationale, enhanced_config, enhanced_device = load_cached_model("altered")
        st.success('✅ Base Shield and Enhanced Shield models loaded successfully!')
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**Device:** CPU")
    st.markdown(f"**Max Length:** 128")
    st.markdown(f"**CNN Filters:** 128")

    st.divider()
    st.subheader("🔍 File Upload")
    is_file_uploader_visible = st.checkbox("Enable File Upload", value=is_file_uploader_visible)

    
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
    if is_file_uploader_visible:
        user_input = None
        st.subheader("📂 Upload  File")
        uploaded_file = st.file_uploader(
            "Choose a text file (.csv) to analyze:",
            type=["csv"],
            help="Upload a text file containing the content you want to analyze for hate speech"
        )
        if uploaded_file is not None:
            try:
                file_content = pd.read_csv(uploaded_file, usecols=['text', 'CF_Rationales', 'label'])
                st.success("✅ File loaded successfully! Scroll down to analyze.")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                user_input = ""
    else:
        st.subheader("📝 Input Text/File")
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
    if is_file_uploader_visible and uploaded_file is not None:
        st.markdown(f"**Filename:** {uploaded_file.name}")
        st.markdown(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        file_rows = len(file_content)
        st.metric("Rows in File", file_rows)
    else:
        st.info("Enter text/file to see statistics")

# Classification button
classify_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

if classify_button:
    if user_input and user_input.strip():
        with st.spinner('🔄 Generating rationale from Mistral AI...'):
            # --- Step 1: Get rationale from Mistral ---
            try:
                raw_rationale = get_rationale_from_mistral(user_input)
                cleaned_rationale = preprocess_rationale_mistral(raw_rationale)
                print(f"Raw rationale from Mistral: {raw_rationale}")
            except Exception as e:
                st.error(f"❌ Error generating/processing rationale: {str(e)}")
                cleaned_rationale = user_input  # fallback to raw input
            
        with st.spinner('🔄 Analyzing text with models...'):
            # Run enhanced model
            enhanced_start = time.time()
            enhanced_model_result = predict_hatespeech(
                text=user_input,
                rationale=cleaned_rationale,  # use cleaned rationale
                model=enhanced_model,
                tokenizer_hatebert=enhanced_tokenizer_hatebert,
                tokenizer_rationale=enhanced_tokenizer_rationale,
                config=enhanced_config,
                device=enhanced_device,
                model_type="altered"
            )
            enhanced_end = time.time()

            # Run base model
            base_start = time.time()
            base_model_result = predict_hatespeech(
                text=user_input,
                rationale=cleaned_rationale,  # use cleaned rationale
                model=base_model,
                tokenizer_hatebert=base_tokenizer_hatebert,
                tokenizer_rationale=base_tokenizer_rationale,
                config=base_config,
                device=base_device,
                model_type="base"
            )
            base_end = time.time()
            
            # Extract results for both models
            base_prediction = base_model_result['prediction']
            base_confidence = base_model_result['confidence']
            base_probabilities = base_model_result['probabilities']
            base_processing_time = base_end - base_start
            
            enhanced_prediction = enhanced_model_result['prediction']
            enhanced_confidence = enhanced_model_result['confidence']
            enhanced_probabilities = enhanced_model_result['probabilities']
            enhanced_rationale_scores = enhanced_model_result['rationale_scores']
            enhanced_tokens = enhanced_model_result['tokens']
            enhanced_processing_time = enhanced_end - enhanced_start
            
            # Display results
            st.divider()
            st.header("📈 Analysis Results")
            
            # Side-by-side results columns
            base_col, enhanced_col = st.columns(2)
            
            # === BASE MODEL RESULTS (LEFT) ===
            with base_col:
                st.subheader("🔵 Base Shield Results")
                
                # Prediction box
                if base_prediction == 1:
                    st.markdown(f'<div class="prediction-box hate-speech">🚨 HATE SPEECH DETECTED</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box not-hate-speech">✅ NOT HATE SPEECH</div>', 
                               unsafe_allow_html=True)
                
                # Metrics
                st.metric("Confidence", f"{base_confidence:.1%}")
                base_m1, base_m2 = st.columns(2)
                with base_m1:
                    st.metric("Not Hate Speech", f"{base_probabilities[0]:.1%}")
                with base_m2:
                    st.metric("Hate Speech", f"{base_probabilities[1]:.1%}")
                st.metric("Processing Time", f"{base_processing_time:.3f}s")
                
                # Probability distribution chart
                if show_probabilities:
                    st.markdown("**📊 Probability Distribution**")
                    fig_base = go.Figure(data=[
                        go.Bar(
                            x=['Not Hate Speech', 'Hate Speech'],
                            y=base_probabilities,
                            marker_color=['#66bb6a', '#ef5350'],
                            text=[f"{p:.1%}" for p in base_probabilities],
                            textposition='auto',
                        )
                    ])
                    fig_base.update_layout(
                        yaxis_title="Probability",
                        yaxis_range=[0, 1],
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig_base, use_container_width=True)
                
                # Technical details for base
                if show_details:
                    with st.expander("View Base Model Outputs"):
                        st.json({
                            'prediction': int(base_prediction),
                            'confidence': float(base_confidence),
                            'probability_not_hate': float(base_probabilities[0]),
                            'probability_hate': float(base_probabilities[1]),
                            'device': 'cpu',
                            'model_config': {
                                'max_length': '128',
                            }
                        })
            
            # === ENHANCED MODEL RESULTS (RIGHT) ===
            with enhanced_col:
                st.subheader("🟢 Enhanced Shield Results")
                
                # Prediction box
                if enhanced_prediction == 1:
                    st.markdown(f'<div class="prediction-box hate-speech">🚨 HATE SPEECH DETECTED</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box not-hate-speech">✅ NOT HATE SPEECH</div>', 
                               unsafe_allow_html=True)
                
                # Metrics
                st.metric("Confidence", f"{enhanced_confidence:.1%}")
                enh_m1, enh_m2 = st.columns(2)
                with enh_m1:
                    st.metric("Not Hate Speech", f"{enhanced_probabilities[0]:.1%}")
                with enh_m2:
                    st.metric("Hate Speech", f"{enhanced_probabilities[1]:.1%}")
                st.metric("Processing Time", f"{enhanced_processing_time:.3f}s")
                
                # Probability distribution chart
                if show_probabilities:
                    st.markdown("**📊 Probability Distribution**")
                    fig_enhanced = go.Figure(data=[
                        go.Bar(
                            x=['Not Hate Speech', 'Hate Speech'],
                            y=enhanced_probabilities,
                            marker_color=['#66bb6a', '#ef5350'],
                            text=[f"{p:.1%}" for p in enhanced_probabilities],
                            textposition='auto',
                        )
                    ])
                    fig_enhanced.update_layout(
                        yaxis_title="Probability",
                        yaxis_range=[0, 1],
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig_enhanced, use_container_width=True)
                
                # Token importance visualization (only for enhanced)
                if show_rationale_viz:
                    st.markdown("**🔍 Token Importance Analysis**")
                    st.caption("Highlighted words show which parts influenced the prediction")
                    
                    # Filter out special tokens and create visualization
                    token_importance = []
                    html_output = "<div style='font-size: 16px; line-height: 2.2; padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>"
                    
                    for token, score in zip(enhanced_tokens, enhanced_rationale_scores):
                        if token not in ['[CLS]', '[SEP]', '[PAD]']:
                            # Clean token
                            display_token = token.replace('##', '')
                            token_importance.append({'Token': display_token, 'Importance': score})
                            
                            # Color intensity based on score
                            alpha = min(score * 1.5, 1.0)  # Scale up visibility
                            if enhanced_prediction == 1:  # Hate speech
                                color = f"rgba(239, 83, 80, {alpha:.2f})"
                            else:  # Not hate speech
                                color = f"rgba(102, 187, 106, {alpha:.2f})"
                            
                            html_output += f"<span style='background-color: {color}; padding: 3px 6px; margin: 1px; border-radius: 4px; display: inline-block;'>{display_token}</span> "
                    
                    html_output += "</div>"
                    st.markdown(html_output, unsafe_allow_html=True)
                    
                    if enhanced_prediction == 1:
                        st.caption("🔴 Darker red = Higher importance for hate speech detection")
                    else:
                        st.caption("🟢 Darker green = Higher importance for non-hate speech classification")
                    
                    # Top important tokens
                    st.markdown("**📋 Top Important Tokens**")
                    df_importance = pd.DataFrame(token_importance)
                    df_importance = df_importance.sort_values('Importance', ascending=False).head(10)
                    df_importance['Importance'] = df_importance['Importance'].apply(lambda x: f"{x:.4f}")
                    
                    st.dataframe(
                        df_importance,
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Technical details for enhanced
                if show_details:
                    with st.expander("View Enhanced Model Outputs"):
                        st.json({
                            'prediction': int(enhanced_prediction),
                            'confidence': float(enhanced_confidence),
                            'probability_not_hate': float(enhanced_probabilities[0]),
                            'probability_hate': float(enhanced_probabilities[1]),
                            'num_tokens': len([t for t in enhanced_tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]),
                            'device': 'cpu',
                            'model_config': {
                                'max_length': '128',
                                'cnn_filters': '128',
                            }
                        })
    if is_file_uploader_visible and uploaded_file is not None:
        st.markdown("**Preview:**")
        st.dataframe(file_content.head(3), use_container_width=True)
        with st.spinner('🔄 Analyzing file with both models... This may take a while for large files.'):
            # Run both models on the file
            # base_result = predict_hatespeech_from_file(...)  # Base model
            # enhanced_result = predict_hatespeech_from_file(...)  # Enhanced model
            enhanced_result = predict_hatespeech_from_file(
                text_list=file_content['text'].tolist(),
                rationale_list=file_content['CF_Rationales'].tolist(),
                true_label=file_content['label'].tolist(),
                model=enhanced_model,
                tokenizer_hatebert=enhanced_tokenizer_hatebert,
                tokenizer_rationale=enhanced_tokenizer_rationale,
                config=enhanced_config,
                device=enhanced_device,
                model_type="altered"
            )
            base_result = predict_hatespeech_from_file(
                text_list=file_content['text'].tolist(),
                rationale_list=file_content['CF_Rationales'].tolist(),
                true_label=file_content['label'].tolist(),
                model=base_model,  
                tokenizer_hatebert=base_tokenizer_hatebert,
                tokenizer_rationale=base_tokenizer_rationale,
                config=base_config,
                device=base_device,
                model_type="base"
            )
            st.success("✅ File analysis complete for both models!")
            st.divider()
            st.header("📊 Analysis Results - Model Comparison")
            
            # Side-by-side results columns
            base_file_col, enhanced_file_col = st.columns(2)
            
            # === BASE MODEL FILE RESULTS (LEFT) ===
            with base_file_col:
                st.subheader("🔵 Base Shield Results")
                
                # Performance Metrics
                st.markdown("**📈 Classification Metrics**")
                base_fm1, base_fm2 = st.columns(2)
                with base_fm1:
                    st.metric("F1 Score", f"{base_result['f1_score']:.4f}")
                    st.metric("Precision", f"{base_result['precision']:.4f}")
                with base_fm2:
                    st.metric("Accuracy", f"{base_result['accuracy']:.4f}")
                    st.metric("Recall", f"{base_result['recall']:.4f}")
                
                # Confusion Matrix Visualization
                st.markdown("**🎯 Confusion Matrix**")
                base_cm = base_result['confusion_matrix']
                fig_base_cm = go.Figure(data=go.Heatmap(
                    z=base_cm,
                    x=['Pred Not Hate', 'Pred Hate'],
                    y=['True Not Hate', 'True Hate'],
                    colorscale='Blues',
                    text=base_cm,
                    texttemplate='%{text}',
                    textfont={"size": 14},
                    showscale=False
                ))
                fig_base_cm.update_layout(height=300)
                st.plotly_chart(fig_base_cm, use_container_width=True)
                
                # Resource Usage
                st.markdown("**⚙️ Resource Usage**")
                base_cpu_col, base_mem_col = st.columns(2)
                with base_cpu_col:
                    st.metric("Avg CPU", f"{base_result['cpu_usage']:.2f}%")
                    st.metric("Peak CPU", f"{base_result['peak_cpu_usage']:.2f}%")
                with base_mem_col:
                    st.metric("Avg Memory", f"{base_result['memory_usage']:.2f} MB")
                    st.metric("Peak Memory", f"{base_result['peak_memory_usage']:.2f} MB")
                
                # Runtime
                st.markdown("**⏱️ Performance**")
                st.metric("Total Runtime", f"{base_result['runtime']:.2f}s")
                st.metric("Avg Time/Sample", f"{base_result['runtime']/file_rows:.3f}s")
            
            # === ENHANCED MODEL FILE RESULTS (RIGHT) ===
            with enhanced_file_col:
                st.subheader("🟢 Enhanced Shield Results")
                
                # Performance Metrics
                st.markdown("**📈 Classification Metrics**")
                enh_fm1, enh_fm2 = st.columns(2)
                with enh_fm1:
                    st.metric("F1 Score", f"{enhanced_result['f1_score']:.4f}")
                    st.metric("Precision", f"{enhanced_result['precision']:.4f}")
                with enh_fm2:
                    st.metric("Accuracy", f"{enhanced_result['accuracy']:.4f}")
                    st.metric("Recall", f"{enhanced_result['recall']:.4f}")
                
                # Confusion Matrix Visualization
                st.markdown("**🎯 Confusion Matrix**")
                enhanced_cm = enhanced_result['confusion_matrix']
                fig_enhanced_cm = go.Figure(data=go.Heatmap(
                    z=enhanced_cm,
                    x=['Pred Not Hate', 'Pred Hate'],
                    y=['True Not Hate', 'True Hate'],
                    colorscale='Greens',
                    text=enhanced_cm,
                    texttemplate='%{text}',
                    textfont={"size": 14},
                    showscale=False
                ))
                fig_enhanced_cm.update_layout(height=300)
                st.plotly_chart(fig_enhanced_cm, use_container_width=True)
                
                # Resource Usage
                st.markdown("**⚙️ Resource Usage**")
                enh_cpu_col, enh_mem_col = st.columns(2)
                with enh_cpu_col:
                    st.metric("Avg CPU", f"{enhanced_result['cpu_usage']:.2f}%")
                    st.metric("Peak CPU", f"{enhanced_result['peak_cpu_usage']:.2f}%")
                with enh_mem_col:
                    st.metric("Avg Memory", f"{enhanced_result['memory_usage']:.2f} MB")
                    st.metric("Peak Memory", f"{enhanced_result['peak_memory_usage']:.2f} MB")
                
                # Runtime
                st.markdown("**⏱️ Performance**")
                st.metric("Total Runtime", f"{enhanced_result['runtime']:.2f}s")
                st.metric("Avg Time/Sample", f"{enhanced_result['runtime']/file_rows:.3f}s")


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
        <p><b>Hate Speech Detection Model Comparison</b></p>
        <p>Base Shield vs Enhanced Shield (HateBERT + Multi-Scale CNN + Attention)</p>
        <p>Side-by-side comparison for performance evaluation</p>
    </div>
""", unsafe_allow_html=True)
