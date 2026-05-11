from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
import re
import json
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ==================== SETUP ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

print("\n" + "="*60)
print("Radhe Radhe! 🙏")
print("Loading Sentiment Analysis Models...")
print("="*60)

try:
    # Load preprocessing metadata
    print("Loading preprocessing...")
    with open(os.path.join(MODEL_DIR, "preprocessing_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    
    negations_dic = meta["negations_dic"]
    neg_pattern = re.compile(meta["neg_pattern_string"])
    print("  ✓ Preprocessing metadata loaded")
    
    # Load vectorizer
    print("Loading vectorizer...")
    with open(os.path.join(MODEL_DIR, "tfidf_combined_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    print(f"  ✓ TF-IDF vectorizer loaded ({len(tfidf.vocabulary_)} features)")
    
    # Load sentiment encoder
    print("Loading encoders...")
    with open(os.path.join(MODEL_DIR, "label_encoder_sentiment.pkl"), "rb") as f:
        le_senti = pickle.load(f)
    print(f"  ✓ Label encoder loaded (classes: {le_senti.classes_})")
    
    # Load models
    print("Loading ML models...")
    with open(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"), "rb") as f:
        nb = pickle.load(f)
    print(f"  ✓ Naive Bayes loaded")
    
    with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "rb") as f:
        svm = pickle.load(f)
    print(f"  ✓ SVM loaded")
    
    print("Loading neural network...")
    lstm = load_model(os.path.join(MODEL_DIR, "lstm_model.keras"), compile=False)
    print(f"  ✓ Neural Network loaded")
    
    # Load model info
    try:
        with open(os.path.join(MODEL_DIR, "model_info.json"), "r") as f:
            model_info = json.load(f)
        print(f"  ✓ Model info loaded")
        print(f"\n📊 Model Accuracies:")
        print(f"   • Naive Bayes: {model_info.get('nb_accuracy', 'N/A'):.2f}%")
        print(f"   • SVM: {model_info.get('svm_accuracy', 'N/A'):.2f}%")
        print(f"   • Neural Net: {model_info.get('lstm_accuracy', 'N/A'):.2f}%")
    except:
        print("  ⚠ Model info not found (not critical)")
    
    print("\n" + "="*60)
    print("✓ ALL MODELS LOADED SUCCESSFULLY!")
    print("="*60 + "\n")
    
    models_loaded = True

except Exception as e:
    print(f"\n✗ ERROR LOADING MODELS: {e}")
    print("="*60 + "\n")
    import traceback
    traceback.print_exc()
    models_loaded = False

# ==================== TEXT CLEANING FUNCTION ====================

# Sentiment keywords to preserve (even if short)
NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'garbage', 
                  'trash', 'waste', 'poor', 'broken', 'useless', 'disappointing', 
                  'disappointed', 'fail', 'failed', 'sucks', 'suck', 'rubbish', 'crap',
                  'junk', 'defective', 'pathetic', 'lousy', 'dreadful', 'horrendous'}
POSITIVE_WORDS = {'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'best',
                  'perfect', 'wonderful', 'fantastic', 'recommend', 'happy', 'satisfied',
                  'nice', 'super', 'outstanding', 'brilliant', 'superb', 'incredible'}

def clean_text(text, negations_dic, neg_pattern):
    """Clean and preprocess text - IMPROVED to keep sentiment words"""
    try:
        if not text or not isinstance(text, str):
            return ""
        
        # Replace special characters
        text = text.replace("Äú", '').replace("Äù", '').replace("‚Äô", "'")
        text = text.lower()
        
        # Handle negations
        text = neg_pattern.sub(lambda x: negations_dic.get(x.group(), x.group()), text)
        
        # Remove non-alphanumeric (except quotes)
        text = re.sub(r"[^a-zA-Z0-9'\"\s]", ' ', text)
        
        # Tokenize and filter - KEEP SENTIMENT WORDS regardless of length
        words = text.split()
        filtered_words = []
        for w in words:
            if w in NEGATIVE_WORDS or w in POSITIVE_WORDS:
                filtered_words.append(w)  # Keep sentiment keywords
            elif len(w) > 3:
                filtered_words.append(w)  # Keep longer words
        
        return ' '.join(filtered_words) if filtered_words else 'empty'
        
    except Exception as e:
        print(f"Error in text cleaning: {e}")
        # Fallback to basic cleaning
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        words = text.split()
        filtered_words = [w for w in words if w in NEGATIVE_WORDS or w in POSITIVE_WORDS or len(w) > 3]
        return ' '.join(filtered_words) if filtered_words else 'empty'

# ==================== PREDICTION FUNCTION ====================

def predict_sentiment(review_text, review_title=""):
    """
    Predict sentiment using ensemble of models
    
    Args:
        review_text (str): Main review text
        review_title (str): Review title (optional)
        
    Returns:
        dict: Prediction results
    """
    
    if not models_loaded:
        return {
            'success': False,
            'error': 'Models not loaded properly. Please check server logs.'
        }
    
    try:
        # Validate input
        if not review_text or not review_text.strip():
            return {
                'success': False,
                'error': 'Please enter review text!'
            }
        
        # Combine text and title
        combined_text = str(review_text).strip()
        if review_title and review_title.strip():
            combined_text += " " + str(review_title).strip()
        
        # Clean text
        cleaned_text = clean_text(combined_text, negations_dic, neg_pattern)
        
        if not cleaned_text or cleaned_text == 'empty':
            cleaned_text = 'product review'  # Fallback
        
        # Vectorize
        vec = tfidf.transform([cleaned_text])
        vec_dense = vec.toarray()
        
        # Get predictions from all models
        try:
            # Naive Bayes
            nb_proba = nb.predict_proba(vec_dense)[0]
            nb_pred = np.argmax(nb_proba)
            
            # SVM
            svm_proba = svm.predict_proba(vec_dense)[0]
            svm_pred = np.argmax(svm_proba)
            
            # Neural Network
            nn_proba = lstm.predict(vec_dense, verbose=0)[0]
            nn_pred = np.argmax(nn_proba)
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Model prediction error: {str(e)}'
            }
        
        # Ensemble voting (weighted average)
        # Weights: NB=35%, NN=35%, SVM=30%
        final_proba = (0.35 * nb_proba + 0.35 * nn_proba + 0.30 * svm_proba)
        
        # Normalize probabilities
        final_proba = final_proba / final_proba.sum()
        
        # KEYWORD BOOSTING: For short inputs with clear sentiment keywords
        original_lower = combined_text.lower()
        words_in_input = set(original_lower.split())
        
        # Check for negative keywords
        neg_found = words_in_input & NEGATIVE_WORDS
        pos_found = words_in_input & POSITIVE_WORDS
        
        # If input has negative keywords but NO positive keywords, boost negative
        if neg_found and not pos_found:
            # Boost negative probability (index 0 = Negative)
            boost = 0.4 * len(neg_found)  # More negative words = more boost
            final_proba[0] += boost
            final_proba = final_proba / final_proba.sum()  # Re-normalize
        
        # If input has positive keywords but NO negative keywords, boost positive
        elif pos_found and not neg_found:
            # Boost positive probability (index 2 = Positive)
            boost = 0.3 * len(pos_found)
            final_proba[2] += boost
            final_proba = final_proba / final_proba.sum()
        
        final_pred = np.argmax(final_proba)
        
        # Decode sentiment
        sentiment = le_senti.inverse_transform([final_pred])[0]
        confidence = float(final_proba[final_pred]) * 100
        
        # Get emoji and color
        if sentiment == 'Positive':
            emoji = '😊'
            color = '#4CAF50'
        elif sentiment == 'Negative':
            emoji = '😞'
            color = '#f44336'
        else:  # Neutral
            emoji = '😐'
            color = '#FF9800'
        
        # Individual predictions
        individual = {
            'naive_bayes': le_senti.inverse_transform([nb_pred])[0],
            'svm': le_senti.inverse_transform([svm_pred])[0],
            'neural_network': le_senti.inverse_transform([nn_pred])[0]
        }
        
        # Probability breakdown
        probabilities = {
            'negative': float(final_proba[0]) * 100,
            'neutral': float(final_proba[1]) * 100,
            'positive': float(final_proba[2]) * 100
        }
        
        return {
            'success': True,
            'sentiment': sentiment,
            'emoji': emoji,
            'color': color,
            'confidence': confidence,
            'individual_predictions': individual,
            'probabilities': probabilities,
            'cleaned_text': cleaned_text
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }

# ==================== ROUTES ====================

@app.route("/")
def index():
    """Render main page"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    
    if not models_loaded:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please restart the server.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received'
            }), 400
        
        review_text = data.get("review_text", "").strip()
        review_title = data.get("review_title", "").strip()
        
        if not review_text:
            return jsonify({
                'success': False,
                'error': 'Please enter review text!'
            }), 400
        
        # Make prediction
        result = predict_sentiment(review_text, review_title)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Route error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route("/examples", methods=["GET"])
def get_examples():
    """Get example reviews for testing"""
    examples = {
        'positive': [
            {
                'title': 'Excellent Product!',
                'text': 'This is an amazing tablet! Great value for money. My kids love it and the battery life is fantastic. Highly recommended for everyone!'
            },
            {
                'title': 'Best Purchase Ever',
                'text': 'I absolutely love this Echo device. The sound quality is superb and Alexa is so helpful. Worth every penny. Five stars!'
            },
            {
                'title': 'Perfect Gift',
                'text': 'Bought this as a gift for my mom and she is thrilled! Easy to use, very functional, and great features. Could not be happier!'
            }
        ],
        'negative': [
            {
                'title': 'Very Disappointed',
                'text': 'This product is terrible. It stopped working after just one week. Complete waste of money. Would not recommend to anyone at all.'
            },
            {
                'title': 'Poor Quality',
                'text': 'The device keeps freezing and the screen is not responsive. Customer service was unhelpful. Very frustrated with this purchase.'
            },
            {
                'title': 'Not Worth It',
                'text': 'Returned it immediately. The sound quality is awful and it does not connect to WiFi properly. Save your money and avoid this.'
            }
        ],
        'neutral': [
            {
                'title': 'It is okay',
                'text': 'The product works as described. Nothing special but gets the job done. Average quality for the price point.'
            },
            {
                'title': 'Mixed Feelings',
                'text': 'Some features are good, others not so much. It does what it is supposed to but could be better. Just okay overall.'
            },
            {
                'title': 'Average Product',
                'text': 'Not great, not terrible. It works fine for basic tasks. Would not say it is amazing but not bad either.'
            }
        ]
    }
    return jsonify(examples)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'message': 'Radhe Radhe! System is ' + ('running' if models_loaded else 'not ready')
    })

@app.route("/model-info", methods=["GET"])
def get_model_info():
    """Get model information"""
    try:
        with open(os.path.join(MODEL_DIR, "model_info.json"), "r") as f:
            info = json.load(f)
        return jsonify({
            'success': True,
            'info': info
        })
    except:
        return jsonify({
            'success': False,
            'error': 'Model info not available'
        }), 404

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ==================== RUN APP ====================

if __name__ == "__main__":
    if not models_loaded:
        print("\n⚠ WARNING: Models not loaded! Server will start but predictions will fail.")
        print("Please check the error messages above.\n")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting Flask server on http://localhost:{port}")
    print("Press CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)