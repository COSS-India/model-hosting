#!/usr/bin/env python3
"""
Test client for IndicLID Triton deployment.
Tests language detection for multiple Indian languages in both native and roman scripts.
"""

import requests
import json
import sys


def test_language_detection(text, server_url="http://localhost:8000"):
    """
    Test language detection for a given text.
    
    Args:
        text: Input text for language detection
        server_url: Triton server URL
    
    Returns:
        dict: Language detection result
    """
    url = f"{server_url}/v2/models/indiclid/infer"
    
    payload = {
        "inputs": [
            {
                "name": "INPUT_TEXT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[text]]
            }
        ],
        "outputs": [
            {
                "name": "OUTPUT_TEXT"
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        output_text = result["outputs"][0]["data"][0]
        lid_result = json.loads(output_text)
        
        return lid_result
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
        return None


def check_server_health(server_url="http://localhost:8000"):
    """Check if Triton server is ready."""
    try:
        response = requests.get(f"{server_url}/v2/health/ready", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    server_url = "http://localhost:8000"
    
    # Check server health
    print("=" * 80)
    print("IndicLID Triton Server Test Client")
    print("=" * 80)
    print()
    
    print("Checking server health...")
    if not check_server_health(server_url):
        print("❌ Server is not ready. Please start the Triton server first.")
        print("   Run: docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 indiclid-triton:latest")
        sys.exit(1)
    
    print("✅ Server is ready!")
    print()
    
    # Test cases for different Indian languages (native and roman scripts)
    test_cases = [
        # Native script tests
        {
            "text": "मैं भारत से हूं",
            "expected_lang": "hin_Deva",
            "lang_name": "Hindi (Devanagari)",
            "description": "Simple Hindi sentence"
        },
        {
            "text": "আমি ভারত থেকে এসেছি",
            "expected_lang": "ben_Beng",
            "lang_name": "Bengali (Bengali script)",
            "description": "Simple Bengali sentence"
        },
        {
            "text": "நான் இந்தியாவிலிருந்து வந்தேன்",
            "expected_lang": "tam_Tamil",
            "lang_name": "Tamil (Tamil script)",
            "description": "Simple Tamil sentence"
        },
        {
            "text": "నేను భారతదేశం నుండి వచ్చాను",
            "expected_lang": "tel_Telu",
            "lang_name": "Telugu (Telugu script)",
            "description": "Simple Telugu sentence"
        },
        {
            "text": "ನಾನು ಭಾರತದಿಂದ ಬಂದಿದ್ದೇನೆ",
            "expected_lang": "kan_Knda",
            "lang_name": "Kannada (Kannada script)",
            "description": "Simple Kannada sentence"
        },
        {
            "text": "હું ભારતથી છું",
            "expected_lang": "guj_Gujr",
            "lang_name": "Gujarati (Gujarati script)",
            "description": "Simple Gujarati sentence"
        },
        {
            "text": "मी भारतातून आलो आहे",
            "expected_lang": "mar_Deva",
            "lang_name": "Marathi (Devanagari)",
            "description": "Simple Marathi sentence"
        },
        {
            "text": "ഞാൻ ഇന്ത്യയിൽ നിന്നാണ്",
            "expected_lang": "mal_Mlym",
            "lang_name": "Malayalam (Malayalam script)",
            "description": "Simple Malayalam sentence"
        },
        {
            "text": "ମୁଁ ଭାରତରୁ ଆସିଛି",
            "expected_lang": "ori_Orya",
            "lang_name": "Odia (Odia script)",
            "description": "Simple Odia sentence"
        },
        {
            "text": "ਮੈਂ ਭਾਰਤ ਤੋਂ ਹਾਂ",
            "expected_lang": "pan_Guru",
            "lang_name": "Punjabi (Gurmukhi script)",
            "description": "Simple Punjabi sentence"
        },
        {
            "text": "میں بھارت سے ہوں",
            "expected_lang": "urd_Arab",
            "lang_name": "Urdu (Perso-Arabic script)",
            "description": "Simple Urdu sentence"
        },
        
        # Roman script tests
        {
            "text": "main bharat se hoon",
            "expected_lang": "hin_Latn",
            "lang_name": "Hindi (Roman)",
            "description": "Romanized Hindi sentence"
        },
        {
            "text": "ami bharot theke esechhi",
            "expected_lang": "ben_Latn",
            "lang_name": "Bengali (Roman)",
            "description": "Romanized Bengali sentence"
        },
        {
            "text": "naan indiyaavilirundhu vandhen",
            "expected_lang": "tam_Latn",
            "lang_name": "Tamil (Roman)",
            "description": "Romanized Tamil sentence"
        },
        {
            "text": "nenu bharatadesam nundi vachchanu",
            "expected_lang": "tel_Latn",
            "lang_name": "Telugu (Roman)",
            "description": "Romanized Telugu sentence"
        },
        {
            "text": "naanu bharatadinda bandiddene",
            "expected_lang": "kan_Latn",
            "lang_name": "Kannada (Roman)",
            "description": "Romanized Kannada sentence"
        },
        {
            "text": "hun bharatthi chhun",
            "expected_lang": "guj_Latn",
            "lang_name": "Gujarati (Roman)",
            "description": "Romanized Gujarati sentence"
        },
        {
            "text": "mi bharatatun aalo aahe",
            "expected_lang": "mar_Latn",
            "lang_name": "Marathi (Roman)",
            "description": "Romanized Marathi sentence"
        },
        {
            "text": "njaan indhyayil ninnaanu",
            "expected_lang": "mal_Latn",
            "lang_name": "Malayalam (Roman)",
            "description": "Romanized Malayalam sentence"
        },
        
        # English test
        {
            "text": "I am from India",
            "expected_lang": "eng_Latn",
            "lang_name": "English",
            "description": "Simple English sentence"
        },
    ]
    
    # Run tests
    print("Running language detection tests...")
    print("=" * 80)
    print()
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{total_count}: {test_case['lang_name']} - {test_case['description']}")
        print(f"Input: {test_case['text']}")
        print(f"Expected: {test_case['expected_lang']}")
        
        result = test_language_detection(test_case['text'], server_url)
        
        if result:
            detected_lang = result.get('langCode', 'unknown')
            confidence = result.get('confidence', 0.0)
            model = result.get('model', 'unknown')
            
            # Check if detection matches expected
            is_correct = detected_lang == test_case['expected_lang']
            status = "✅" if is_correct else "⚠️"
            
            print(f"{status} Detected: {detected_lang} (confidence: {confidence:.4f}, model: {model})")
            
            if is_correct:
                success_count += 1
            else:
                print(f"   Note: Expected {test_case['expected_lang']} but got {detected_lang}")
        else:
            print(f"❌ Failed to get result!")
        
        print()
        print("-" * 80)
        print()
    
    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {total_count}")
    print(f"Correct detections: {success_count}")
    print(f"Incorrect/Failed: {total_count - success_count}")
    print(f"Accuracy: {success_count/total_count*100:.1f}%")
    print()
    
    if success_count >= total_count * 0.8:  # 80% threshold
        print("🎉 Tests passed with good accuracy!")
        return 0
    else:
        print("⚠️  Accuracy below threshold. Please check the logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

