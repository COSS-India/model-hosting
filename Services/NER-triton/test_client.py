#!/usr/bin/env python3
"""
Test client for IndicNER Triton deployment.
Tests NER inference for multiple Indian languages.
"""

import requests
import json
import sys


def test_ner_inference(text, lang_id, server_url="http://localhost:8300"):
    """
    Test NER inference for a given text and language.
    
    Args:
        text: Input text for NER
        lang_id: Language ID (ISO code)
        server_url: Triton server URL
    
    Returns:
        dict: NER prediction result
    """
    url = f"{server_url}/v2/models/ner/infer"
    
    payload = {
        "inputs": [
            {
                "name": "INPUT_TEXT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[text]]
            },
            {
                "name": "LANG_ID",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[lang_id]]
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
        ner_result = json.loads(output_text)
        
        return ner_result
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
        return None


def check_server_health(server_url="http://localhost:8300"):
    """Check if Triton server is ready."""
    try:
        response = requests.get(f"{server_url}/v2/health/ready", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    server_url = "http://localhost:8300"
    
    # Check server health
    print("=" * 80)
    print("IndicNER Triton Server Test Client")
    print("=" * 80)
    print()
    
    print("Checking server health...")
    if not check_server_health(server_url):
        print("❌ Server is not ready. Please start the Triton server first.")
        print("   Run: docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 ner-triton:latest")
        sys.exit(1)
    
    print("✅ Server is ready!")
    print()
    
    # Test cases for different Indian languages
    test_cases = [
        {
            "text": "राम दिल्ली में रहते हैं",
            "lang": "hi",
            "lang_name": "Hindi",
            "description": "Simple sentence with person and location"
        },
        {
            "text": "मुंबई भारत का सबसे बड़ा शहर है",
            "lang": "hi",
            "lang_name": "Hindi",
            "description": "Sentence with locations"
        },
        {
            "text": "সচিন তেন্ডুলকার ভারতের একজন বিখ্যাত ক্রিকেটার",
            "lang": "bn",
            "lang_name": "Bengali",
            "description": "Bengali sentence with person name"
        },
        {
            "text": "கோவை தமிழ்நாட்டில் உள்ளது",
            "lang": "ta",
            "lang_name": "Tamil",
            "description": "Tamil sentence with location"
        },
        {
            "text": "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ",
            "lang": "kn",
            "lang_name": "Kannada",
            "description": "Kannada sentence with location"
        },
        {
            "text": "હૈદરાબાદ તેલંગાણાની રાજધાની છે",
            "lang": "gu",
            "lang_name": "Gujarati",
            "description": "Gujarati sentence with location"
        },
    ]
    
    # Run tests
    print("Running NER tests...")
    print("=" * 80)
    print()
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{total_count}: {test_case['lang_name']} - {test_case['description']}")
        print(f"Input: {test_case['text']}")
        print(f"Language: {test_case['lang']}")
        
        result = test_ner_inference(test_case['text'], test_case['lang'], server_url)
        
        if result:
            print(f"✅ Success!")
            print(f"Source: {result.get('source', 'N/A')}")
            
            entities = result.get('nerPrediction', [])
            if entities:
                print(f"Found {len(entities)} entities:")
                for entity in entities:
                    print(f"  - {entity['entity']}: {entity['class']} (score: {entity['score']:.4f})")
            else:
                print("  No entities found")
            
            success_count += 1
        else:
            print(f"❌ Failed!")
        
        print()
        print("-" * 80)
        print()
    
    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    print()
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

