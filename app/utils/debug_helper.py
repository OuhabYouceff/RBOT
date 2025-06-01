"""
Debug script to identify the source of the 500 error.
Run this step by step to isolate the issue.
"""

import sys
import traceback
import os
import json
import logging

def test_imports():
    """Test 1: Check if all imports work."""
    print("=== Testing Imports ===")
    
    try:
        import openai
        print("✓ openai imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import openai: {e}")
        return False
    
    try:
        from app.utils.config import OPENAI_API_KEY, LLM_MODEL, SYSTEM_PROMPT, MAX_CONTEXT_LENGTH
        print("✓ Config imported successfully")
        print(f"  - API Key exists: {'Yes' if OPENAI_API_KEY else 'No'}")
        print(f"  - Model: {LLM_MODEL}")
        print(f"  - System prompt length: {len(SYSTEM_PROMPT) if SYSTEM_PROMPT else 0}")
        print(f"  - Max context length: {MAX_CONTEXT_LENGTH}")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        print("Check if app/utils/config.py exists and has the required variables")
        return False
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False
    
    return True

def test_openai_client():
    """Test 2: Check if OpenAI client initializes."""
    print("\n=== Testing OpenAI Client ===")
    
    try:
        import openai
        from app.utils.config import OPENAI_API_KEY, LLM_MODEL
        
        if not OPENAI_API_KEY:
            print("✗ OPENAI_API_KEY is empty")
            return False
        
        if len(OPENAI_API_KEY) < 10:
            print("✗ OPENAI_API_KEY seems too short")
            return False
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("✓ OpenAI client created successfully")
        
        # Test a simple API call
        print("Testing API call...")
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✓ OpenAI API call successful")
        print(f"  - Response: {completion.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI client error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_enhanced_client():
    """Test 3: Check if enhanced client works."""
    print("\n=== Testing Enhanced Client ===")
    
    try:
        from app.services.openai_client import OpenAIClient, ResponseType, FollowUpResponse, DirectResponse
        
        print("✓ All classes imported successfully")
        
        client = OpenAIClient()
        print("✓ Enhanced client created successfully")
        
        # Test vague question detection
        print("Testing vague question detection...")
        is_vague, category = client.analyze_question_specificity("Quel est le capital minimum ?")
        print(f"✓ Vague detection works: is_vague={is_vague}, category={category}")
        
        # Test with a specific question
        is_vague2, category2 = client.analyze_question_specificity("Quel est le capital minimum pour une SARL ?")
        print(f"✓ Specific question test: is_vague={is_vague2}, category={category2}")
        
        # Test response generation with empty context
        print("Testing response generation...")
        response = client.generate_response(
            query="Test question",
            context=[],
            language='fr',
            force_direct=True
        )
        
        print(f"✓ Response generation works: type={type(response)}")
        print(f"  - Response type: {response.response_type}")
        
        if hasattr(response, 'response'):
            print(f"  - Response content: {response.response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced client error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_prompt_templates():
    """Test 4: Check if prompt templates work."""
    print("\n=== Testing Prompt Templates ===")
    
    try:
        from app.utils.prompt_templates import (
            SYSTEM_PROMPT_FR,
            SYSTEM_PROMPT_AR,
            format_context,
            get_no_results_response,
            format_final_response
        )
        
        print("✓ All prompt templates imported successfully")
        
        # Test format_context with empty documents
        context = format_context([], 'fr')
        print(f"✓ Empty context formatting works: {context}")
        
        # Test no results response
        no_results_fr = get_no_results_response('fr')
        no_results_ar = get_no_results_response('ar')
        print("✓ No results responses work for both languages")
        
        # Test format_final_response
        final_response = format_final_response(
            "Test question",
            "Test answer",
            [],
            'fr'
        )
        print("✓ Final response formatting works")
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt templates error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_flask_app():
    """Test 5: Check your Flask app structure."""
    print("\n=== Testing Flask App ===")
    
    # Show current directory and files
    print(f"Current directory: {os.getcwd()}")
    
    # Look for app directory structure
    if os.path.exists('app'):
        print("✓ Found app directory")
        app_contents = os.listdir('app')
        print(f"  - App contents: {app_contents}")
        
        # Check for main.py
        if 'main.py' in app_contents:
            print("✓ Found app/main.py")
        else:
            print("✗ app/main.py not found")
    else:
        print("✗ app directory not found")
    
    # Look for common app files in root
    app_files = ['app.py', 'main.py', 'server.py', 'run.py']
    found_app = None
    
    for app_file in app_files:
        if os.path.exists(app_file):
            found_app = app_file
            print(f"✓ Found root app file: {found_app}")
            break
    
    if found_app:
        # Try to read and check the structure
        try:
            with open(found_app, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if '/api/chat' in content:
                print("✓ Found /api/chat endpoint")
            else:
                print("✗ /api/chat endpoint not found")
                
            if '@app.route' in content or '@router' in content:
                print("✓ Found route decorators")
            else:
                print("✗ No route decorators found")
                
            if 'Flask' in content:
                print("✓ Flask import found")
            else:
                print("✗ Flask import not found")
                
        except Exception as e:
            print(f"✗ Error reading app file: {e}")
    else:
        print("✗ No main app file found")
    
    return found_app is not None or os.path.exists('app')

def test_request_format():
    """Test 6: Check expected request format."""
    print("\n=== Testing Request Format ===")
    
    # Show expected request format
    expected_formats = {
        "simple": {
            "query": "Quel est le capital minimum ?",
            "language": "fr"
        },
        "with_context": {
            "query": "Quel est le capital minimum ?",
            "language": "fr",
            "context": {}
        },
        "follow_up": {
            "query": "Quel est le capital minimum ?",
            "language": "fr",
            "selected_option": "SARL (Société à Responsabilité Limitée)",
            "is_follow_up": True
        }
    }
    
    print("Expected request formats:")
    for format_name, format_data in expected_formats.items():
        print(f"\n{format_name.upper()}:")
        print(json.dumps(format_data, indent=2, ensure_ascii=False))
    
    # Test JSON parsing
    try:
        for format_name, format_data in expected_formats.items():
            json_str = json.dumps(format_data, ensure_ascii=False)
            parsed = json.loads(json_str)
            print(f"✓ {format_name} JSON parsing works")
        return True
    except Exception as e:
        print(f"✗ JSON parsing error: {e}")
        return False

def test_full_integration():
    """Test 7: Full integration test."""
    print("\n=== Testing Full Integration ===")
    
    try:
        from app.services.openai_client import OpenAIClient, DirectResponse, FollowUpResponse
        from app.utils.prompt_templates import format_context
        
        # Create client
        client = OpenAIClient()
        
        # Test with a vague question (should return FollowUpResponse)
        print("Testing vague question...")
        vague_response = client.generate_response(
            query="Quel est le capital minimum ?",
            context=[],
            language='fr'
        )
        
        if isinstance(vague_response, FollowUpResponse):
            print("✓ Vague question correctly detected")
            print(f"  - Main response: {vague_response.main_response[:50]}...")
            print(f"  - Follow-up question: {vague_response.follow_up_question}")
            print(f"  - Options count: {len(vague_response.options)}")
        else:
            print("✗ Vague question not detected correctly")
            return False
        
        # Test with a specific question (should return DirectResponse)
        print("\nTesting specific question...")
        specific_response = client.generate_response(
            query="Quel est le capital minimum pour une SARL ?",
            context=[],
            language='fr'
        )
        
        if isinstance(specific_response, DirectResponse):
            print("✓ Specific question correctly processed")
            print(f"  - Response: {specific_response.response[:100]}...")
        else:
            print("✗ Specific question not processed correctly")
            return False
        
        print("✓ Full integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def create_minimal_working_example():
    """Create a minimal working example."""
    print("\n=== Creating Minimal Working Example ===")
    
    minimal_app = '''
from flask import Flask, request, jsonify
import traceback
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        logging.info("Received request to /api/chat")
        
        # Get request data
        data = request.get_json()
        logging.info(f"Request data: {data}")
        
        if not data:
            return jsonify({"error": "No JSON data"}), 400
        
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        language = data.get('language', 'fr')
        
        # Try to use the actual OpenAI client
        try:
            from app.services.openai_client import OpenAIClient
            from app.services.openai_client import DirectResponse, FollowUpResponse
            
            client = OpenAIClient()
            response_obj = client.generate_response(
                query=query,
                context=[],
                language=language
            )
            
            if isinstance(response_obj, FollowUpResponse):
                response = {
                    "type": "clarification_needed",
                    "response": response_obj.main_response,
                    "follow_up_question": response_obj.follow_up_question,
                    "options": response_obj.options,
                    "context": {"awaiting_clarification": True}
                }
            elif isinstance(response_obj, DirectResponse):
                response = {
                    "type": "direct_answer",
                    "response": response_obj.response,
                    "context": {"awaiting_clarification": False}
                }
            else:
                response = {
                    "type": "direct_answer",
                    "response": str(response_obj),
                    "context": {"awaiting_clarification": False}
                }
                
        except Exception as openai_error:
            logging.error(f"OpenAI client error: {openai_error}")
            # Fallback to simple response
            response = {
                "type": "direct_answer",
                "response": f"Simple test response for: {query}",
                "context": {"awaiting_clarification": False}
            }
        
        logging.info(f"Sending response: {response}")
        return jsonify({"success": True, "response": response})
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "API is running"})

if __name__ == '__main__':
    print("Starting minimal test app...")
    app.run(debug=True, host='127.0.0.1', port=5000)
'''
    
    with open('minimal_test_app.py', 'w', encoding='utf-8') as f:
        f.write(minimal_app)
    
    print("✓ Created minimal_test_app.py")
    print("\nTo test:")
    print("1. Run: python minimal_test_app.py")
    print("2. Test health: curl http://127.0.0.1:5000/api/health")
    print("3. Test chat: curl -X POST http://127.0.0.1:5000/api/chat -H 'Content-Type: application/json' -d '{\"query\":\"Quel est le capital minimum ?\", \"language\":\"fr\"}'")

def create_test_config():
    """Create a test config file if missing."""
    print("\n=== Creating Test Config ===")
    
    config_path = 'app/utils/config.py'
    
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        test_config = '''
"""
Configuration file for RNE Chatbot.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# System Prompt
SYSTEM_PROMPT = """
Tu es un assistant juridique spécialisé dans les lois du Registre National des Entreprises (RNE) en Tunisie.
Ta mission est de fournir des informations précises et utiles basées sur la documentation officielle du RNE.
"""

# Context Configuration
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))

# Database Configuration (if needed)
DATABASE_URL = os.getenv("DATABASE_URL", "")

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "5000"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"
'''
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(test_config)
        
        print(f"✓ Created test config at {config_path}")
        print("⚠️  Remember to add your OPENAI_API_KEY to .env file")
    else:
        print(f"✓ Config already exists at {config_path}")

def main():
    """Run all tests."""
    print("🚀 Starting Comprehensive Debugging Process...\n")
    
    # Create test config if missing
    create_test_config()
    
    # Run tests in order
    tests = [
        ("Imports", test_imports),
        ("OpenAI Client", test_openai_client),
        ("Enhanced Client", test_enhanced_client),
        ("Prompt Templates", test_prompt_templates),
        ("Flask App", test_flask_app),
        ("Request Format", test_request_format),
        ("Full Integration", test_full_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"Running {test_name} Test...")
            print('='*50)
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup looks good.")
    else:
        print(f"\n🔧 Next steps based on failed tests:")
        
        failed_tests = [name for name, result in results if not result]
        
        if "Imports" in failed_tests:
            print("1. Fix import issues - check if all required packages are installed")
            print("   pip install openai python-dotenv")
        elif "OpenAI Client" in failed_tests:
            print("1. Check OpenAI API key in .env file")
            print("2. Verify API key is valid and has credits")
        elif "Enhanced Client" in failed_tests:
            print("1. Check the enhanced client implementation")
            print("2. Verify all classes are properly defined")
        elif "Prompt Templates" in failed_tests:
            print("1. Check prompt templates implementation")
        elif "Flask App" in failed_tests:
            print("1. Check Flask app structure and main.py")
        else:
            print("1. Check specific test failures above")
        
        print(f"\n🛠️  Creating minimal working example...")
        create_minimal_working_example()

if __name__ == "__main__":
    main()