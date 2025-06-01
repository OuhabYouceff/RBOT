"""
Script to verify that your data folder is set up correctly
and show what JSON files will be processed.
"""

import os
import json
import glob
from pathlib import Path

def verify_data_setup():
    """Verify the data folder setup and show what files will be processed."""
    
    print("🔍 RNE Chatbot Data Setup Verification")
    print("=" * 50)
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    print(f"Current directory: {current_dir}")
    print(f"Looking for data folder: {data_dir}")
    
    # Expected files
    expected_files = {
        "external_data.json": "Business and fiscal knowledge",
        "rne_laws.json": "RNE legal procedures",
        "fiscal_knowledge.json": "Additional fiscal information"
    }
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print("❌ Data folder not found!")
        print(f"Please create a folder called 'data' in: {current_dir}")
        print("Expected files:")
        for filename, description in expected_files.items():
            print(f"  - {filename}: {description}")
        return False
    
    print("✅ Data folder found!")
    
    # Find JSON files
    json_pattern = os.path.join(data_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print("❌ No JSON files found in data folder!")
        print(f"Please place your JSON files in: {data_dir}")
        print("Expected files:")
        for filename, description in expected_files.items():
            print(f"  - {filename}: {description}")
        return False
    
    print(f"✅ Found {len(json_files)} JSON file(s):")
    
    # Check for expected files
    found_files = {os.path.basename(f): f for f in json_files}
    
    for expected_file, description in expected_files.items():
        if expected_file in found_files:
            print(f"✅ {expected_file} - {description}")
        else:
            print(f"⚠️  {expected_file} - {description} (not found)")
    
    # Check for unexpected files
    unexpected_files = set(found_files.keys()) - set(expected_files.keys())
    if unexpected_files:
        print(f"\n📋 Additional files found:")
        for filename in unexpected_files:
            print(f"   📄 {filename}")
    
    total_items = 0
    valid_files = 0
    
    print(f"\n📊 File Analysis:")
    print("-" * 30)
    
    for i, json_file in enumerate(sorted(json_files), 1):
        filename = os.path.basename(json_file)
        file_size = os.path.getsize(json_file) / 1024  # Size in KB
        
        print(f"\n{i}. {filename} ({file_size:.1f} KB)")
        
        # Try to read and analyze the file
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Analyze the structure based on filename
            if filename == "external_data.json":
                print("   🏢 External business/fiscal data")
                items_count = analyze_external_data_format(data)
                total_items += items_count
                
            elif "rne" in filename.lower():
                print("   📋 RNE legal procedures")
                items_count = analyze_rne_format(data)
                total_items += items_count
                
            else:
                # Generic analysis
                if isinstance(data, list):
                    items_count = len(data)
                    total_items += items_count
                    print(f"   📋 List with {items_count} items")
                    
                    if data:
                        first_item = data[0]
                        if isinstance(first_item, dict):
                            if "combined_content" in first_item:
                                print("   📖 Format: Combined content")
                            elif "code" in first_item:
                                print("   📋 Format: RNE laws format")
                            else:
                                print(f"   📄 Format: Custom format with keys: {list(first_item.keys())[:5]}")
                                
                elif isinstance(data, dict):
                    items_count = analyze_dict_format(data)
                    total_items += items_count
            
            valid_files += 1
            print("   ✅ Valid JSON")
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Invalid JSON: {e}")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"   Valid JSON files: {valid_files}/{len(json_files)}")
    print(f"   Total data items: {total_items}")
    
    if valid_files == len(json_files) and total_items > 0:
        print(f"✅ Setup looks good! Ready to initialize indices.")
        print(f"\nNext steps:")
        print(f"1. Run: python init_data.py --verify-only")
        print(f"2. If verification passes, run: python init_data.py")
        return True
    else:
        print(f"❌ Please fix the issues above before proceeding.")
        return False

def analyze_external_data_format(data):
    """Analyze external_data.json format."""
    if isinstance(data, list):
        items_count = len(data)
        print(f"   📋 List with {items_count} items")
        
        if data:
            first_item = data[0]
            if isinstance(first_item, dict) and "combined_content" in first_item:
                print("   📖 Combined content format detected")
                # Try to categorize content
                sample_content = first_item["combined_content"][:200].lower()
                categories = []
                if "impôt" in sample_content or "fiscal" in sample_content:
                    categories.append("fiscalité")
                if "entreprise" in sample_content or "société" in sample_content:
                    categories.append("création_entreprise")
                if "contrat" in sample_content:
                    categories.append("juridique")
                if "employé" in sample_content or "travail" in sample_content:
                    categories.append("droit_travail")
                
                if categories:
                    print(f"   🏷️  Detected categories: {', '.join(categories)}")
                    
        return items_count
        
    elif isinstance(data, dict):
        if "combined_content" in data:
            print("   📖 Single combined content item")
            return 1
        else:
            return analyze_dict_format(data)
    
    return 0

def analyze_rne_format(data):
    """Analyze RNE laws format."""
    if isinstance(data, list):
        items_count = len(data)
        print(f"   📋 List with {items_count} RNE items")
        
        if data:
            first_item = data[0]
            if isinstance(first_item, dict):
                if "code" in first_item:
                    print("   📋 RNE code format detected")
                if "french_content" in first_item:
                    print("   🇫🇷 French content available")
                if "arabic_content" in first_item:
                    print("   🇹🇳 Arabic content available")
                    
        return items_count
        
    elif isinstance(data, dict):
        if "code" in data:
            print("   📋 Single RNE law item")
            return 1
        else:
            return analyze_dict_format(data)
    
    return 0

def analyze_dict_format(data):
    """Analyze generic dictionary format."""
    nested_items = 0
    for key, value in data.items():
        if isinstance(value, list):
            nested_items += len(value)
            print(f"   📁 Found {len(value)} items in '{key}'")
    
    if nested_items > 0:
        print(f"   📁 Structured data with {nested_items} nested items")
        return nested_items
    else:
        print(f"   📄 Single item with keys: {list(data.keys())[:5]}")
        return 1

def show_folder_structure():
    """Show the expected folder structure."""
    print("\n📁 Expected folder structure:")
    print("your-project/")
    print("├── data/")
    print("│   ├── external_data.json      # Business and fiscal knowledge")
    print("│   ├── rne_laws.json           # RNE legal procedures")
    print("│   └── fiscal_knowledge.json   # Additional fiscal information")
    print("├── config.py")
    print("├── init_data.py")
    print("├── verify_setup.py")
    print("└── preprocessing/")
    print("    └── data_loader.py")
    print("\n🎯 Priority files:")
    print("  1. external_data.json - Main business/fiscal knowledge base")
    print("  2. rne_laws.json - Legal procedures for business registration")
    print("  3. Any additional JSON files with relevant knowledge")

if __name__ == "__main__":
    success = verify_data_setup()
    
    if not success:
        show_folder_structure()
        print(f"\n💡 Tips:")
        print(f"- Make sure your JSON files are valid")
        print(f"- Files can contain 'combined_content' or structured RNE data")
        print(f"- The system will automatically detect and process different formats")