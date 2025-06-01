"""
Language detection utilities for the RNE chatbot.
"""

import re
import logging
from typing import List, Optional

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    logging.warning("langdetect not available. Using regex-based language detection.")
    LANGDETECT_AVAILABLE = False

class LanguageDetector:
    """Class for language detection and related utilities."""
    
    def __init__(self, supported_languages: Optional[List[str]] = None, default_language: str = 'fr'):
        """
        Initialize the language detector.
        
        Args:
            supported_languages: List of supported language codes.
            default_language: Default language to use if detection fails.
        """
        self.supported_languages = supported_languages or ['fr', 'ar']
        self.default_language = default_language
        
        # Regular expressions for language detection backup
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        self.french_keywords = {
            'quel', 'quelle', 'quels', 'quelles', 'comment', 'pourquoi', 'quand', 'où',
            'combien', 'est', 'sont', 'être', 'avoir', 'faire', 'aller', 'venir',
            'société', 'entreprise', 'création', 'capital', 'délai', 'document'
        }
        
        self.arabic_keywords = {
            'ما', 'ماذا', 'كيف', 'لماذا', 'متى', 'أين', 'كم', 'هل',
            'شركة', 'مؤسسة', 'تأسيس', 'رأس المال', 'وثائق', 'مدة'
        }
        
        logging.info(f"Initialized LanguageDetector with supported languages: {self.supported_languages}")
        
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to detect language.
            
        Returns:
            Detected language code or default language if detection fails.
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 2:
            return self.default_language
        
        text = text.strip()
        
        # First try using langdetect if available
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(text)
                
                # Map similar language codes
                if detected_lang in ['ar', 'arb']:
                    return 'ar' if 'ar' in self.supported_languages else self.default_language
                elif detected_lang == 'fr':
                    return 'fr' if 'fr' in self.supported_languages else self.default_language
                    
                # If detected language is supported, return it
                if detected_lang in self.supported_languages:
                    return detected_lang
                    
                # If detected language is not supported, try regex patterns
                return self._detect_with_patterns(text)
                
            except Exception as e:
                logging.debug(f"Langdetect failed: {e}. Falling back to pattern matching.")
                # Fallback to regex pattern matching
                return self._detect_with_patterns(text)
        else:
            # Use pattern-based detection
            return self._detect_with_patterns(text)
    
    def _detect_with_patterns(self, text: str) -> str:
        """
        Detect language using regex patterns and keyword matching.
        
        Args:
            text: Input text.
            
        Returns:
            Detected language code or default language.
        """
        text_lower = text.lower()
        
        # Check for Arabic characters first (most reliable)
        if self.arabic_pattern.search(text):
            return 'ar' if 'ar' in self.supported_languages else self.default_language
        
        # Count keyword matches for each language
        french_matches = sum(1 for word in self.french_keywords if word in text_lower)
        arabic_matches = sum(1 for word in self.arabic_keywords if word in text)
        
        # Determine language based on keyword matches
        if arabic_matches > french_matches and arabic_matches > 0:
            return 'ar' if 'ar' in self.supported_languages else self.default_language
        elif french_matches > 0:
            return 'fr' if 'fr' in self.supported_languages else self.default_language
        
        # Check for specific patterns
        # French pattern: Latin characters with accents
        if re.search(r'[àâäéèêëïîôöùûüÿç]', text_lower):
            return 'fr' if 'fr' in self.supported_languages else self.default_language
        
        # Default fallback
        return self.default_language
    
    def is_arabic(self, text: str) -> bool:
        """
        Check if text is in Arabic.
        
        Args:
            text: Input text.
            
        Returns:
            True if text is in Arabic, False otherwise.
        """
        return self.detect_language(text) == 'ar'
    
    def is_french(self, text: str) -> bool:
        """
        Check if text is in French.
        
        Args:
            text: Input text.
            
        Returns:
            True if text is in French, False otherwise.
        """
        return self.detect_language(text) == 'fr'
    
    def get_direction(self, language: str) -> str:
        """
        Get text direction for the given language.
        
        Args:
            language: Language code.
            
        Returns:
            'rtl' for right-to-left languages, 'ltr' otherwise.
        """
        rtl_languages = ['ar', 'he', 'fa', 'ur']
        return 'rtl' if language in rtl_languages else 'ltr'
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get the full name of a language from its code.
        
        Args:
            language_code: ISO language code.
            
        Returns:
            Full language name.
        """
        language_names = {
            'fr': 'Français',
            'ar': 'العربية',
            'en': 'English',
            'es': 'Español',
            'de': 'Deutsch'
        }
        
        return language_names.get(language_code, language_code.upper())
    
    def validate_language(self, language_code: str) -> str:
        """
        Validate and normalize a language code.
        
        Args:
            language_code: Language code to validate.
            
        Returns:
            Validated language code or default language if invalid.
        """
        if not language_code or not isinstance(language_code, str):
            return self.default_language
            
        # Normalize to lowercase
        lang = language_code.lower().strip()
        
        # Map common variations
        language_mappings = {
            'french': 'fr',
            'francais': 'fr',
            'français': 'fr',
            'arabic': 'ar',
            'arabe': 'ar',
            'العربية': 'ar'
        }
        
        # Check mappings first
        if lang in language_mappings:
            lang = language_mappings[lang]
        
        # Check if the language is supported
        if lang in self.supported_languages:
            return lang
        
        return self.default_language