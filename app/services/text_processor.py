"""
Text preprocessing utilities for the RNE chatbot.
"""

import re
import logging
from typing import List, Optional

# Try to import language detection and NLTK
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    logging.warning("langdetect not available. Language detection will default to French.")
    LANGDETECT_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    
    # Try to find stopwords, download if not available
    try:
        nltk.data.find('corpora/stopwords')
        NLTK_STOPWORDS_AVAILABLE = True
    except LookupError:
        logging.info("NLTK stopwords not found. Downloading...")
        try:
            nltk.download('stopwords', quiet=True)
            NLTK_STOPWORDS_AVAILABLE = True
        except Exception as e:
            logging.warning(f"Failed to download NLTK stopwords: {e}")
            NLTK_STOPWORDS_AVAILABLE = False
            
except ImportError:
    logging.warning("NLTK not available. Will use basic stopword filtering.")
    NLTK_STOPWORDS_AVAILABLE = False

class TextProcessor:
    """Class for text preprocessing operations."""
    
    def __init__(self):
        """Initialize the text processor."""
        # Define basic French stopwords if NLTK is not available
        self.basic_fr_stopwords = {
            'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'de', 'du', 'à', 'au', 'aux',
            'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 
            'que', 'qui', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi', 
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'on',
            'pour', 'par', 'en', 'dans', 'sur', 'sous', 'avec', 'sans', 'chez', 'entre',
            'est', 'sont', 'était', 'étaient', 'être', 'avoir', 'a', 'ai', 'as', 'avons', 'avez', 'ont'
        }
        
        # Define basic Arabic stopwords if NLTK is not available
        self.basic_ar_stopwords = {
            'من', 'إلى', 'عن', 'على', 'في', 'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك',
            'الذي', 'التي', 'الذين', 'اللواتي', 'أنا', 'أنت', 'هو', 'هي', 'نحن', 'أنتم', 'هم', 'هن',
            'كان', 'كانت', 'كانوا', 'يكون', 'تكون', 'يكونوا', 'كن', 'أن', 'لأن', 'لكن', 'إذا', 'لو',
            'قد', 'لقد', 'قال', 'قالت', 'يقول', 'تقول', 'أو', 'أم', 'أي', 'كل', 'بعض', 'جميع'
        }
        
        # Use NLTK stopwords if available, otherwise use basic ones
        if NLTK_STOPWORDS_AVAILABLE:
            try:
                self.fr_stopwords = set(stopwords.words('french'))
                # Add our basic stopwords to NLTK's list
                self.fr_stopwords.update(self.basic_fr_stopwords)
                
                self.ar_stopwords = set(stopwords.words('arabic'))
                # Add our basic stopwords to NLTK's list
                self.ar_stopwords.update(self.basic_ar_stopwords)
                
                logging.info("Using NLTK stopwords with custom additions")
            except Exception as e:
                logging.warning(f"Error loading NLTK stopwords: {e}. Using basic stopwords instead.")
                self.fr_stopwords = self.basic_fr_stopwords
                self.ar_stopwords = self.basic_ar_stopwords
        else:
            self.fr_stopwords = self.basic_fr_stopwords
            self.ar_stopwords = self.basic_ar_stopwords
            logging.info("Using basic stopwords")
        
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text.
            
        Returns:
            Language code ('fr', 'ar', or 'en' if not detected).
        """
        if not text or not text.strip():
            return 'fr'  # Default to French
            
        if not LANGDETECT_AVAILABLE:
            # Simple heuristic: if text contains Arabic characters, assume Arabic
            if re.search(r'[\u0600-\u06FF]', text):
                return 'ar'
            return 'fr'
            
        try:
            lang = detect(text)
            if lang == 'fr':
                return 'fr'
            elif lang in ['ar', 'arb']:
                return 'ar'
            return lang
        except Exception as e:
            logging.debug(f"Language detection failed: {e}. Defaulting to French.")
            # Fallback heuristic
            if re.search(r'[\u0600-\u06FF]', text):
                return 'ar'
            return 'fr'
    
    def normalize_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Normalize text by removing special characters and extra whitespace.
        
        Args:
            text: Input text to normalize.
            language: Language of the text ('fr' or 'ar'). If None, will be detected.
            
        Returns:
            Normalized text.
        """
        if not text:
            return ""
            
        if not language:
            language = self.detect_language(text)
            
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters while preserving language-specific characters
        if language == 'ar':
            # Preserve Arabic unicode range, numbers, and basic punctuation
            text = re.sub(r'[^\u0600-\u06FF\u0660-\u0669\s\d\.\,\!\?\:\;]', ' ', text)
        else:
            # For non-Arabic, remove special chars but keep alphanumeric, accented chars, and basic punctuation
            text = re.sub(r'[^\w\s\u00C0-\u017F\.\,\!\?\:\;]', ' ', text)
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase for non-Arabic text
        if language != 'ar':
            text = text.lower()
        
        return text
    
    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize text into words using a simple space-based approach.
        
        Args:
            text: Input text to tokenize.
            language: Language of the text ('fr' or 'ar'). If None, will be detected.
            
        Returns:
            List of tokens.
        """
        if not text:
            return []
            
        if not language:
            language = self.detect_language(text)
            
        # Simple tokenization by whitespace and punctuation
        # Split on whitespace and common punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter out very short tokens (less than 2 characters) except for meaningful short words
        meaningful_short = {'à', 'au', 'du', 'de', 'le', 'la', 'un', 'et', 'ou', 'si', 'en', 'on'}
        if language == 'ar':
            meaningful_short.update({'في', 'من', 'إلى', 'على', 'عن', 'هو', 'هي', 'لا', 'ما'})
        
        tokens = [token for token in tokens if len(token) >= 2 or token.lower() in meaningful_short]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str], language: Optional[str] = None) -> List[str]:
        """
        Remove stopwords from a list of tokens.
        
        Args:
            tokens: List of tokens.
            language: Language of the tokens ('fr' or 'ar'). If None, will be detected.
            
        Returns:
            List of tokens with stopwords removed.
        """
        if not tokens:
            return []
            
        if not language:
            # Try to detect language from the joined tokens
            language = self.detect_language(' '.join(tokens))
            
        if language == 'fr':
            return [token for token in tokens if token.lower() not in self.fr_stopwords]
        elif language == 'ar':
            return [token for token in tokens if token not in self.ar_stopwords]
        else:
            # For other languages, don't remove stopwords
            return tokens
    
    def preprocess(self, text: str, language: Optional[str] = None, remove_stops: bool = True) -> List[str]:
        """
        Full preprocessing pipeline: normalize, tokenize, and optionally remove stopwords.
        
        Args:
            text: Input text to preprocess.
            language: Language of the text ('fr' or 'ar'). If None, will be detected.
            remove_stops: Whether to remove stopwords.
            
        Returns:
            List of preprocessed tokens.
        """
        if not text:
            return []
            
        if not language:
            language = self.detect_language(text)
            
        try:
            normalized_text = self.normalize_text(text, language)
            tokens = self.tokenize(normalized_text, language)
            
            if remove_stops:
                filtered_tokens = self.remove_stopwords(tokens, language)
                return filtered_tokens
            else:
                return tokens
                
        except Exception as e:
            logging.error(f"Error in text preprocessing: {e}")
            # Return basic tokenization as fallback
            return text.split()
    
    def segment_questions(self, text: str) -> List[str]:
        """
        Segment text into multiple questions if present.
        
        Args:
            text: Input text that may contain multiple questions.
            
        Returns:
            List of individual questions.
        """
        if not text:
            return []
            
        # Split on question marks (both Latin and Arabic) and newlines
        segments = re.split(r'[?؟\n]+', text)
        
        # Clean up segments and add back question marks where appropriate
        questions = []
        for segment in segments:
            segment = segment.strip()
            if segment:
                # Add question mark if the segment looks like a question
                if any(word in segment.lower() for word in ['quel', 'comment', 'pourquoi', 'quand', 'où', 'combien']):
                    if not segment.endswith('?') and not segment.endswith('؟'):
                        segment += ' ?'
                elif any(word in segment for word in ['ماذا', 'كيف', 'لماذا', 'متى', 'أين', 'كم']):
                    if not segment.endswith('?') and not segment.endswith('؟'):
                        segment += ' ؟'
                        
                questions.append(segment)
        
        # If no questions were detected, return the original text as a single item
        if not questions:
            return [text]
            
        return questions
    
    def extract_keywords(self, text: str, language: Optional[str] = None, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text.
            language: Language of the text.
            max_keywords: Maximum number of keywords to return.
            
        Returns:
            List of keywords.
        """
        if not text:
            return []
            
        # Preprocess text to get clean tokens
        tokens = self.preprocess(text, language, remove_stops=True)
        
        # Filter out very short tokens
        keywords = [token for token in tokens if len(token) >= 3]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
                
        return unique_keywords[:max_keywords]