"""
Utilities for formatting chatbot responses.
"""

import re
import logging
from typing import List, Dict, Any, Optional

class ResponseFormatter:
    """
    Class for formatting chatbot responses based on language and query type.
    """
    
    @staticmethod
    def format_response(
        response: str, 
        question: str, 
        documents: List[Dict[str, Any]], 
        language: str = 'fr'
    ) -> Dict[str, Any]:
        """
        Format the final response for the API output.
        
        Args:
            response: Generated response from the LLM.
            question: Original user question.
            documents: Retrieved documents used for context.
            language: Language code ('fr' or 'ar').
            
        Returns:
            Dictionary with formatted response data.
        """
        try:
            # Extract any referenced codes from the response
            referenced_codes = ResponseFormatter._extract_rne_codes(response)
            
            # Format document references
            references = []
            seen_codes = set()
            
            for doc in documents:
                document = doc.get('document', {})
                code = document.get('code', '')
                
                # Only include if code is referenced in response or if no codes were referenced
                if not referenced_codes or code in referenced_codes:
                    if code and code not in seen_codes:
                        references.append({
                            'code': code,
                            'procedure': document.get('procedure', ''),
                            'type_entreprise': document.get('type_entreprise', ''),
                            'score': doc.get('score', 0.0),
                            'pdf_link': document.get('pdf_link', ''),
                            'language': document.get('language', language)
                        })
                        seen_codes.add(code)
            
            # Determine text direction based on language
            text_direction = ResponseFormatter._get_text_direction(language)
            
            return {
                'response': response.strip() if response else '',
                'language': language,
                'text_direction': text_direction,
                'references': references,
                'query': question.strip() if question else '',
                'document_count': len(documents),
                'referenced_codes': referenced_codes
            }
            
        except Exception as e:
            logging.error(f"Error formatting response: {e}")
            # Return basic format on error
            return {
                'response': response or '',
                'language': language,
                'text_direction': 'ltr',
                'references': [],
                'query': question or '',
                'document_count': 0,
                'referenced_codes': []
            }
    
    @staticmethod
    def format_multi_response(
        responses: List[Dict[str, Any]],
        original_query: str,
        language: str = 'fr'
    ) -> Dict[str, Any]:
        """
        Format responses for multiple questions in a single query.
        
        Args:
            responses: List of response dictionaries for each question.
            original_query: Original user query containing multiple questions.
            language: Language code ('fr' or 'ar').
            
        Returns:
            Dictionary with formatted combined response data.
        """
        try:
            if not responses:
                return ResponseFormatter.format_response('', original_query, [], language)
            
            # Combine all responses
            combined_response = ""
            all_references = []
            all_codes = set()
            total_docs = 0
            
            for i, resp in enumerate(responses):
                question = resp.get('query', '')
                answer = resp.get('response', '')
                
                # Add question and answer with proper formatting
                if language == 'ar':
                    combined_response += f"**السؤال {i+1}:** {question}\n\n"
                    combined_response += f"**الإجابة {i+1}:** {answer}\n\n"
                    if i < len(responses) - 1:  # Don't add separator after last response
                        combined_response += "---\n\n"
                else:
                    combined_response += f"**Question {i+1}:** {question}\n\n"
                    combined_response += f"**Réponse {i+1}:** {answer}\n\n"
                    if i < len(responses) - 1:
                        combined_response += "---\n\n"
                    
                # Collect references
                resp_refs = resp.get('references', [])
                for ref in resp_refs:
                    code = ref.get('code', '')
                    if code and code not in all_codes:
                        all_references.append(ref)
                        all_codes.add(code)
                
                # Collect referenced codes
                resp_codes = resp.get('referenced_codes', [])
                all_codes.update(resp_codes)
                
                total_docs += resp.get('document_count', 0)
            
            # Sort references by score (highest first)
            all_references.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Determine text direction based on language
            text_direction = ResponseFormatter._get_text_direction(language)
            
            return {
                'response': combined_response.strip(),
                'language': language,
                'text_direction': text_direction,
                'references': all_references,
                'query': original_query.strip() if original_query else '',
                'question_count': len(responses),
                'document_count': total_docs,
                'referenced_codes': list(all_codes)
            }
            
        except Exception as e:
            logging.error(f"Error formatting multi-response: {e}")
            # Return basic format on error
            return ResponseFormatter.format_response('', original_query, [], language)
    
    @staticmethod
    def format_clarification_response(
        main_response: str,
        follow_up_question: str,
        options: List[str],
        language: str = 'fr'
    ) -> Dict[str, Any]:
        """
        Format a clarification response when the query is too vague.
        
        Args:
            main_response: Main response explaining why clarification is needed.
            follow_up_question: Question asking for clarification.
            options: List of options for the user to choose from.
            language: Language code ('fr' or 'ar').
            
        Returns:
            Dictionary with formatted clarification response.
        """
        try:
            text_direction = ResponseFormatter._get_text_direction(language)
            
            return {
                'type': 'clarification_needed',
                'response': main_response.strip() if main_response else '',
                'follow_up_question': follow_up_question.strip() if follow_up_question else '',
                'options': [opt.strip() for opt in options if opt and opt.strip()],
                'language': language,
                'text_direction': text_direction,
                'awaiting_clarification': True
            }
            
        except Exception as e:
            logging.error(f"Error formatting clarification response: {e}")
            return {
                'type': 'clarification_needed',
                'response': main_response or '',
                'follow_up_question': follow_up_question or '',
                'options': options or [],
                'language': language,
                'text_direction': 'ltr',
                'awaiting_clarification': True
            }
    
    @staticmethod
    def format_error_response(
        error_message: str,
        language: str = 'fr'
    ) -> Dict[str, Any]:
        """
        Format an error response.
        
        Args:
            error_message: Error message to display.
            language: Language code ('fr' or 'ar').
            
        Returns:
            Dictionary with formatted error response.
        """
        try:
            text_direction = ResponseFormatter._get_text_direction(language)
            
            # Default error messages by language
            default_messages = {
                'fr': "Désolé, une erreur s'est produite. Veuillez réessayer.",
                'ar': "آسف، حدث خطأ. يرجى المحاولة مرة أخرى."
            }
            
            if not error_message or not error_message.strip():
                error_message = default_messages.get(language, default_messages['fr'])
            
            return {
                'type': 'error',
                'response': error_message.strip(),
                'language': language,
                'text_direction': text_direction,
                'error': True
            }
            
        except Exception as e:
            logging.error(f"Error formatting error response: {e}")
            return {
                'type': 'error',
                'response': error_message or 'An error occurred',
                'language': language,
                'text_direction': 'ltr',
                'error': True
            }
    
    @staticmethod
    def _extract_rne_codes(text: str) -> List[str]:
        """
        Extract RNE codes from a text.
        
        Args:
            text: Text to extract codes from.
            
        Returns:
            List of extracted RNE codes.
        """
        if not text or not isinstance(text, str):
            return []
        
        try:
            # Pattern to match RNE codes (e.g., RNE M 004.37, RNE-M-004.37, etc.)
            patterns = [
                r'RNE\s+[A-Z]\s+\d+\.\d+',  # RNE M 004.37
                r'RNE-[A-Z]-\d+\.\d+',      # RNE-M-004.37
                r'RNE[A-Z]\d+\.\d+',        # RNEM004.37
                r'[A-Z]\s+\d+\.\d+',        # M 004.37 (when RNE is implied)
            ]
            
            codes = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                codes.extend(matches)
            
            # Normalize codes and remove duplicates
            normalized_codes = []
            seen = set()
            
            for code in codes:
                # Normalize the format to "RNE X XXX.XX"
                normalized = re.sub(r'[^A-Z0-9\.]', ' ', code.upper())
                normalized = re.sub(r'\s+', ' ', normalized).strip()
                
                if not normalized.startswith('RNE'):
                    normalized = f"RNE {normalized}"
                
                if normalized not in seen:
                    normalized_codes.append(normalized)
                    seen.add(normalized)
            
            return normalized_codes
            
        except Exception as e:
            logging.error(f"Error extracting RNE codes: {e}")
            return []
    
    @staticmethod
    def _get_text_direction(language: str) -> str:
        """
        Get text direction for the given language.
        
        Args:
            language: Language code.
            
        Returns:
            'rtl' for right-to-left languages, 'ltr' otherwise.
        """
        rtl_languages = ['ar', 'he', 'fa', 'ur']
        return 'rtl' if language in rtl_languages else 'ltr'
    
    @staticmethod
    def truncate_response(text: str, max_length: int = 1000) -> str:
        """
        Truncate response text if it's too long.
        
        Args:
            text: Text to truncate.
            max_length: Maximum length allowed.
            
        Returns:
            Truncated text with ellipsis if needed.
        """
        if not text or len(text) <= max_length:
            return text
        
        # Try to cut at a sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclamation = truncated.rfind('!')
        
        # Find the last sentence ending
        last_sentence_end = max(last_period, last_question, last_exclamation)
        
        if last_sentence_end > max_length * 0.8:  # If we can preserve at least 80% of content
            return text[:last_sentence_end + 1]
        else:
            return text[:max_length].rstrip() + '...'
    
    @staticmethod
    def clean_response(text: str) -> str:
        """
        Clean and normalize response text.
        
        Args:
            text: Text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove multiple consecutive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        
        return text.strip()