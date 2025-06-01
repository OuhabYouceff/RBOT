"""
Enhanced OpenAI integration with vague question detection for RNE chatbot.
"""

import openai
import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging

from app.utils.config import OPENAI_API_KEY, LLM_MODEL, SYSTEM_PROMPT, MAX_CONTEXT_LENGTH

class ResponseType(Enum):
    DIRECT_ANSWER = "direct_answer"
    CLARIFICATION_NEEDED = "clarification_needed"
    NO_RESULTS = "no_results"

class FollowUpResponse:
    """Structure for responses that need clarification."""
    
    def __init__(self, main_response: str, follow_up_question: str, options: List[str]):
        self.main_response = main_response
        self.follow_up_question = follow_up_question
        self.options = options
        self.response_type = ResponseType.CLARIFICATION_NEEDED

class DirectResponse:
    """Structure for direct answers."""
    
    def __init__(self, response: str):
        self.response = response
        self.response_type = ResponseType.DIRECT_ANSWER

class OpenAIClient:
    """
    Enhanced client for interacting with OpenAI's LLM API with vague question detection.
    """
    
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = LLM_MODEL):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key.
            model: Model identifier to use (e.g., 'gpt-4', 'gpt-3.5-turbo').
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = SYSTEM_PROMPT
        
        # Define common vague question patterns and their clarifications
        self.clarification_patterns = {
            'capital': {
                'keywords': ['capital', 'minimum', 'montant', 'combien'],
                'main_response': "Le capital minimum dépend du type de société que vous souhaitez créer. Pour vous donner une réponse précise, j'ai besoin de connaître le type de société.",
                'follow_up': "Quel type de société souhaitez-vous créer ?",
                'options': [
                    "SARL (Société à Responsabilité Limitée)",
                    "SA (Société Anonyme)", 
                    "EURL (Entreprise Unipersonnelle à Responsabilité Limitée)",
                    "SUARL (Société Unipersonnelle à Responsabilité Limitée)"
                ]
            },
            'delais': {
                'keywords': ['délai', 'delais', 'temps', 'durée', 'combien de temps'],
                'main_response': "Les délais de création varient selon le type d'entreprise et les procédures choisies. Pour vous donner une estimation précise, j'ai besoin de plus d'informations.",
                'follow_up': "Quel type d'entreprise souhaitez-vous créer ?",
                'options': [
                    "Personne physique (commerçant individuel)",
                    "SARL (Société à Responsabilité Limitée)",
                    "SA (Société Anonyme)",
                    "Autre type de société"
                ]
            },
            'documents': {
                'keywords': ['document', 'pièce', 'papier', 'dossier', 'fournir'],
                'main_response': "Les documents requis dépendent du type d'entreprise et de la procédure de création choisie. Pour vous fournir la liste exacte, j'ai besoin de connaître votre situation.",
                'follow_up': "Quel type d'entreprise envisagez-vous de créer ?",
                'options': [
                    "Personne physique (commerçant individuel)",
                    "SARL (Société à Responsabilité Limitée)", 
                    "SA (Société Anonyme)",
                    "Association",
                    "Autre"
                ]
            },
            'cout': {
                'keywords': ['coût', 'cout', 'prix', 'frais', 'redevance', 'tarif'],
                'main_response': "Les coûts de création d'entreprise varient selon le type d'entreprise et les services choisis. Pour vous donner un devis précis, j'ai besoin de connaître vos besoins.",
                'follow_up': "Quel type d'entreprise souhaitez-vous créer ?",
                'options': [
                    "Personne physique (commerçant individuel)",
                    "SARL (Société à Responsabilité Limitée)",
                    "SA (Société Anonyme)",
                    "Autre type de société"
                ]
            },
            'creation': {
                'keywords': ['créer', 'creation', 'comment', 'étapes', 'procédure'],
                'main_response': "La procédure de création varie selon le type d'entreprise que vous souhaitez établir. Pour vous guider efficacement, j'ai besoin de connaître votre projet.",
                'follow_up': "Quel type d'entreprise souhaitez-vous créer ?",
                'options': [
                    "Personne physique (commerçant individuel)",
                    "SARL (Société à Responsabilité Limitée)",
                    "SA (Société Anonyme)",
                    "Association ou ONG",
                    "Autre"
                ]
            }
        }
    
    def analyze_question_specificity(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Analyze if a question is too vague and needs clarification.
        
        Args:
            query: User query to analyze.
            
        Returns:
            Tuple of (is_vague, clarification_category)
        """
        try:
            if not query or not isinstance(query, str):
                return False, None
                
            query_lower = query.lower()
            
            # Check for vague patterns
            for category, pattern in self.clarification_patterns.items():
                # Check if query contains keywords from this category
                if any(keyword in query_lower for keyword in pattern['keywords']):
                    # Check if the query lacks specificity (no company type mentioned)
                    company_types = ['sarl', 'sa', 'eurl', 'suarl', 'personne physique', 'association', 'ong']
                    has_specific_type = any(comp_type in query_lower for comp_type in company_types)
                    
                    if not has_specific_type:
                        logging.info(f"Detected vague question in category: {category}")
                        return True, category
            
            return False, None
            
        except Exception as e:
            logging.error(f"Error in analyze_question_specificity: {str(e)}")
            return False, None
    
    def generate_clarification_response(self, category: str, language: str = 'fr') -> FollowUpResponse:
        """
        Generate a clarification response for vague questions.
        
        Args:
            category: Category of vague question.
            language: Language to respond in.
            
        Returns:
            FollowUpResponse object with clarification details.
        """
        if category not in self.clarification_patterns:
            # Fallback generic clarification
            if language == 'fr':
                return FollowUpResponse(
                    main_response="Votre question nécessite plus de précisions pour que je puisse vous donner une réponse adaptée.",
                    follow_up_question="Pouvez-vous préciser le type d'entreprise qui vous intéresse ?",
                    options=[
                        "Personne physique (commerçant individuel)",
                        "SARL (Société à Responsabilité Limitée)",
                        "SA (Société Anonyme)",
                        "Autre"
                    ]
                )
            else:  # Arabic
                return FollowUpResponse(
                    main_response="سؤالك يحتاج لمزيد من التوضيح باش نقدر نعطيك إجابة مناسبة.",
                    follow_up_question="تنجم توضح نوع الشركة اللي تحب تعملها؟",
                    options=[
                        "شخص طبيعي (تاجر فردي)",
                        "شركة ذات مسؤولية محدودة (SARL)",
                        "شركة مساهمة (SA)",
                        "نوع آخر"
                    ]
                )
        
        pattern = self.clarification_patterns[category]
        
        # Handle Arabic language
        if language == 'ar':
            # Arabic translations for common patterns
            arabic_patterns = {
                'capital': {
                    'main_response': "رأس المال الأدنى يتوقف على نوع الشركة اللي تحب تعملها. باش نعطيك إجابة دقيقة، نحتاج نعرف نوع الشركة.",
                    'follow_up': "أشنوا نوع الشركة اللي تحب تعملها؟",
                    'options': [
                        "شركة ذات مسؤولية محدودة (SARL)",
                        "شركة مساهمة (SA)",
                        "مؤسسة فردية ذات مسؤولية محدودة (EURL)",
                        "شركة فردية ذات مسؤولية محدودة (SUARL)"
                    ]
                },
                'delais': {
                    'main_response': "المدة متاع التأسيس تتوقف على نوع الشركة والإجراءات اللي باش تختارها. باش نعطيك تقدير دقيق، نحتاج معلومات زيادة.",
                    'follow_up': "أشنوا نوع الشركة اللي تحب تعملها؟",
                    'options': [
                        "شخص طبيعي (تاجر فردي)",
                        "شركة ذات مسؤولية محدودة (SARL)",
                        "شركة مساهمة (SA)",
                        "نوع آخر من الشركات"
                    ]
                }
                # Add more Arabic translations as needed
            }
            
            if category in arabic_patterns:
                arabic_pattern = arabic_patterns[category]
                return FollowUpResponse(
                    main_response=arabic_pattern['main_response'],
                    follow_up_question=arabic_pattern['follow_up'],
                    options=arabic_pattern['options']
                )
        
        return FollowUpResponse(
            main_response=pattern['main_response'],
            follow_up_question=pattern['follow_up'],
            options=pattern['options']
        )
    
    def generate_response(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        language: str = 'fr',
        system_prompt: Optional[str] = None,
        force_direct: bool = False
    ) -> Any:  # Returns either DirectResponse or FollowUpResponse
        """
        Generate a response, detecting if clarification is needed first.
        
        Args:
            query: User query.
            context: List of retrieved documents for context.
            language: Language to respond in ('fr' or 'ar').
            system_prompt: Optional custom system prompt.
            force_direct: If True, skip vague question detection.
            
        Returns:
            Either DirectResponse or FollowUpResponse based on query specificity.
        """
        try:
            # Check if question is vague and needs clarification
            if not force_direct:
                is_vague, category = self.analyze_question_specificity(query)
                if is_vague and category:
                    return self.generate_clarification_response(category, language)
            
            # Generate direct response using existing logic
            # Use custom system prompt if provided, otherwise use default
            if system_prompt:
                prompt = system_prompt
            else:
                prompt = self._get_system_prompt(language)
                
            # Format context for the prompt
            formatted_context = self._format_context(context)
            
            # Prepare messages for the chat completion
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Contexte:\n{formatted_context}\n\nQuestion: {query}"}
            ]
            
            # Call the OpenAI API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                top_p=0.9,
                stream=False
            )
            
            # Extract and return the response
            response = completion.choices[0].message.content
            return DirectResponse(response)
            
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {str(e)}")
            error_msg = f"Désolé, je n'ai pas pu générer une réponse. Erreur: {str(e)}"
            if language == 'ar':
                error_msg = f"آسف، ما نجمتش نولد إجابة. خطأ: {str(e)}"
            return DirectResponse(error_msg)
    
    def handle_follow_up_response(
        self, 
        original_query: str, 
        selected_option: str, 
        context: List[Dict[str, Any]], 
        language: str = 'fr'
    ) -> DirectResponse:
        """
        Handle follow-up response after user selects an option.
        
        Args:
            original_query: The original vague question.
            selected_option: The option selected by the user.
            context: Retrieved documents for context.
            language: Language to respond in.
            
        Returns:
            DirectResponse with specific answer.
        """
        # Combine original query with selected option for more specific query
        specific_query = f"{original_query} - {selected_option}"
        
        # Generate response with force_direct=True to skip vague detection
        response = self.generate_response(
            specific_query, 
            context, 
            language, 
            force_direct=True
        )
        
        return response
    
    def segment_questions(self, query: str) -> List[str]:
        """
        Use the LLM to segment a query into multiple questions if needed.
        
        Args:
            query: User query that may contain multiple questions.
            
        Returns:
            List of individual questions.
        """
        prompt = """
Divise uniquement les phrases contenant plusieurs questions en questions distinctes.
Ne transforme pas des sujets ou titres en questions. Ne fais pas de brainstorming.
Retourne une liste de questions, une par ligne. Si le texte contient une seule question, retourne-la telle quelle.

Exemples :

Texte : "Quel est le délai de création d'une SARL et quelles sont les pièces à fournir ?"
Sortie :
Quel est le délai de création d'une SARL ?
Quelles sont les pièces à fournir ?

Texte : "Quels sont les frais pour créer une entreprise individuelle ?"
Sortie :
Quels sont les frais pour créer une entreprise individuelle ?

Texte : "Création du SARL et checklist"
Sortie :
Création du SARL et checklist

Texte : "Quel est le Délais de la Création Personne physique commerçant et quel sont les Redevances à acquitter pour tout type d'entreprise ?"
Sortie :
Quel est le Délais de la Création Personne physique commerçant ?
Quel sont les Redevances à acquitter pour tout type d'entreprise ?

Texte : """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                stream=False
            )
            
            response = completion.choices[0].message.content
            
            # Split the response into lines and filter out empty lines
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            
            # If no questions were detected, return the original query
            if not questions:
                return [query]
                
            return questions
            
        except Exception as e:
            logging.error(f"Error in question segmentation: {str(e)}")
            # On error, just return the original query
            return [query]
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a string for the LLM context.
        
        Args:
            context: List of retrieved documents.
            
        Returns:
            Formatted context string.
        """
        if not context:
            return "Aucun contexte pertinent trouvé."
            
        formatted_text = ""
        
        for i, item in enumerate(context):
            doc = item['document']
            score = item['score']
            
            formatted_text += f"--- Document {i+1} (Pertinence: {score:.2f}) ---\n"
            formatted_text += f"Code: {doc['code']}\n"
            formatted_text += f"Type d'entreprise: {doc['type_entreprise']}\n"
            formatted_text += f"Genre d'entreprise: {doc['genre_entreprise']}\n"
            formatted_text += f"Procédure: {doc['procedure']}\n"
            formatted_text += f"Redevance demandée: {doc['redevance_demandee']}\n"
            formatted_text += f"Délais: {doc['delais']}\n"
            
            # Add the raw content
            raw_content = doc.get('raw_content', {})
            if raw_content:
                formatted_text += "Contenu détaillé:\n"
                
                for key, value in raw_content.items():
                    if isinstance(value, list):
                        formatted_text += f"{key}: {', '.join(value)}\n"
                    else:
                        formatted_text += f"{key}: {value}\n"
                        
            formatted_text += f"Lien PDF: {doc.get('pdf_link', 'Non disponible')}\n\n"
            
        # Ensure we don't exceed the context limit
        if len(formatted_text) > MAX_CONTEXT_LENGTH:
            formatted_text = formatted_text[:MAX_CONTEXT_LENGTH] + "...[Contexte tronqué]"
            
        return formatted_text
    
    def _get_system_prompt(self, language: str) -> str:
        """
        Get the appropriate system prompt based on the language.
        
        Args:
            language: Language code ('fr' or 'ar').
            
        Returns:
            System prompt in the specified language.
        """
        if language == 'ar':
            return """إنت معاون قانوني مختص في قوانين السجل الوطني للمؤسسات (RNE) في تونس.
    خدمتك إنك تجاوب على الأسئلة وتعطي معلومات صحيحة، مبنية على الوثائق الرسمية متاع السجل.
    لازمك ديما تحكي مع الناس باللهجة التونسية، بالدارجة، باش تكون أقرب ليهم وأسهل للفهم.
    حافظ على نبرة محترفة، وما تعطي كان المعلومة اللي مأكدة وموثقة.
    وإذا المعلومة ما هيش موجودة، ولا ماكش متأكد، قول هذا بكل وضوح."""

        else:
            return self.system_prompt