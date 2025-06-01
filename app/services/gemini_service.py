import google.generativeai as genai
from app.core.config import settings
from app.models.chat import FollowUpQuestion, SegmentedQuestions
from typing import Optional, List
import json
import re

class GeminiService:
    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key not found in environment variables")
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        arabic_chars = any('\u0600' <= char <= '\u06FF' for char in text)
        french_words = ['est', 'le', 'la', 'les', 'et', 'de', 'du', 'comment', 'quel', 'quelle']
        english_words = ['the', 'and', 'how', 'what', 'where', 'when', 'company', 'business']
        
        if arabic_chars:
            return "arabic"
        elif any(word.lower() in text.lower() for word in french_words):
            return "french"
        elif any(word.lower() in text.lower() for word in english_words):
            return "english"
        else:
            return "french"
    
    def _extract_json_from_response(self, text: str) -> dict:
        """Extract JSON from Gemini response"""
        try:
            text = text.strip()
            return json.loads(text)
        except:
            try:
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                
                return self._manual_json_extraction(text)
            except Exception as e:
                print(f"JSON parsing failed: {e}")
                return self._manual_json_extraction(text)
    
    def _manual_json_extraction(self, text: str) -> dict:
        """Manually extract key fields when JSON parsing fails"""
        result = {}
        
        answer_match = re.search(r'"answer":\s*"([^"]*(?:\\.[^"]*)*)"', text)
        suggestions_match = re.search(r'"suggestions":\s*\[(.*?)\]', text, re.DOTALL)
        suggest_forms_match = re.search(r'"suggest_forms":\s*(true|false)', text)
        
        if answer_match:
            answer = answer_match.group(1)
            answer = answer.replace('\\"', '"').replace('\\n', '\n')
            result["answer"] = answer
        else:
            lines = text.split('\n')
            answer_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith(('{', '}'))]
            result["answer"] = ' '.join(answer_lines[:3])
        
        if suggestions_match:
            suggestions_text = suggestions_match.group(1)
            suggestions = re.findall(r'"([^"]*)"', suggestions_text)
            result["suggestions"] = suggestions
        else:
            result["suggestions"] = []
        
        result["suggest_forms"] = suggest_forms_match.group(1) == "true" if suggest_forms_match else False
        result["form_context"] = ""
        
        return result
    
    async def check_needs_additional_info(self, query: str) -> Optional[FollowUpQuestion]:
        """Check if query needs additional information - WITH EXAMPLES"""
        language = self._detect_language(query)
        
        prompt = f"""
        You are an expert assistant for RNE Tunisia. Analyze if this user query needs additional information.

        EXAMPLES OF QUERIES THAT NEED FOLLOW-UP:

        Example 1:
        User: "Quel est le capital minimum ?"
        Response: {{
            "needs_info": true,
            "main_response": "Le capital minimum dépend du type de société que vous souhaitez créer.",
            "follow_up_question": "Quel type de société souhaitez-vous créer ?",
            "options": ["SARL", "SA", "EURL", "SUARL", "Autre"]
        }}

        Example 2:
        User: "What are the required documents?"
        Response: {{
            "needs_info": true,
            "main_response": "The required documents depend on the type of registration you want to do.",
            "follow_up_question": "What type of registration do you need?",
            "options": ["Company registration", "Individual registration", "Association registration", "Modification"]
        }}

        Example 3:
        User: "Quels sont les frais ?"
        Response: {{
            "needs_info": true,
            "main_response": "Les frais varient selon le type de service RNE demandé.",
            "follow_up_question": "Quel service RNE vous intéresse ?",
            "options": ["Inscription", "Modification", "Extrait", "Traduction"]
        }}

        EXAMPLES OF QUERIES THAT DON'T NEED FOLLOW-UP:

        Example 1:
        User: "Comment créer une SARL ?"
        Response: {{"needs_info": false}}

        Example 2:
        User: "Documents nécessaires inscription RNE"
        Response: {{"needs_info": false}}

        Example 3:
        User: "capital minimum SARL"
        Response: {{"needs_info": false}}

        Now analyze this query: "{query}"

        If additional info needed, respond ONLY with:
        {{
            "needs_info": true,
            "main_response": "Brief explanation in {language}",
            "follow_up_question": "Specific question in {language}",
            "options": ["option1", "option2", "option3"]
        }}

        If NO additional info needed, respond ONLY with:
        {{
            "needs_info": false
        }}

        Return ONLY JSON, no other text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            print(f"Gemini follow-up check response: {response.text}")
            result = self._extract_json_from_response(response.text)
            
            if result.get("needs_info", False):
                return FollowUpQuestion(
                    question=result.get("follow_up_question", "Could you please provide more details?"),
                    options=result.get("options", ["Option 1", "Option 2"])
                ), result.get("main_response", "I need more information to provide an accurate answer.")
            return None, None
            
        except Exception as e:
            print(f"Error in check_needs_additional_info: {e}")
            return None, None
    
    async def segment_query(self, query: str) -> SegmentedQuestions:
        """Segment query into logical questions"""
        language = self._detect_language(query)
        
        prompt = f"""
        Analyze this query and determine if it contains multiple distinct questions.

        EXAMPLES:

        Example 1:
        User: "How to register a company and what documents do I need?"
        Response: {{
            "multiple_questions": true,
            "questions": ["How to register a company in Tunisia?", "What documents are needed for company registration?"]
        }}

        Example 2:
        User: "Quel est le capital minimum et les frais d'inscription ?"
        Response: {{
            "multiple_questions": true,
            "questions": ["Quel est le capital minimum pour créer une société?", "Quels sont les frais d'inscription au RNE?"]
        }}

        Example 3:
        User: "Comment créer une SARL ?"
        Response: {{
            "multiple_questions": false,
            "questions": ["Comment créer une SARL ?"]
        }}

        User query: "{query}"

        Response format:
        {{
            "multiple_questions": true/false,
            "questions": ["question1", "question2", ...]
        }}

        Return ONLY JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = self._extract_json_from_response(response.text)
            
            questions = result.get("questions", [query])
            return SegmentedQuestions(
                questions=questions,
                original_query=query
            )
            
        except Exception as e:
            print(f"Error in segment_query: {e}")
            return SegmentedQuestions(questions=[query], original_query=query)
    
    async def format_final_answer(self, query: str, rag_results: List, language: str = "auto") -> dict:
        """Format final answer - ULTRA CONCISE"""
        if language == "auto":
            language = self._detect_language(query)
        
        answers_by_question = []
        for r in rag_results:
            answers_by_question.append(f"Q: {r['question']}\nA: {r['answer']}")
        
        structured_info = "\n\n".join(answers_by_question)
        
        prompt = f"""
        You are an expert for RNE Tunisia. Give a SHORT, direct answer.

        User query: "{query}"
        Information: {structured_info}

        EXAMPLES OF GOOD SHORT ANSWERS:

        Example 1:
        Query: "capital minimum SARL"
        Response: {{
            "answer": "Le capital minimum pour une SARL en Tunisie est de 1 000 TND. Il doit être déposé en banque avant l'immatriculation.",
            "suggestions": ["Où déposer le capital ?", "Documents pour SARL"],
            "suggest_forms": true
        }}

        Example 2:
        Query: "Comment créer une SARL ?"
        Response: {{
            "answer": "Pour créer une SARL: 1) Rédiger statuts 2) Déposer capital (1000 TND) 3) S'immatriculer au RNE 4) Publier au JORT.",
            "suggestions": ["Documents nécessaires", "Frais d'inscription"],
            "suggest_forms": true
        }}

        RULES:
        1. Maximum 2-3 sentences
        2. Include TND amounts when relevant
        3. Be direct and factual
        4. Suggest forms when about registration/modification

        Respond ONLY with:
        {{
            "answer": "Short direct answer in {language}",
            "suggestions": ["suggestion1", "suggestion2"] or [],
            "suggest_forms": true/false
        }}

        Return ONLY JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = self._extract_json_from_response(response.text)
            
            return {
                "answer": result.get("answer", "Une erreur s'est produite."),
                "suggestions": result.get("suggestions", []),
                "suggest_forms": result.get("suggest_forms", False),
                "form_context": ""
            }
        except Exception as e:
            print(f"Error in format_final_answer: {e}")
            return {
                "answer": "Une erreur s'est produite.",
                "suggestions": [],
                "suggest_forms": False,
                "form_context": ""
            }

gemini_service = GeminiService()