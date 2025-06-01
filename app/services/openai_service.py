from openai import OpenAI
from app.core.config import settings

class OpenAIService:
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not found")
        try:
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=30.0
            )
        except Exception as e:
            print(f"OpenAI client initialization error: {e}")
            self.client = None
    
    async def web_search_answer(self, question: str) -> str:
        """Use OpenAI to search and answer questions about Tunisia business law"""
        if not self.client:
            return self._fallback_tunisia_answer(question)
            
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert on Tunisia business law and RNE (Registre National des Entreprises).

                        EXPERTISE AREAS:
                        - Tunisia company registration procedures
                        - RNE (Registre National des Entreprises) specific processes
                        - Tunisia company types: SARL, SA, EURL, SUARL, Entreprise Individuelle
                        - Capital requirements in Tunisian Dinars (TND)
                        - Legal requirements and documentation for Tunisia
                        - INNORPI (Institut National de la Normalisation et de la Propriété Industrielle)
                        - Tunisia tax obligations and business licenses
                        - CNSS (Caisse Nationale de Sécurité Sociale) affiliations
                        - Journal Officiel de la République Tunisienne (JORT) publications

                        RESPONSE RULES:
                        - Keep responses SHORT (2-3 sentences maximum)
                        - Include specific TND amounts when relevant
                        - Reference Tunisia-specific institutions (RNE, INNORPI, CNSS, etc.)
                        - Be direct and factual
                        - Respond in the same language as the question
                        - No long explanations unless specifically requested"""
                    },
                    {"role": "user", "content": f"Question about Tunisia business/RNE: {question}"}
                ],
                max_tokens=200,  # Keep responses short
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}")
            return self._fallback_tunisia_answer(question)
    
    def _fallback_tunisia_answer(self, question: str) -> str:
        """Fallback Tunisia-specific answers when OpenAI is unavailable"""
        question_lower = question.lower()
        
        # Detect language
        if any(word in question_lower for word in ['comment', 'quel', 'création', 'société']):
            language = "french"
        else:
            language = "english"
        
        # Capital questions
        if any(word in question_lower for word in ['capital', 'minimum']):
            if 'sarl' in question_lower:
                if language == "french":
                    return "Le capital minimum pour une SARL en Tunisie est de 1 000 TND. Il doit être déposé en banque avant l'immatriculation au RNE."
                else:
                    return "The minimum capital for a SARL in Tunisia is 1,000 TND. It must be deposited in a bank before RNE registration."
            elif 'sa' in question_lower and 'sarl' not in question_lower:
                if language == "french":
                    return "Le capital minimum pour une SA en Tunisie est de 5 000 TND. Au moins 25% doit être libéré lors de la constitution."
                else:
                    return "The minimum capital for a SA in Tunisia is 5,000 TND. At least 25% must be released upon incorporation."
            else:
                if language == "french":
                    return "Le capital minimum varie selon le type de société : SARL 1000 TND, SA 5000 TND, EURL 1000 TND, SUARL 1000 TND."
                else:
                    return "Minimum capital varies by company type: SARL 1000 TND, SA 5000 TND, EURL 1000 TND, SUARL 1000 TND."
        
        # Creation/registration questions
        elif any(word in question_lower for word in ['créer', 'création', 'create', 'register', 'immatriculer']):
            if 'sarl' in question_lower:
                if language == "french":
                    return "Pour créer une SARL : 1) Rédiger statuts 2) Déposer capital (1000 TND) 3) Obtenir certificat négatif INNORPI 4) S'immatriculer au RNE."
                else:
                    return "To create a SARL: 1) Draft articles 2) Deposit capital (1000 TND) 3) Get INNORPI negative certificate 4) Register with RNE."
            else:
                if language == "french":
                    return "Création d'entreprise en Tunisie : choix forme juridique, statuts, dépôt capital, certificat négatif, immatriculation RNE, publication JORT."
                else:
                    return "Company creation in Tunisia: choose legal form, articles, capital deposit, negative certificate, RNE registration, JORT publication."
        
        # Documents questions
        elif any(word in question_lower for word in ['documents', 'document', 'requis', 'required', 'nécessaires']):
            if language == "french":
                return "Documents RNE : statuts notariés, certificat dépôt capital, certificat négatif INNORPI, CIN associés, justificatif siège social."
            else:
                return "RNE documents: notarized articles, capital deposit certificate, INNORPI negative certificate, partners' IDs, registered office proof."
        
        # Forms questions
        elif any(word in question_lower for word in ['formulaire', 'form']):
            if 'association' in question_lower:
                if language == "french":
                    return "Formulaire RNE-F-003 pour créer une association. Disponible sur le site officiel du RNE avec statuts et liste membres."
                else:
                    return "Form RNE-F-003 to create an association. Available on official RNE website with bylaws and member list."
            elif any(word in question_lower for word in ['société', 'company', 'sarl', 'sa']):
                if language == "french":
                    return "Formulaire RNE-F-002 pour immatriculation société. Documents : statuts, capital, certificat négatif, CIN, siège social."
                else:
                    return "Form RNE-F-002 for company registration. Documents: articles, capital, negative certificate, IDs, registered office."
            else:
                if language == "french":
                    return "Formulaires RNE disponibles selon le type : F-001 (personne physique), F-002 (société), F-003 (association)."
                else:
                    return "RNE forms available by type: F-001 (individual), F-002 (company), F-003 (association)."
        
        # Default response
        else:
            if language == "french":
                return "Pour des informations spécifiques sur les entreprises en Tunisie, consultez le site officiel du RNE ou contactez un expert-comptable."
            else:
                return "For specific information about businesses in Tunisia, consult the official RNE website or contact a local accountant."

openai_service = OpenAIService()