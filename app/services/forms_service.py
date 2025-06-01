from typing import List, Optional, Dict
import re
import google.generativeai as genai
from app.core.config import settings
import json

class RNEForm:
    def __init__(self, code: str, title: str, subtitle: str, url: str):
        self.code = code
        self.title = title
        self.subtitle = subtitle
        self.url = url
    
    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "title": self.title,
            "subtitle": self.subtitle,
            "url": self.url
        }

class FormsService:
    def __init__(self):
        # Initialize Gemini for LLM-based form detection
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Available forms - NO KEYWORDS, just data
        self.forms = [
            RNEForm(
                code="RNE-F-001",
                title="**RNE F 001 Déclaration Immatriculation Personne Physique**",
                subtitle="Formulaire d'immatriculation d'une personne physique",
                url="https://home.registre-entreprises.tn/wp-content/uploads/2022/10/RNE-F-001_declation_immatriculation_personne_physique.pdf"
            ),
            RNEForm(
                code="RNE-F-002",
                title="**RNE F 002 Déclaration Immatriculation Personne Morale**",
                subtitle="Formulaire d'immatriculation d'une personne morale",
                url="https://home.registre-entreprises.tn/wp-content/uploads/2022/10/RNE-F-002_declation_immatriculation_personne_motale.pdf"
            ),
            RNEForm(
                code="RNE-F-003",
                title="**RNE F 003 Déclaration Immatriculation Association**",
                subtitle="Formulaire d'immatriculation d'une association",
                url="https://home.registre-entreprises.tn/wp-content/uploads/2022/10/RNE-F-003_declation_immatriculation_association.pdf"
            ),
            RNEForm(
                code="RNE-F-004",
                title="**RNE F 004 Déclaration Modification Personne Physique**",
                subtitle="Formulaire de modification d'une personne physique",
                url="https://home.registre-entreprises.tn/wp-content/uploads/2022/10/RNE-F-004_declaration_modification_personne_physique.pdf"
            ),
            RNEForm(
                code="RNE-F-005",
                title="**RNE F 005 Déclaration Modification Société/Établissement Public**",
                subtitle="Formulaire de modification d'une société ou établissement public",
                url="https://home.registre-entreprises.tn/wp-content/uploads/2022/10/RNE-F-005_declaration_modification_personne_morale.pdf"
            ),
            RNEForm(
                code="RNE-F-006",
                title="**RNE F 006 Déclaration Modification Association**",
                subtitle="Formulaire de modification d'une association",
                url="https://home.registre-entreprises.tn/wp-content/uploads/2023/12/RNE-F-006-declaration_modification_association.pdf"
            ),
            RNEForm(
                code="RNE-F-007",
                title="**RNE F 007 Demande Traduction Extrait Registre**",
                subtitle="Formulaire de demande de traduction d'un extrait du registre",
                url="https://home.registre-entreprises.tn/wp-content/uploads/2023/06/RNE-F-006-DEMANDE-DE-TRADUCTION-DUN-EXTRAIT-DU-REGISTRE-1.pdf"
            )
        ]
    
    def find_relevant_forms(self, query: str, context: str = "") -> List[RNEForm]:
        """Find forms using PURE LLM - NO HARDCODING"""
        
        # Create forms description for LLM
        forms_description = ""
        for form in self.forms:
            forms_description += f"- {form.code}: {form.title} ({form.subtitle})\n"
        
        prompt = f"""
        You are an expert on RNE Tunisia forms. Analyze the user query and context to determine which RNE forms are relevant.

        AVAILABLE FORMS:
        {forms_description}

        EXAMPLES:

        Example 1:
        Query: "Comment immatriculer une SARL ?"
        Response: ["RNE-F-002"]
        Reason: SARL is a société (personne morale), needs F-002 for registration

        Example 2:
        Query: "Déclaration Immatriculation Personne Physique"
        Response: ["RNE-F-001"]
        Reason: Explicitly asks for individual person registration

        Example 3:
        Query: "How to modify company information ?"
        Response: ["RNE-F-005"]
        Reason: Company modification needs F-005

        Example 4:
        Query: "Formulaire pour créer une association"
        Response: ["RNE-F-003"]
        Reason: Association registration needs F-003

        Example 5:
        Query: "Quel est le capital minimum ?"
        Response: []
        Reason: Just asking about capital, no form needed

        User Query: "{query}"
        Context: "{context}"

        Return ONLY a JSON array with relevant form codes:
        ["RNE-F-XXX", "RNE-F-YYY"] or []

        Rules:
        - Return maximum 2 forms
        - Be very specific to the user's actual need
        - If just asking general questions (like capital amount), return []
        - Only suggest forms when user actually needs to do registration/modification/translation
        """
        
        try:
            response = self.model.generate_content(prompt)
            print(f"Forms LLM response: {response.text}")
            
            # Extract JSON array from response
            text = response.text.strip()
            
            # Try to parse JSON array
            try:
                form_codes = json.loads(text)
            except:
                # Try to extract array from text
                array_match = re.search(r'\[(.*?)\]', text)
                if array_match:
                    array_content = array_match.group(1)
                    form_codes = re.findall(r'["\']([^"\']*)["\']', array_content)
                else:
                    form_codes = []
            
            # Find forms by codes
            relevant_forms = []
            for code in form_codes:
                for form in self.forms:
                    if form.code == code:
                        relevant_forms.append(form)
                        break
            
            print(f"Selected forms for '{query}': {[f.code for f in relevant_forms]}")
            return relevant_forms
            
        except Exception as e:
            print(f"Error in LLM forms detection: {e}")
            return []
    
    def get_form_by_code(self, code: str) -> Optional[RNEForm]:
        """Get a specific form by its code"""
        for form in self.forms:
            if form.code.lower() == code.lower():
                return form
        return None
    
    def get_all_forms(self) -> List[RNEForm]:
        """Get all available forms"""
        return self.forms

forms_service = FormsService()