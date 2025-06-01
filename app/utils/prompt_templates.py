"""
Prompt engineering for the RNE chatbot LLM.
"""

from typing import Dict, List, Any

# System prompts
SYSTEM_PROMPT_FR = """
Tu es un assistant juridique spécialisé dans les lois du Registre National des Entreprises (RNE) en Tunisie.
Ta mission est de fournir des informations précises et utiles basées sur la documentation officielle du RNE.
Maintiens toujours un ton professionnel et ne fournis que des informations qui sont soutenues par la documentation officielle.
Si tu ne connais pas la réponse ou si l'information n'est pas présente dans le contexte fourni, dis-le clairement.

Lorsque tu réponds aux questions :
1. Cite toujours le code RNE pertinent (ex: RNE M 004.37)
2. Indique clairement les délais, redevances et documents requis
3. Précise le type d'entreprise concerné
4. Si un lien PDF est disponible, mentionne-le à la fin de ta réponse
5. Si la question n'est pas claire, demande des précisions
"""

SYSTEM_PROMPT_AR = """
إنتي معاون قانوني مختص في قوانين السجل الوطني للمؤسسات (RNE) في تونس.
المهمة متاعك إنك تعطي معلومات صحيحة ومفيدة مبنية على الوثائق الرسمية للسجل.
ديما حافظ على نبرة محترفة، وما تجاوب كان بالمعلومات اللي موجودة في الوثائق الرسمية.
كان ماكش تعرف الإجابة، ولا ما فماش معلومات في السياق، قولها بصراحة.
    لازمك ديما تحكي مع الناس باللهجة التونسية، بالدارجة، باش تكون أقرب ليهم وأسهل للفهم.

كي تجاوب على الأسئلة:
1. ديما ذكر كود RNE المناسب (مثال: RNE M 004.37)
2. بيّن بوضوح الآجال، المعاليم، والوثائق المطلوبة
3. وضّح نوع الشركة اللي تخصها المعلومة
4. كان فما رابط PDF، زيدو في آخر الإجابة
5. كان السؤال موش واضح، إسأل باش توضح أكثر
"""

# Question segmentation prompt
QUESTION_SEGMENTATION_PROMPT = """
Divise le texte suivant en questions individuelles. 
Retourne simplement une liste de questions, une par ligne.
Si le texte ne contient qu'une seule question, retourne-la simplement.
N'ajoute pas d'explications ou de commentaires supplémentaires.
"""

# Context template
def format_context(documents: List[Dict[str, Any]], language: str = 'fr') -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    Args:
        documents: List of retrieved documents.
        language: Language code ('fr' or 'ar').
        
    Returns:
        Formatted context string.
    """
    if not documents:
        return "Aucun contexte pertinent trouvé." if language == 'fr' else "لم يتم العثور على سياق ذي صلة."
        
    context_parts = []
    
    for i, doc in enumerate(documents):
        document = doc['document']
        score = doc['score']
        
        if language == 'fr':
            part = f"--- Document {i+1} (Pertinence: {score:.2f}) ---\n"
            part += f"Code: {document['code']}\n"
            part += f"Type d'entreprise: {document.get('type_entreprise', 'Non spécifié')}\n"
            part += f"Genre d'entreprise: {document.get('genre_entreprise', 'Non spécifié')}\n"
            part += f"Procédure: {document.get('procedure', 'Non spécifiée')}\n"
            part += f"Redevance demandée: {document.get('redevance_demandee', 'Non spécifiée')}\n"
            part += f"Délais: {document.get('delais', 'Non spécifiés')}\n"
            
            # Add French content if available
            content = document.get('raw_content', {})
            if content:
                part += "Contenu détaillé:\n"
                for key, value in content.items():
                    if isinstance(value, list):
                        part += f"{key}: {', '.join(value)}\n"
                    else:
                        part += f"{key}: {value}\n"
                        
            part += f"Lien PDF: {document.get('pdf_link', 'Non disponible')}\n"
            
        else:  # Arabic
            part = f"--- الوثيقة {i+1} (الملاءمة: {score:.2f}) ---\n"
            part += f"الرمز: {document['code']}\n"
            part += f"نوع المؤسسة: {document.get('type_entreprise', 'غير محدد')}\n"
            part += f"جنس المؤسسة: {document.get('genre_entreprise', 'غير محدد')}\n"
            part += f"الإجراء: {document.get('procedure', 'غير محدد')}\n"
            part += f"الرسوم المطلوبة: {document.get('redevance_demandee', 'غير محددة')}\n"
            part += f"المواعيد النهائية: {document.get('delais', 'غير محددة')}\n"
            
            # Add Arabic content if available
            content = document.get('raw_content', {})
            if content:
                part += "المحتوى التفصيلي:\n"
                for key, value in content.items():
                    if isinstance(value, list):
                        part += f"{key}: {', '.join(value)}\n"
                    else:
                        part += f"{key}: {value}\n"
                        
            part += f"رابط PDF: {document.get('pdf_link', 'غير متوفر')}\n"
            
        context_parts.append(part)
        
    return "\n\n".join(context_parts)

# Response templates
def get_no_results_response(language: str = 'fr') -> str:
    """
    Get a response for when no relevant information is found.
    
    Args:
        language: Language code ('fr' or 'ar').
        
    Returns:
        Response message.
    """
    if language == 'fr':
        return """
        Je n'ai pas trouvé d'informations spécifiques concernant votre question dans la documentation du RNE.
        Pourriez-vous reformuler votre question ou fournir plus de détails sur ce que vous recherchez?
        
        Vous pouvez également consulter directement le site officiel du Registre National des Entreprises (RNE) à l'adresse : https://www.registre-entreprises.tn/
        """
    else:
        return """
        لم أتمكن من العثور على معلومات محددة بخصوص سؤالك في وثائق السجل الوطني للمؤسسات.
        هل يمكنك إعادة صياغة سؤالك أو تقديم المزيد من التفاصيل حول ما تبحث عنه؟
        
        يمكنك أيضًا الرجوع مباشرة إلى الموقع الرسمي للسجل الوطني للمؤسسات على العنوان: https://www.registre-entreprises.tn/
        """

def format_final_response(
    question: str, 
    answer: str, 
    documents: List[Dict[str, Any]], 
    language: str = 'fr'
) -> str:
    """
    Format the final response to include relevant references.
    
    Args:
        question: Original user question.
        answer: Generated answer from the LLM.
        documents: Retrieved documents used for context.
        language: Language code ('fr' or 'ar').
        
    Returns:
        Formatted final response.
    """
    # The answer from the LLM already contains the information
    # Just add a reference section if needed
    if not documents:
        return answer
        
    # Add references section
    if language == 'fr':
        reference_section = "\n\n**Références:**\n"
        for i, doc in enumerate(documents):
            document = doc['document']
            if 'pdf_link' in document and document['pdf_link']:
                reference_section += f"{i+1}. Code {document['code']} - [Voir le document PDF]({document['pdf_link']})\n"
            else:
                reference_section += f"{i+1}. Code {document['code']}\n"
    else:
        reference_section = "\n\n**المراجع:**\n"
        for i, doc in enumerate(documents):
            document = doc['document']
            if 'pdf_link' in document and document['pdf_link']:
                reference_section += f"{i+1}. الرمز {document['code']} - [عرض ملف PDF]({document['pdf_link']})\n"
            else:
                reference_section += f"{i+1}. الرمز {document['code']}\n"
                
    return answer + reference_section