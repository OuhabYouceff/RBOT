from app.services.gemini_service import gemini_service
from app.services.rag_service import rag_service
from app.services.forms_service import forms_service
from app.models.chat import ChatRequest, ChatResponse, FollowUpQuestion, ErrorResponse, RNEFormData
from typing import Union, List

class ChatPipelineService:
    
    def _extract_conversation_context(self, conversation_history: List, current_message: str) -> str:
        """Extract relevant context from conversation history"""
        if not conversation_history or len(conversation_history) == 0:
            return ""
        
        # Get last few messages for context
        recent_messages = conversation_history[-6:]  # Last 6 messages
        
        context_parts = []
        for msg in recent_messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                if msg.type == "user":
                    context_parts.append(f"User asked: {msg.content}")
                elif msg.type == "bot":
                    # Extract key info from bot responses
                    content = msg.content
                    if "formulaire" in content.lower() or "form" in content.lower():
                        context_parts.append(f"Bot mentioned forms: {content[:100]}...")
                    elif "suarl" in content.lower() or "sarl" in content.lower():
                        context_parts.append(f"Bot discussed company types: {content[:100]}...")
                    elif "document" in content.lower():
                        context_parts.append(f"Bot mentioned documents: {content[:100]}...")
        
        context = "\n".join(context_parts[-4:])  # Last 4 relevant parts
        
        if context:
            print(f"📋 Extracted context: {context[:200]}...")
        
        return context
    
    async def process_chat(self, request: ChatRequest) -> Union[ChatResponse, ErrorResponse]:
        """Main chat pipeline with conversation history awareness"""
        try:
            user_query = request.message
            conversation_history = request.conversation_history or []
            
            print(f"🎯 Processing: '{user_query}' with {len(conversation_history)} history messages")
            
            # Extract conversation context
            conversation_context = self._extract_conversation_context(conversation_history, user_query)
            
            # Create enhanced query with context for analysis
            enhanced_query = user_query
            if conversation_context:
                enhanced_query = f"Context from conversation:\n{conversation_context}\n\nCurrent user question: {user_query}"
            
            # Step 1: Check if needs additional information (with context)
            print("📋 Step 1: Checking if follow-up needed...")
            follow_up_result = await gemini_service.check_needs_additional_info(enhanced_query)
            
            if follow_up_result[0]:
                follow_up_question = follow_up_result[0]
                main_response = follow_up_result[1]
                print(f"✅ Follow-up needed: {follow_up_question.question}")
                
                return ChatResponse(
                    answer=main_response,
                    follow_up=follow_up_question,
                    suggestions=[],
                    forms=[]
                )
            
            print("✅ No follow-up needed, continuing...")
            
            # Step 2: Segment query into questions (with context)
            print("📋 Step 2: Segmenting query...")
            segmented_result = await gemini_service.segment_query(enhanced_query)
            questions_to_process = segmented_result.questions
            print(f"Segmented into: {questions_to_process}")
            
            # Step 3: Process each question with OpenAI
            print("📋 Step 3: Processing questions with OpenAI...")
            rag_results = await rag_service.query_multiple(questions_to_process)
            
            for i, result in enumerate(rag_results):
                print(f"Question {i+1}: '{result.question}' -> Source: {result.source}")
            
            # Step 4: Format final answer (with conversation context)
            print("📋 Step 4: Formatting final answer...")
            rag_dict_results = [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "confidence": r.confidence,
                    "source": r.source
                }
                for r in rag_results
            ]
            
            language = gemini_service._detect_language(user_query)
            
            # Include conversation context in final answer formatting
            final_result = await gemini_service.format_final_answer(
                enhanced_query,  # Use enhanced query with context
                rag_dict_results, 
                language
            )
            
            final_answer = final_result.get("answer", "Une erreur s'est produite.")
            final_suggestions = final_result.get("suggestions", [])
            forms_suggested = final_result.get("suggest_forms", False)
            
            # Step 5: Find relevant forms (with context)
            print("📋 Step 5: Finding relevant forms...")
            relevant_forms = []
            
            if forms_suggested:
                # Include conversation context for better form detection
                forms_context = f"{conversation_context}\n{final_answer} {' '.join([r['answer'] for r in rag_dict_results])}"
                forms_found = forms_service.find_relevant_forms(user_query, forms_context)
                
                relevant_forms = [
                    RNEFormData(
                        code=form.code,
                        title=form.title,
                        subtitle=form.subtitle,
                        url=form.url
                    )
                    for form in forms_found
                ]
                
                print(f"Found {len(relevant_forms)} forms: {[f.code for f in relevant_forms]}")
            
            print("✅ Pipeline completed successfully")
            
            return ChatResponse(
                answer=final_answer,
                follow_up=None,
                suggestions=final_suggestions,
                forms=relevant_forms
            )
            
        except Exception as e:
            print(f"💥 Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            
            return ErrorResponse(error=f"Une erreur s'est produite: {str(e)}")

chat_pipeline_service = ChatPipelineService()