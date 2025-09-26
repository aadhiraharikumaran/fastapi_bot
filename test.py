from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import os
from dotenv import load_dotenv
import uvicorn
from typing import Optional, Any
from contextlib import asynccontextmanager
from loguru import logger
import sys
import asyncio
import uuid
import json
import sqlite3
from supabase import create_client, Client
import google.generativeai as genai
import httpx
import re

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Configure logger
# ----------------------------
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    colorize=True
)

if os.getenv("DEBUG", "False").lower() != "true":
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO"
    )


# ----------------------------
# Enhanced Supabase client with debugging
# ----------------------------
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    logger.info(f"Supabase URL: {url}")
    logger.info(f"Supabase Key present: {bool(key)}")
    
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY) must be set")
        logger.error(f"URL: {url}, Key present: {bool(key)}")
        raise ValueError("Missing Supabase configuration")

    if os.getenv("SUPABASE_SERVICE_KEY"):
        logger.info("Using Supabase service key (bypasses RLS)")
    else:
        logger.warning("Using anon key - ensure RLS policies allow inserts")

    try:
        client = create_client(url, key)
        # Test connection with a simple query
        result = client.table('message_logs').select('id').limit(1).execute()
        logger.success("Supabase connection test successful")
        return client
    except Exception as e:
        logger.error(f"Supabase connection failed: {str(e)}")
        raise


# ----------------------------
# Gemini AI Client Setup
# ----------------------------
def get_gemini_client():
    """Initialize Gemini client with API key from environment"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("Gemini client and model initialized successfully")
        return model
    except AttributeError as e:
        logger.error(
            f"Failed to initialize Gemini client (AttributeError): {str(e)}. Ensure google-generativeai is up-to-date (run: pip install --upgrade google-generativeai)")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while initializing Gemini client: {str(e)}")
        return None


# ----------------------------
# FAQ Chatbot Functions
# ----------------------------
def fetch_numbered_data():
    """Fetch all content with numbers from FAQ database"""
    db_path = "extracted_data.db"
    if not os.path.exists(db_path):
        logger.error(f"FAQ Database file not found at '{db_path}'. FAQ functionality will be disabled.")
        return {}, {}

    logger.info(f"🗄️ Fetching numbered content from FAQ database ({db_path})")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT keywords, content FROM extracted_data")
        records = cursor.fetchall()
        conn.close()

        numbered_content = {}
        keywords_summary = {}

        for i, record in enumerate(records, 1):
            numbered_content[i] = record[1]  # Full content
            keywords_summary[i] = record[0]  # Keywords for selection

        logger.success(f"✅ Loaded {len(numbered_content)} numbered FAQ content sections")
        return numbered_content, keywords_summary

    except Exception as e:
        logger.error(f"💥 FAQ Database error: {str(e)}")
        return {}, {}


def llm_select_best_content(query, keywords_summary, gemini_model, request_id):
    """Let LLM select the best content number using Gemini"""
    logger.info(f"{request_id}:-🧠 LLM analyzing query for FAQ: '{query[:50]}...'")

    if not gemini_model:
        logger.warning("Gemini client not available for FAQ selection")
        return 1

    try:
        sections_text = "\n".join([f"{num}. {keywords[:100]}..." for num, keywords in keywords_summary.items()])

        selection_prompt = f"""
You are helping select the most relevant content section for this user query about Narayan Seva Sansthan.

Available Content Sections:
{sections_text}

User Query: "{query}"

Analyze the query and respond with ONLY the number (1-{len(keywords_summary)}) of the most relevant content section.
Do not provide any explanation, just the number.
"""

        response = gemini_model.generate_content(selection_prompt)
        selected_number = response.text.strip()

        try:
            selected_num = int(selected_number)
            if 1 <= selected_num <= len(keywords_summary):
                logger.success(f"{request_id}:-✅ LLM selected content section #{selected_num}")
                return selected_num
            else:
                logger.warning(f"{request_id}:-⚠️ LLM returned invalid number: {selected_number}")
                return 1
        except ValueError:
            logger.warning(f"{request_id}:-⚠️ LLM returned non-numeric response: {selected_number}")
            return 1

    except Exception as e:
        logger.error(f"{request_id}:-💥 LLM FAQ selection error: {str(e)}")
        return 1


def generate_faq_response(content, question, gemini_model, request_id):
    """Generate final FAQ answer using selected content"""
    logger.info("🤖 Generating FAQ response")

    if not gemini_model:
        return "Sorry, our FAQ service is temporarily unavailable."

    try:
        prompt = f"""
You are a humble Sevak (volunteer) of Narayan Seva Sansthan.
Answer the following question kindly and devotionally, using this content:

Content:
{content}

Question:
{question}

Provide a short, sweet, and crisp answer in plain text (max 3-4 sentences). 
Do not add extra commentary. 
"""

        response = gemini_model.generate_content(prompt)
        faq_answer = response.text.strip()
        logger.success(f"{request_id}:-✅ Final response generated ({len(faq_answer)} chars)")
        return faq_answer

    except Exception as e:
        logger.error(f"{request_id}:-💥 FAQ response generation error: {str(e)}")
        return f"Sorry, could not generate an answer at the moment: {str(e)}"


# ----------------------------
# Image Analysis Function
# ----------------------------
async def analyze_image_with_gemini(image_url: str, gemini_model, request_id) -> dict:
    if not gemini_model:
        return {"transcription": "", "status": "error", "error": "Gemini client not available"}

    try:
        logger.info(f"{request_id}:- Analyzing image from URL: {image_url}")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()

        mime_type = resp.headers.get('content-type', 'image/jpeg').split(';')[0]

        image_part = {
            "mime_type": mime_type,
            "data": resp.content
        }

        response = gemini_model.generate_content(
            ["Explain what is in this image clearly and in detail.", image_part]
        )

        transcription = response.text.strip()
        logger.info(f"{request_id}:- Image analysis completed. Transcription length: {len(transcription)}")
        return {"transcription": transcription, "status": "success", "error": None}

    except Exception as e:
        logger.error(f"{request_id}:- Image analysis error: {e},{image_url},{repr(e)}", exc_info=True)
        return {"transcription": "", "status": "error", "error": str(e)}


# -------------------------------
# Donation Processing
# -------------------------------
async def process_donation_transcript(transcript: str, user_name: str, gemini_model, request_id) -> dict:
    if not transcript or not gemini_model:
        return {
            "is_donation_screenshot": False,
            "extraction_details": {},
            "generated_response": None
        }

    try:
        prompt = f"""
Analyze the following transcript to determine if it contains donation/payment details and generate an appropriate response.

TRANSCRIPT: "{transcript}"
USER NAME: {user_name}

TASK 1 - DETECTION:
Analyze if this transcript contains donation/payment information. Look for:
- Payment amounts (₹, Rs., rupees, numbers with currency symbols)
- Transaction IDs (UPI transaction ID, transaction ID, txn ID, reference number, UTR, Google Pay transaction ID)
- Payment apps (Google Pay, PhonePe, Paytm, UPI, Bank transfer)
- Dates/timestamps
- Payment-related keywords (paid, sent, received, transaction, transfer, amount, successful, completed)
- Bank account details or payment confirmations
- Screenshot indicators of payment apps

TASK 2 - EXTRACTION:
If donation detected, extract these details:
- amount: Exact amount with currency symbol (e.g., "₹500", "₹1,000")
- transaction_id: Exact transaction/reference/UPI ID found
- date_time: Date/time if mentioned (format: DD/MM/YYYY or DD-MM-YYYY)
- payment_app: Payment method (Google Pay, PhonePe, Paytm, UPI, Bank Transfer)
- detected_language: Determine if response should be in "hindi" or "english" based on transcript content
- account_no: Bank account number or UPI ID if mentioned

TASK 3 - RESPONSE GENERATION:
If donation detected, generate a warm acknowledgment message following these rules:

FOR HINDI RESPONSES:
- Start with: "जय नारायण {user_name} जी!"
- Thank for the SPECIFIC amount extracted
- Include transaction ID if found: "लेनदेन ID: [transaction_id]"
- Include date if found: "दिनांक: [date]"
- Mention: "आपकी रसीद जल्द ही भेजी जाएगी।"
- Say the donation "will truly make a significant difference in achieving our goals" (translate to Hindi)
- End with: "कृतज्ञता सहित,\nनारायण सेवा संस्थान"
- Use ONLY Hindi throughout

FOR ENGLISH RESPONSES:
- Start with: "Dear {user_name},"
- Thank for the SPECIFIC amount extracted
- Include transaction ID if found: "Transaction ID: [transaction_id]"
- Include date if found: "Date: [date]"
- Mention: "Your receipt will be sent shortly."
- Say the donation "will truly make a significant difference in achieving our goals"
- End with: "With heartfelt gratitude,\nNarayan Seva Sansthan"
- Use ONLY English throughout

IMPORTANT RULES:
1. If ANY detail is missing (amount, transaction ID, or date), create acknowledgment WITHOUT including that missing detail
2. Use ACTUAL extracted values, never use placeholders like "N/A", "[amount]", or "Not available"
3. Do not say "धन्यवाद पत्र" - always use "रसीद" for receipt in Hindi
4. If no donation detected, return is_donation_screenshot as false
5. Only return true for is_donation_screenshot if there are clear payment/transaction indicators

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "is_donation_screenshot": true/false,
    "extraction_details": {{
        "amount": "extracted amount with ₹ symbol or null",
        "transaction_id": "extracted ID or null",
        "date_time": "extracted date or null",
        "payment_app": "detected app name or null",
        "detected_language": "hindi/english",
        "account_no": "extracted account number or null",
    }},
    "generated_response": "complete acknowledgment message or null"
}}
"""

        response = gemini_model.generate_content(prompt)
        result_text = response.text.strip()

        # Fixed regex to handle JSON code block properly
        result_text = re.sub(r'^```json\n', '', result_text, flags=re.MULTILINE)
        result_text = re.sub(r'\n```$', '', result_text, flags=re.MULTILINE)
        result_text = result_text.strip()

        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            json_text = json_match.group()
        else:
            json_text = result_text

        try:
            result = json.loads(json_text)
        except json.JSONDecodeError:
            fixed_json = json_text.replace("true/false", "false").replace("null,", "null,")
            result = json.loads(fixed_json)

        if not isinstance(result, dict):
            raise ValueError("Invalid response format from LLM")

        extraction_details = result.get("extraction_details", {})
        if extraction_details:
            cleaned_details = {}
            for key, value in extraction_details.items():
                if value is not None and value != "null" and str(value).strip():
                    cleaned_details[key] = value
            extraction_details = cleaned_details

        if result.get("is_donation_screenshot") and 'detected_language' not in extraction_details:
            extraction_details['detected_language'] = 'hindi'

        logger.info(
            f"{request_id}:-Unified donation analysis result: {result.get('is_donation_screenshot')}, Details: {extraction_details}")

        return {
            "is_donation_screenshot": result.get("is_donation_screenshot", False),
            "extraction_details": extraction_details,
            "generated_response": result.get("generated_response", None)
        }

    except json.JSONDecodeError as e:
        logger.error(f"{request_id}:-JSON parsing error in unified donation processing: {e},{repr(e)}", exc_info=True)
        logger.error(f"{request_id}:-Raw LLM response: {result_text}")
        return {
            "is_donation_screenshot": False,
            "extraction_details": {},
            "generated_response": None
        }

    except Exception as e:
        logger.error(f"{request_id}:-Error in unified donation processing: {e},{repr(e)}", exc_info=True)
        return {
            "is_donation_screenshot": False,
            "extraction_details": {},
            "generated_response": None
        }


# Some few shot examples (truncated for brevity - keep your existing FEW_SHOT_EXAMPLES)
FEW_SHOT_EXAMPLES = """
[Your existing FEW_SHOT_EXAMPLES content here - too long to include fully]
"""

def classify_message_with_gemini(message: str, gemini_model, request_id) -> dict:
    # [Your existing classify_message_with_gemini function - unchanged]
    pass

async def LLM_reply_greeting(Question_Script, Question_Language, original_message: str, user_name: str, gemini_model, WA_Msg_Type, request_id) -> str:
    # [Your existing LLM_reply_greeting function - unchanged]
    pass

async def LLM_reply_follow_up(Question_Script, Question_Language, original_message: str, user_name: str, gemini_model, request_id):
    # [Your existing LLM_reply_follow_up function - unchanged]
    pass

async def LLM_reply_ok(Question_Script, Question_Language, original_message: str, user_name: str, gemini_model, request_id):
    # [Your existing LLM_reply_ok function - unchanged]
    pass

# ----------------------------
# Enhanced Supabase logging with fallback
# ----------------------------
async def log_to_local_file(log_data: dict, request_id: str):
    """Fallback local file logging"""
    try:
        os.makedirs("local_logs", exist_ok=True)
        log_file = f"local_logs/logs_{datetime.now().strftime('%Y-%m-%d')}.json"
        
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                **log_data
            }, indent=2, ensure_ascii=False) + ",\n")
            
        logger.info(f"{request_id}:-Logged locally to {log_file}")
    except Exception as e:
        logger.error(f"{request_id}:-Local logging also failed: {e}")

async def log_to_supabase(log_data: dict, request_id, table: str = "message_logs"):
    """Enhanced Supabase logging with proper error handling"""
    try:
        if not supabase:
            logger.error(f"{request_id}:-Supabase client not initialized")
            # Fallback to local file logging
            await log_to_local_file(log_data, request_id)
            return False

        logger.info(f"{request_id}:-Attempting to log to Supabase table '{table}'")
        
        # Remove any None values that might cause issues
        clean_data = {k: v for k, v in log_data.items() if v is not None}
        serialized_data = serialize_datetime_recursive(clean_data)
        
        logger.debug(f"{request_id}:-Log data prepared for Supabase")
        
        # Insert into Supabase
        response = supabase.table(table).insert(serialized_data).execute()
        
        # Check for errors in response
        if hasattr(response, 'error') and response.error:
            logger.error(f"{request_id}:-Supabase insert error: {response.error}")
            await log_to_local_file(log_data, request_id)
            return False
            
        logger.success(f"{request_id}:-Successfully logged to Supabase table '{table}'")
        return True
        
    except Exception as e:
        logger.error(f"{request_id}:-Supabase log failed: {str(e)}")
        logger.error(f"{request_id}:-Error type: {type(e).__name__}")
        # Fallback to local logging
        await log_to_local_file(log_data, request_id)
        return False

# ----------------------------
# Helper functions
# ----------------------------
def serialize_datetime_recursive(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetime_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_recursive(item) for item in obj]
    return obj

# ----------------------------
# Forward to replica system
# ----------------------------
async def forward_message_to_replica(payload: dict, request_id):
    replica_url = "https://nss-code-replica.onrender.com/message"
    try:
        safe_payload = serialize_datetime_recursive(payload)
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(replica_url, json=safe_payload)
            logger.info(f"{request_id}:-Forwarded message to replica. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"{request_id}:-Failed to forward message to replica: {e}")

supabase: Client = None
gemini_model = None
numbered_content = {}
keywords_summary = {}

# ----------------------------
# Enhanced Lifespan handler
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, gemini_model, numbered_content, keywords_summary
    logger.info("Starting FastAPI app on port {}", os.getenv('PORT', 10000))

    try:
        supabase = get_supabase_client()
        logger.success("Supabase client initialized successfully")
    except Exception as e:
        logger.error("Supabase connection failed: {}", e)
        supabase = None

    try:
        gemini_model = get_gemini_client()
        if gemini_model:
            logger.success("Gemini AI initialized successfully")
        else:
            logger.error("Gemini AI initialization failed")
    except Exception as e:
        logger.error("Gemini AI initialization failed: {}", e)
        gemini_model = None

    numbered_content, keywords_summary = fetch_numbered_data()
    if not numbered_content:
        logger.warning("FAQ content could not be loaded. FAQ chatbot functionality will be limited.")
    else:
        logger.success(f"Loaded {len(numbered_content)} FAQ content sections")

    yield
    logger.info("Application shutdown complete")

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="WhatsApp Message Processor with AI Classification",
    description="WhatsApp message processing service with AI classification and Supabase logging",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Pydantic models (unchanged)
# ----------------------------
class MessageRequest(BaseModel):
    WA_Auto_Id: Optional[int] = None
    WA_In_Out: Optional[str] = None
    Account_Code: Optional[int] = None
    WA_Received_At: Optional[datetime] = None
    NGCode: Optional[int] = None
    Wa_Name: Optional[str] = None
    MobileNo: Optional[str] = None
    WA_Msg_To: Optional[str] = None
    WA_Msg_Text: Optional[str] = None
    WA_Msg_Type: Optional[str] = None
    Integration_Type: Optional[str] = None
    WA_Message_Id: Optional[str] = None
    WA_Url: Optional[str] = None
    Status: Optional[str] = "success"
    Donor_Name: Optional[str] = None

class MessageResponse(BaseModel):
    phone_number: str
    ai_response: str
    ai_reason: str
    WA_Auto_Id: Optional[int]
    WA_Message_Id: Optional[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str

# ----------------------------
# Debug endpoints for Supabase testing
# ----------------------------
@app.get("/debug/supabase", tags=["Debug"])
async def debug_supabase():
    """Debug Supabase connection"""
    try:
        if not supabase:
            return {"status": "error", "message": "Supabase client not initialized"}
        
        # Test query
        result = supabase.table("message_logs").select("id").limit(1).execute()
        
        return {
            "status": "success",
            "message": "Supabase connection working",
            "table_exists": True,
            "test_result": "Connection successful"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-log", tags=["Debug"])
async def test_log():
    """Test logging functionality"""
    request_id = "test-" + str(uuid.uuid4())[:8]
    test_data = {
        "test": True,
        "timestamp": datetime.now().isoformat(),
        "message": "Test log entry from debug endpoint",
        "request_id": request_id
    }
    
    success = await log_to_supabase(test_data, request_id)
    return {
        "status": "success" if success else "partial_success",
        "message": "Test log completed" if success else "Test log completed with fallback",
        "request_id": request_id,
        "supabase_connected": supabase is not None
    }

# ----------------------------
# /message endpoint (unchanged except for enhanced logging)
# ----------------------------
@app.post("/message", response_model=MessageResponse, tags=["Message Processing"])
async def handle_message(request: MessageRequest):
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    phone_number = request.MobileNo or request.WA_Msg_To or "Unknown"

    payload = request.model_dump(exclude_none=True)
    log_data = {
        "request_id": request_id,
        "endpoint": "/message",
        "method": "POST",
        "status": "processing",
        "processing_start_time": start_time.isoformat(),
        "raw_request": payload,
        "wa_auto_id": request.WA_Auto_Id,
        "wa_in_out": request.WA_In_Out,
        "account_code": request.Account_Code,
        "wa_received_at": request.WA_Received_At,
        "ng_code": request.NGCode,
        "wa_name": request.Wa_Name,
        "mobile_no": phone_number,
        "wa_msg_to": request.WA_Msg_To,
        "wa_msg_text": request.WA_Msg_Text,
        "wa_msg_type": request.WA_Msg_Type,
        "integration_type": request.Integration_Type,
        "wa_message_id": request.WA_Message_Id,
        "wa_url": request.WA_Url,
        "status": request.Status,
        "donor_name": request.Donor_Name
    }

    # Use enhanced logging function
    await log_to_supabase(log_data, request_id)

    # [Rest of your handle_message function remains unchanged]
    # ... your existing message processing logic ...

    # Handle different message types
    message_text = request.WA_Msg_Text or ""
    user_name = request.Donor_Name or request.Wa_Name or "Sevak"
    wa_msg_type = request.WA_Msg_Type.lower() if request.WA_Msg_Type else None

    # Classify the message
    classification_result = classify_message_with_gemini(message_text, gemini_model, request_id)
    classification = classification_result.get("classification", "General|No_Module")
    confidence = classification_result.get("confidence", "LOW")
    reasoning = classification_result.get("reasoning", "No classification provided")
    interested_to_donate = classification_result.get("Interested_To_Donate", "no")
    question_language = classification_result.get("Question_Language", "hi")
    question_script = classification_result.get("Question_Script", "Devanagari")

    # Log classification result
    log_data.update({
        "classification": classification,
        "confidence": confidence,
        "reasoning": reasoning,
        "interested_to_donate": interested_to_donate,
        "question_language": question_language,
        "question_script": question_script
    })

    # Handle image messages
    if wa_msg_type == "image" and request.WA_Url:
        image_analysis = await analyze_image_with_gemini(request.WA_Url, gemini_model, request_id)
        log_data["image_transcription"] = image_analysis.get("transcription")
        log_data["image_analysis_status"] = image_analysis.get("status")
        log_data["image_analysis_error"] = image_analysis.get("error")

        if image_analysis.get("status") == "success" and image_analysis.get("transcription"):
            donation_result = await process_donation_transcript(
                image_analysis["transcription"], user_name, gemini_model, request_id
            )
            log_data["donation_analysis"] = donation_result

            if donation_result.get("is_donation_screenshot"):
                ai_response = donation_result.get("generated_response",
                                                  "Thank you for your donation! We'll process it soon.")
                log_data["ai_response"] = ai_response
                log_data["status"] = "success"
                await log_to_supabase(log_data, request_id)
                await forward_message_to_replica(payload, request_id)
                return MessageResponse(
                    phone_number=phone_number,
                    ai_response=ai_response,
                    ai_reason=reasoning,
                    WA_Auto_Id=request.WA_Auto_Id,
                    WA_Message_Id=request.WA_Message_Id
                )

    # Handle specific classifications
    main_classification, sub_classification = classification.split("|") if "|" in classification else (
    classification, "No_Module")

    if main_classification == "General":
        if sub_classification == "Greeting":
            ai_response = await LLM_reply_greeting(
                question_script, question_language, message_text, user_name, gemini_model, wa_msg_type, request_id
            )
        elif sub_classification == "Follow-up":
            ai_response = await LLM_reply_follow_up(
                question_script, question_language, message_text, user_name, gemini_model, request_id
            )
        elif sub_classification == "Ok":
            ai_response = await LLM_reply_ok(
                question_script, question_language, message_text, user_name, gemini_model, request_id
            )
        else:
            ai_response = f"🙏 Jai Shree Narayan {user_name}! Thank you for contacting Narayan Seva Sansthan. How can I assist you today?"
    elif main_classification in ["Donation Related Enquiries", "Ticket Related Enquiry"]:
        if interested_to_donate == "yes":
            ai_response = f"🙏 Jai Shree Narayan {user_name}! Thank you for your interest in donating. Please share your preferred donation method (e.g., UPI, bank transfer) or visit https://x.ai/donate for details."
        else:
            selected_content_num = llm_select_best_content(message_text, keywords_summary, gemini_model, request_id)
            selected_content = numbered_content.get(selected_content_num, "No relevant content found.")
            ai_response = generate_faq_response(selected_content, message_text, gemini_model, request_id)
    else:
        selected_content_num = llm_select_best_content(message_text, keywords_summary, gemini_model, request_id)
        selected_content = numbered_content.get(selected_content_num, "No relevant content found.")
        ai_response = generate_faq_response(selected_content, message_text, gemini_model, request_id)

    log_data["ai_response"] = ai_response
    log_data["status"] = "success"
    await log_to_supabase(log_data, request_id)
    await forward_message_to_replica(payload, request_id)

    return MessageResponse(
        phone_number=phone_number,
        ai_response=ai_response,
        ai_reason=reasoning,
        WA_Auto_Id=request.WA_Auto_Id,
        WA_Message_Id=request.WA_Message_Id
    )

# ----------------------------
# /health endpoint
# ----------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
async def health_check():
    supabase_status = "connected" if supabase else "disconnected"
    gemini_status = "connected" if gemini_model else "disconnected"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service=f"WhatsApp Message Processor (Supabase: {supabase_status}, Gemini: {gemini_status})",
        version="1.0.0"
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
