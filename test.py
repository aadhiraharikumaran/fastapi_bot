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
from supabase import create_client, Client
import google.generativeai as genai
from google.generativeai.types import ContentDict
import httpx
import requests

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
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO"
    )

# ----------------------------
# Supabase client
# ----------------------------
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY) must be set")
        raise ValueError("Missing Supabase configuration")

    if os.getenv("SUPABASE_SERVICE_KEY"):
        logger.info("Using Supabase service key (bypasses RLS)")
    else:
        logger.warning("Using anon key - ensure RLS policies allow inserts")

    return create_client(url, key)

# ----------------------------
# Gemini AI Classification Setup
# ----------------------------
def get_gemini_client():
    """Initialize Gemini client with API key from environment"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return None
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini client initialized successfully")
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

# ----------------------------
# Image Analysis Function
# ----------------------------
async def analyze_image_with_gemini(image_url: str, gemini_client) -> dict:
    """Analyze image using Gemini API and return transcription"""
    if not gemini_client:
        return {
            "transcription": "",
            "status": "error",
            "error": "Gemini client not available"
        }

    try:
        logger.info(f"Analyzing image from URL: {image_url}")

        # Download image
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()

        # Get MIME type from response headers or default to jpeg
        mime_type = resp.headers.get('content-type', 'image/jpeg').split(';')[0]

        # Create content for Gemini
        image_content = ContentDict(
            parts=[
                {"text": "Explain what is in this image clearly and in detail."},
                {"inline_data": {"mime_type": mime_type, "data": resp.content}}
            ]
        )

        # Send to Gemini
        response = gemini_client.generate_content(image_content)

        transcription = response.text.strip()
        logger.info(f"Image analysis completed. Transcription length: {len(transcription)}")

        return {
            "transcription": transcription,
            "status": "success",
            "error": None
        }

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return {
            "transcription": "",
            "status": "error",
            "error": str(e)
        }

# Few-shot examples for classification
FEW_SHOT_EXAMPLES = """
DONATION & TICKET RELATED ENQUIRIES:
- "I'm interested . Please mention purpose of acceptance of donation. I have send to feed Divang children in memory of my demised wife Manorama Vijay."
- "Can we send contribution in the above account?"
- "Please send receipt"
- "Pls send pay tm link for donation"
- "Kindly send my last year's donation recepit for income tax"
- "Send final receipt to claim rebate"
- "Payment karo 900"
- "6500/- paid"
- "Receipt plz ??"
- "My today's donation ☝️my donation Id is 1047733"
- "Finally I succeeded today in transferring ₹2000/- to the sanstha a/c for feeding children on amavasya.🙏"
- "Donation for bhadrapada shani amavasya"
- "For children's meal"
- "Can you share 1 pdf for all 12 months donation for tax itr"

EDUCATION & TRAINING ENQUIRIES:
- "Us time pese pareeksha aa gahi thi to me sekhne ke liye nahi aa para tha"
- "20.9.25ke bad muje computer course karna h"
- "आप के यहां विकलांगो को part time जॉब मिलेगा क्या"
- "Education jankari chahiye"

GENERAL INFORMATION ENQUIRIES:
- "Glad to have information sent by you"
- "Aapka fecebook me video dekh kar aap se sampark kar rahe hai"
- "में आप से मिलना चाहता हूँ बच्चे के साथ"
- "Katha karna chahta hu"
- "जानकारी दीजिये"
- "Narayan Seva Sansthan se. Ham judna chahte hain. aur katha bhi karna chahte hain."
- "Katha karne ke liye tyar hai"
- "Call kb kr skte he"
- "कब का डेट हो सकता है कथा हमारी"
- "M sansthan ka Sahyog karna chahta hu"
- "Sir mujhe apni bhanji ko dikhana hai"
- "Lucknow me kab tak camp lage ga"
- "कैंप kha lage ga"

GREETING RELATED TEXT:
- "Ok thanks."
- "Ok"
- "Dhanyawad Shri Radhe😊"
- "Jay -jay shri narayan, jay narayan, jay shri narayan. 🙏🙏🙏🙏🙏🙏🙏"
- "आपका बहुत बहुत आभार"
- "Hii"
- "🙏💐 Ram Ram ji 🙏🙏"
- "Jay shree shyam..."
- "Namaste👃👃"
- "Jai shree shyam 🙏"
- "Good morning saheb. God bless all of you and have a nice day."
- "Ram ram ji"
- "Jai Narayan"

MEDICAL / TREATMENT ENQUIRIES:
- "सर क्या यह सुविधा मुझे हरिद्वार में मिल सकता है हरिद्वार फिजियोथैरेपी केंद्र का डिटेल्स मिल सकता है"
- "मैं अपने बच्चों के लिए दिखाना चाहता हूं इस संस्था में कब आना होगा"
- "मैं दोनों पैरों से दिव्यांग हु"
- "I am a disabled person."
- "My name is, Randhir Kumar Singh,age-42,I am from bihar.my left leg go back hipper from knee.I feel very pain and my walking is difficult have you any solution?"
- "तो मेरे बेटे का ऑपरेशन का बोला था"
- "Above knee left leg"
- "Tumor near liver ..need to get surgery done ...robotic"
- "Kya mujhe usme above knee artificial limb mil sakta hai"
- "Sir ji mera train se pair kat gya tha"
- "मेरा नाम राजकुमार है।मेरा दाहिने पैर गुदने के निचे से कट गया है।"
- "Me ek pair se viklang"

OPERATIONAL / CALL HANDLING ENQUIRIES:
- "Please call me"
- "Not connecting your number"
- "विश्व प्रेम संस्थान. वृन्दावन आपके नारायण सेवा संस्थान की सेवा के लिए सदैव समर्पित है.... आप 9675333000 पर संपर्क कर सकते हैं"

SPAM:
- Instagram/Facebook/YouTube links
- Unrelated promotional content
- Random forwards and links
"""

def classify_message_with_gemini(message: str, gemini_client) -> dict:
    """Classify message using Gemini API"""
    if not gemini_client:
        return {
            "classification": "GENERAL INFORMATION ENQUIRIES",
            "confidence": "LOW",
            "reasoning": "Gemini client not available"
        }

    if not message or message.strip() == "":
        return {
            "classification": "GREETING RELATED TEXT",
            "confidence": "MEDIUM",
            "reasoning": "Empty or whitespace message"
        }

    prompt = f"""
You are a message classification system for a social service organization. Based on the following examples, classify the given message into one of these categories:

1. DONATION & TICKET RELATED ENQUIRIES
2. EDUCATION & TRAINING ENQUIRIES  
3. GENERAL INFORMATION ENQUIRIES
4. GREETING RELATED TEXT
5. MEDICAL / TREATMENT ENQUIRIES
6. OPERATIONAL / CALL HANDLING ENQUIRIES
7. SPAM

Here are the few-shot examples:

{FEW_SHOT_EXAMPLES}

Now classify this message: "{message}"

Respond in this exact JSON format:
{{
    "classification": "CATEGORY_NAME",
    "confidence": "HIGH/MEDIUM/LOW",
    "reasoning": "Brief explanation for the classification"
}}
"""

    try:
        response = gemini_client.generate_content(prompt)
        result_text = response.text.strip()

        # Clean up the response if it has markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()

        result = json.loads(result_text)
        logger.info(
            f"Message classified as: {result.get('classification')} with confidence: {result.get('confidence')}")
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error from Gemini response: {e}")
        return {
            "classification": "GENERAL INFORMATION ENQUIRIES",
            "confidence": "LOW",
            "reasoning": f"JSON parsing error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {
            "classification": "GENERAL INFORMATION ENQUIRIES",
            "confidence": "LOW",
            "reasoning": f"Gemini API error: {str(e)}"
        }

supabase: Client = None
gemini_client = None

# ----------------------------
# Lifespan handler
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, gemini_client
    logger.info("Starting FastAPI app on port {}", os.getenv('PORT', 10000))

    try:
        supabase = get_supabase_client()
        logger.info("Supabase Configuration: ✓ Set")
    except Exception as e:
        logger.error("Supabase connection failed: {}", e)

    try:
        gemini_client = get_gemini_client()
        if gemini_client:
            logger.info("Gemini AI Configuration: ✓ Set")
        else:
            logger.warning("Gemini AI Configuration: ✗ Not available")
    except Exception as e:
        logger.error("Gemini AI initialization failed: {}", e)

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
# Helper functions
# ----------------------------
def serialize_datetime_recursive(obj: Any) -> Any:
    """Recursively serialize datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetime_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_recursive(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_datetime_recursive(item) for item in obj)
    else:
        return obj

def to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Safely convert datetime to ISO string"""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        return dt.isoformat()
    try:
        return dt.isoformat()
    except AttributeError:
        logger.warning(f"Cannot convert {type(dt)} to ISO format: {dt}")
        return str(dt)

# ----------------------------
# Pydantic models
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
# Improved Supabase logging
# ----------------------------
async def log_to_supabase(log_data: dict, table: str = "message_logs"):
    """Log data to Supabase with proper datetime serialization"""
    try:
        if supabase:
            serialized_data = serialize_datetime_recursive(log_data.copy())
            json.dumps(serialized_data)
            result = supabase.table(table).insert(serialized_data).execute()
            logger.debug(f"Successfully logged to Supabase table '{table}' with {len(serialized_data)} fields")
            return result
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization error: {e}")
        logger.error(f"Problematic data keys: {list(log_data.keys())}")
    except Exception as e:
        logger.error(f"Supabase log failed: {e}")

# ----------------------------
# /message Forward to system 2 for testing
# ----------------------------
async def forward_message_to_replica(payload: dict):
    """Forward message payload to external replica service"""
    replica_url = "https://nss-code-replica.onrender.com/message"
    try:
        safe_payload = serialize_datetime_recursive(payload)
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(replica_url, json=safe_payload)
            logger.info(f"Forwarded message to replica. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to forward message to replica: {e}")

# ----------------------------
# /message endpoint with AI classification
# ----------------------------
@app.post("/message", response_model=MessageResponse, tags=["Message Processing"])
async def handle_message(request: MessageRequest):
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    phone_number = request.MobileNo or request.WA_Msg_To or "Unknown"
    payload = request.model_dump(exclude_none=True)
    asyncio.create_task(forward_message_to_replica(payload))

    log_data = {
        "request_id": request_id,
        "endpoint": "/message",
        "method": "POST",
        "status": "processing",
        "processing_start_time": start_time,
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
        "message_length": len(request.WA_Msg_Text) if request.WA_Msg_Text else 0,
        "parameters_received": len([k for k, v in payload.items() if v is not None]),
        "includes_wa_auto_id": request.WA_Auto_Id is not None,
        "includes_wa_message_id": request.WA_Message_Id is not None,
        "donor_name": request.Donor_Name,
        "transcription": None,
    }

    try:
        classification_result = {
            "classification": "GENERAL INFORMATION ENQUIRIES",
            "confidence": "LOW",
            "reasoning": "Classification not available"
        }

        transcription = None

        if request.WA_Msg_Type and request.WA_Msg_Type.lower() == "image" and request.WA_Url:
            logger.info(f"Processing image message from {phone_number}")
            image_analysis = await analyze_image_with_gemini(request.WA_Url, gemini_client)
            if image_analysis["status"] == "success":
                transcription = image_analysis["transcription"]
                log_data["transcription"] = transcription
                if transcription and gemini_client:
                    logger.info(f"Classifying image transcription: {transcription[:100]}...")
                    classification_result = classify_message_with_gemini(transcription, gemini_client)
                else:
                    classification_result = {
                        "classification": "GENERAL INFORMATION ENQUIRIES",
                        "confidence": "LOW",
                        "reasoning": "Image transcription unavailable"
                    }
            else:
                logger.error(f"Image analysis failed: {image_analysis['error']}")
                log_data["transcription"] = f"Error: {image_analysis['error']}"
                classification_result = {
                    "classification": "GENERAL INFORMATION ENQUIRIES",
                    "confidence": "LOW",
                    "reasoning": f"Image analysis failed: {image_analysis['error']}"
                }
        elif request.WA_Msg_Text and gemini_client:
            logger.info(f"Classifying text message: {request.WA_Msg_Text[:100]}...")
            classification_result = classify_message_with_gemini(request.WA_Msg_Text, gemini_client)
        elif not request.WA_Msg_Text and not transcription:
            classification_result = {
                "classification": "GREETING RELATED TEXT",
                "confidence": "MEDIUM",
                "reasoning": "No message text or image transcription available"
            }

        log_data.update({
            "ai_classification": classification_result["classification"],
            "ai_confidence": classification_result["confidence"],
            "ai_reasoning": classification_result["reasoning"]
        })

        # Modified response logic for greeting messages with polite tone and emojis
        if classification_result["classification"] == "GREETING RELATED TEXT":
            if request.Donor_Name:
                ai_response = f"Greetings {request.Donor_Name}, we sincerely appreciate your message! Jai Narayan! 🙏✨"
            else:
                ai_response = "Greetings, we sincerely appreciate your message! Jai Narayan! 🙏✨"
        else:
            ai_response = "Thank you for reaching out! We're unable to respond to this query at the moment. Please contact our support team for assistance. 🙏😊"

        response_data = {
            "phone_number": phone_number,
            "ai_response": ai_response,
            "ai_reason": classification_result["classification"]
        }

        if request.WA_Auto_Id is not None:
            response_data["WA_Auto_Id"] = request.WA_Auto_Id
        if request.WA_Message_Id is not None:
            response_data["WA_Message_Id"] = request.WA_Message_Id

        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        log_data.update({
            "status": "success",
            "processing_end_time": end_time,
            "processing_duration_ms": duration_ms,
            "response_phone_number": response_data["phone_number"],
            "response_ai_response": response_data["ai_response"],
            "response_ai_reason": response_data["ai_reason"],
            "response_wa_auto_id": response_data.get("WA_Auto_Id"),
            "response_wa_message_id": response_data.get("WA_Message_Id"),
            "raw_response": response_data
        })

        asyncio.create_task(log_to_supabase(log_data))
        logger.info(
            f"Request {request_id} processed successfully in {duration_ms}ms. Classification: {classification_result['classification']}")
        return MessageResponse(**response_data)

    except Exception as e:
        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        log_data.update({
            "status": "error",
            "processing_end_time": end_time,
            "processing_duration_ms": duration_ms,
            "error_type": "internal_error",
            "error_message": str(e),
            "raw_response": {"error": str(e)}
        })
        asyncio.create_task(log_to_supabase(log_data))
        logger.error(f"Request {request_id} failed after {duration_ms}ms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Additional endpoints for classification testing
# ----------------------------
@app.get("/categories")
async def get_categories():
    """Get list of available classification categories"""
    return {
        "categories": [
            "DONATION & TICKET RELATED ENQUIRIES",
            "EDUCATION & TRAINING ENQUIRIES",
            "GENERAL INFORMATION ENQUIRIES",
            "GREETING RELATED TEXT",
            "MEDICAL / TREATMENT ENQUIRIES",
            "OPERATIONAL / CALL HANDLING ENQUIRIES",
            "SPAM"
        ]
    }

@app.post("/classify-only", tags=["Classification"])
async def classify_only(request: dict):
    """Standalone classification endpoint for testing"""
    message = request.get("WA_Msg_Text", "")
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini AI not available")
    result = classify_message_with_gemini(message, gemini_client)
    return {
        "message": message,
        "classification": result["classification"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"]
    }

# ----------------------------
# Health & metrics
# ----------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="whatsapp-message-processor",
        version="1.0.0"
    )

@app.get("/metrics")
async def metrics():
    return {
        "service": "whatsapp-message-processor",
        "supabase_enabled": supabase is not None,
        "gemini_ai_enabled": gemini_client is not None,
        "timestamp": datetime.now().isoformat()
    }

# ----------------------------
# Exception handlers
# ----------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found",
                 "available_endpoints": ["/message", "/health", "/metrics", "/categories", "/classify-only"]}
    )

@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    return JSONResponse(
        status_code=405,
        content={"error": "Method not allowed"}
    )

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    host = "0.0.0.0"
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run(
        "test:app",
        host=host,
        port=port,
        workers=workers if not debug_mode else 1,
        reload=debug_mode,
        log_level="debug" if debug_mode else "info"
    )
