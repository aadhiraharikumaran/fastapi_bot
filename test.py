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
# FAQ Chatbot Functions with proper error handling
# ----------------------------
def fetch_numbered_data():
"""Fetch all content with numbers from FAQ database"""
db_path = "extracted_data.db"
if not os.path.exists(db_path):
logger.error(f"FAQ Database file not found at '{db_path}'. FAQ functionality will be disabled.")
        return {}, {}
        # Return default content instead of empty dictionaries
        default_content = {
            1: "Narayan Seva Sansthan is a non-profit organization dedicated to serving the underprivileged through various humanitarian initiatives including medical camps, disability aid, and community support programs."
        }
        default_keywords = {
            1: "narayan seva sansthan, nonprofit, charity, donation, help, medical camps, disability aid"
        }
        return default_content, default_keywords

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
        # Return default content on error
        default_content = {
            1: "Narayan Seva Sansthan provides various services including medical camps, disability aid, and community support programs. Contact us for more information."
        }
        default_keywords = {
            1: "medical camps, disability aid, community support, narayan seva, donation"
        }
        return default_content, default_keywords


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
# Some few shot examples (truncated for brevity)
FEW_SHOT_EXAMPLES = """
[Your existing FEW_SHOT_EXAMPLES content here - too long to include fully]
Classification:- Donation Related Enquiries, Sub_Classification:- Announce Related
-दिव्यांग बच्चों के भोजन लिए सहयोग 3000 का
5000/ का
दिव्यांग कन्या विवाह  हेतु सादर समर्पित...धन्यवाद..
में संस्थान से जुड़कर करना चाहता हु।
-I will pay 4.5k
-Ok I will pay
What I can do what is the payment
M sansthan ka Sahyog karna chahta hu

Classification:- Donation Related Enquiries, Sub_Classification:- Receipts Related
-I didn't get receipt for 4500
-No need to send receipt pls 🙏🏻
-Yes  only send me the donation receipt for ten thousand also send hard copy by post
-Rasid Sohan Ram Prajapat ke Name se Mil jayega kya
-Recipt भेज दो na sir ji
-Please send receipt
-Sorry, actually I need the receipts for July 24 & August 24..Kindly do the needful 🙏🏻🙏🏻
-Is there any Receipt ??
-Please send receipt
-कृपया सितंबर 2024 में दी गई डोनेशन राशि रुपए 10000 की रसीद एवं इनकम टैक्स सर्टिफिकेट प्रदान करने की कृपा करें
-Can you send for final year 2024-2025
-Can you please share all for last financial year
-U can share the receipt..if possible??
-Nd send the receipt again with correct name..
-Thanks. No receipt/80G benefits needed.
-rasid NAme : Shah Hansaben Manharlal
-"Pls. Send receipt  of deposit amount
-रसीद की हार्ड कॉपी जरूर भेजना।
-Subject: Request for Acknowledgement Receipts - July & August 2025
Dear Sir/Ma'am,
I have not yet received the acknowledgement receipts for the months of July 2025 and August 2025. May I kindly request you to share the same at the earliest.
Your support in this matter will be highly appreciated.
Thanks & regards,
Nilesh Bhagat 🙏🏻"
-Receipt plz ??
-Rasid. Sanjeev Kumar
-"PLEASE SEND ME RECEIPT ON WHATSAPP
-NO NEED TO SEND BY POST"
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
# Fixed Classification Function - Always returns a dictionary
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
def classify_message_with_gemini(message: str, gemini_model, request_id) -> dict:
    if not gemini_model:
        logger.error(f"{request_id}:-Gemini model not initialized, returning default classification")
        return {"classification": "General|No_Module", "confidence": "LOW",
                "reasoning": "Gemini client not available", "Interested_To_Donate": "no",
                "Question_Language": "hi", "Question_Script": "Devanagari"}
    
    if not message or not message.strip():
        logger.warning(f"{request_id}:-Empty message received, returning default classification")
        return {"classification": "General|No_Module", "confidence": "MEDIUM",
                "reasoning": "Empty or whitespace message", "Interested_To_Donate": "no",
                "Question_Language": "hi", "Question_Script": "Devanagari"}
    
    prompt = f"""You are a sophisticated classification AI for Narayan Seva Sansthan.
    Your primary task is to analyze the user's input and return a single, valid JSON object with no extra text or explanations.

    Analyze the following input:
    - User Message: {message}
    - Here are some Few Shot Examples: {FEW_SHOT_EXAMPLES}

    Respond in this exact JSON format:
    {{
    "classification": "classification_name|Sub_classification_name",
    "Interested_To_Donate": "yes/no",
    "Question_Language": "Question_language_name",
    "Question_Script": "Question_script_name",
    "confidence": "HIGH/MEDIUM/LOW",
    "reasoning": "Brief explanation for the classification",
    }}
    """
    
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
        response = gemini_model.generate_content(prompt)
        result_text = response.text.strip().replace("```json\n", "").replace("\n```", "").strip()
        logger.debug(f"{request_id}:-Raw Gemini response: {result_text}")

        # Check for errors in response
        if hasattr(response, 'error') and response.error:
            logger.error(f"{request_id}:-Supabase insert error: {response.error}")
            await log_to_local_file(log_data, request_id)
            return False
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            json_text = json_match.group()
        else:
            json_text = result_text

        logger.success(f"{request_id}:-Successfully logged to Supabase table '{table}'")
        return True
        result = json.loads(json_text)
        logger.info(f"{request_id}:-Message classified as: {result.get('classification')} with confidence: {result.get('confidence')}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"{request_id}:-JSON parsing error in classification: {str(e)}, Raw response: {result_text}")
        return {"classification": "General|No_Module", "confidence": "LOW",
                "reasoning": f"JSON parsing error: {str(e)}", "Interested_To_Donate": "no",
                "Question_Language": "hi", "Question_Script": "Devanagari"}
except Exception as e:
        logger.error(f"{request_id}:-Supabase log failed: {str(e)}")
        logger.error(f"{request_id}:-Error type: {type(e).__name__}")
        # Fallback to local logging
        await log_to_local_file(log_data, request_id)
        return False
        logger.error(f"{request_id}:-Gemini classification error: {str(e)}")
        return {"classification": "General|No_Module", "confidence": "LOW",
                "reasoning": f"API error: {str(e)}", "Interested_To_Donate": "no",
                "Question_Language": "hi", "Question_Script": "Devanagari"}

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
# Greeting response LLM Function
# ----------------------------
async def forward_message_to_replica(payload: dict, request_id):
    replica_url = "https://nss-code-replica.onrender.com/message"
async def LLM_reply_greeting(
        Question_Script,
        Question_Language,
        original_message: str,
        user_name: str,
        gemini_model,
        WA_Msg_Type,
        request_id
) -> str:
    if not gemini_model:
        return f"🙏 Jai Shree Narayan {user_name}! Narayan Seva Sansthan se sampark karne ke liye dhanyawad, mai apki kaise sahayta kar sakti hu?"

try:
        safe_payload = serialize_datetime_recursive(payload)
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(replica_url, json=safe_payload)
            logger.info(f"{request_id}:-Forwarded message to replica. Status: {response.status_code}")
        if WA_Msg_Type and WA_Msg_Type.lower() == "image":
            prompt = '''
            You are Priya, a helpful and friendly assistant for Narayan Seva Sansthan. Your sole purpose is to generate a warm, welcoming greeting reply based on the provided user information and the content of an image transcription.

            ### Rules and Guidelines:
            IMPORTANT:- below You will get the transcription of an image, generate a reply to it like someone has sent you that message in the same language with same script

            1. Identify the primary greeting: Read the "Text" section of the image transcription to identify the greeting. If multiple greetings are present (e.g., "jai hanuman jai shree ram jai shree krishna"), select **only one** to use in your reply. Prioritize the most prominent greeting from the transcription.
            2. Personalize the greeting:
                - If a `user_name` is provided, include it naturally in the greeting. For example, "Hello [Name] ji!" or "नमस्ते [Name] जी!".
                - If no `user_name` is available, use a generic greeting.
            3. Craft a natural response:
                - The greeting you generate should be in the same language and script as detected from the transcription (e.g., "Suprabhat" is Hindi in Latin script, "सुप्रभात" is Hindi in Devanagari script).
                - Use a conversational and friendly tone. Don't simply repeat the exact greeting word-for-word if it sounds unnatural. For instance, if the image has "Jai Shri Ram Jai Shree Krishna Jai ram," a good response would be "Jai Shri Krishna! How can I assist you today?" Just pick any one out of it and maintain the question language and script
            4. Add a helpful message: After the initial greeting, add a friendly and helpful message. This could be a question like "How can I assist you today?" or a simple statement of welcome.
            5. Avoid extra information: Do not include any explanations, tags, or formatting in your final response. Only provide the greeting message.
            6. Understand the previous chat history, use if required otherwise stick to your task for being a Greetings bot
            7. Use Emojis whenever necessary
            8. Reply with a nicely formatted message, which should look like it is sent by a human
            9. Do not generate a response greater than 300 characters
            '''
            prompt += f'''
            \nName:- {user_name}
            \nQuestion's Language:- {Question_Language}
            \nQuestion's Script:- {Question_Script}
            \n User's Question {original_message}
            '''

            response = gemini_model.generate_content(prompt)
            dynamic_response = response.text.strip()
            if len(dynamic_response) > 300 or not dynamic_response:
                raise Exception("Response too long or empty")
            return dynamic_response
        else:
            prompt = '''
            \nYou are Narayan Seva Sansthan's Assistant.\n
            \nYour name is Priya, introduce yourself along with the greeting.\n
            \nYour role in this step is ONLY to generate a greeting reply.\n
            \nYou can be a bit creative, show that you are very happy to receive message from them, don't literally say it
            \nAsk how can I help you

            \nRules:\n
            \n1. Understand the previous chat history, use if required otherwise stick to your task for being a Greetings bot
            \n3. If the user_name is recognized (old user) and their name is provided, always include their name naturally in the greeting.\n
            \n- Example: "Hello Ramesh ji! How are you today?"\n
            \n- Example: "नमस्ते सीमा जी! आप कैसी हैं?"\n
            \n4. Always respond in the same combination of "Question's_language" and "Questions_script" as detected.\n
            \n- If language = "hi" and script = "Latin" → Hindi in Latin script.\n
            \n- If language = "hi" and script = "Devanagari" → Hindi in Devanagari script.\n
            \n- If language = "en" and script = "Latin" → English in Latin script.\n
            \n5. Do not translate into any other language or script. Always mirror the detected language + script.\n
            \n6. Do not add any explanations, tags, or formatting. Only output the greeting message.\n
            eg if the user says "Radhe radhe" you too should reply with " Radhe Radhe Rahul ji, ..... ?"
            eg if the user says "Hello" you should reply with "Hi Rahul ji,...."
            eg if the user says "जय श्री कृष्णा" you should reply "जय श्री कृष्णा साहिल जी,.....
            \n7. Use Emojis whenever necessary
            '''
            prompt += f'''
            \nName:- {user_name}
            \nQuestion's Language:- {Question_Language}
            \nQuestion's Script:- {Question_Script}
            \n User's Question {original_message}
            '''

            response = gemini_model.generate_content(prompt)
            dynamic_response = response.text.strip()
            if len(dynamic_response) > 300 or not dynamic_response:
                raise Exception("Response too long or empty")
            return dynamic_response

except Exception as e:
        logger.error(f"{request_id}:-Failed to forward message to replica: {e}")
        logger.error(f"{request_id}:-Dynamic greeting generation failed: {e},{repr(e)}", exc_info=True)
        return f"🙏 Jai Shree Narayan {user_name}! Narayan Seva Sansthan se sampark karne ke liye dhanyawad, mai apki kaise sahayta kar sakti hu?"

supabase: Client = None
gemini_model = None
numbered_content = {}
keywords_summary = {}

# ----------------------------
# Enhanced Lifespan handler
# Follow-Up response LLM Function
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, gemini_model, numbered_content, keywords_summary
    logger.info("Starting FastAPI app on port {}", os.getenv('PORT', 10000))

async def LLM_reply_follow_up(
        Question_Script,
        Question_Language,
        original_message: str,
        user_name: str,
        gemini_model,
        request_id):
    if not gemini_model:
        return f"Jai Shree Narayan {user_name}!, hum aapki baat samajh rahe hain, aapki maang jald puri karne ki koshish karenge 🙏\nAdhik jankari ke liye iss number par sampark kijiye: +91-294 66 22 222\ndhanyawad🙏"
try:
        supabase = get_supabase_client()
        logger.success("Supabase client initialized successfully")
    except Exception as e:
        logger.error("Supabase connection failed: {}", e)
        supabase = None
        prompt = f'''
        You are Priya, Narayan Seva Sansthan's friendly assistant.
        Your only role here is to generate a reply to FOLLOW UP related MESSAGE.
        Be natural, positive, and show happiness in receiving the message.

        Rules:-
        1. Firstly read the chat History and try to identify with what respect is the user asking follow-up about and then based on that design the reply to follow-up message.
        2. The language should be compassionate and avoid being overly formal or robotic
        3. If user_name is provided, always include it naturally with "ji".
        4. Mirror the detected Question's Language and Script exactly.
        \n. Always respond in the same combination of "Question's_language" and "Questions_script" as detected.\n
        \n- If language = "hi" and script = "Latin" → Hindi in Latin script.\n
        \n- If language = "hi" and script = "Devanagari" → Hindi in Devanagari script.\n
        \n- If language = "en" and script = "Latin" → English in Latin script.\n
        eg. "if user says "jawab do please" and earlier he asked about the donation receipt, firstly since the question language is hindi and script is english "Ji Sahil Ji, so your reply should somewhat be like(whatever the username is), shama chahta hu, apki donation receipt ko hum jald sei jald aap tak pohochane ki koshish krenge.\nDeri ke liye maafi chahte hai.🙏"
        3. The message must reference the specific context of the last communication.
        4. Provide assurance that their request will be completed as soon as possible.
        5. Give them a positive reply
        6. not to long not to short, sweet and simple
        7. Be apologetic.
        8. Do not generate a response greater than 300 characters

        I have provided the elements needed for analysis and reply creation below
        \nName:- {user_name}
        \nQuestion's Language:- {Question_Language}
        \nQuestion's Script:- {Question_Script}
        \nUser's Input:- {original_message}
        '''
        response = gemini_model.generate_content(contents=prompt)
        dynamic_response = response.text.strip()
        if len(dynamic_response) > 300 or not dynamic_response:
            raise Exception("Response too long or empty")
        return dynamic_response

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
        logger.error(f"{request_id}:-Dynamic follow_up generation failed: {e},{repr(e)}", exc_info=True)
        return f"Jai Shree Narayan {user_name}!, hum aapki baat samajh rahe hain, aapki maang jald puri karne ki koshish karenge 🙏\nAdhik jankari ke liye iss number par sampark kijiye: +91-294 66 22 222\ndhanyawad🙏"

    yield
    logger.info("Application shutdown complete")

# ----------------------------
# FastAPI app
# Ok response LLM Function
# ----------------------------
app = FastAPI(
    title="WhatsApp Message Processor with AI Classification",
    description="WhatsApp message processing service with AI classification and Supabase logging",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)
async def LLM_reply_ok(
        Question_Script,
        Question_Language,
        original_message: str,
        user_name: str,
        gemini_model,
        request_id):
    if not gemini_model:
        return f"Thik hai {user_name} ji, Narayan Seva Sansthan aapke sahayta ke liye hamesha hai, dhanyawad 🙏"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    try:
        prompt = f'''
        You are Priya, Narayan Seva Sansthan's friendly assistant.
        Your only role here is to generate a reply to Ok related MESSAGE.
        Be natural, positive, and show happiness in receiving the message.

        Rules:-
        1. Firstly read the chat History and try to identify with what respect is the user is saying Ok about and then based on that design the reply to Ok related message.
        2. The language should be compassionate and avoid being overly formal or robotic
        3. Mirror the detected Question's Language and Script exactly.
        4. If user asked a question you replied to it and then if the user replied with ok, then first reply reply to his okay and then ask a follow up question if necessary.
        5. If he asked anything about donation and then we gave him the information, reply to that okay and then try to persuade him about the donation.
        6. Be Natural while replying to ok. if you feel there's nothing to add up end the conversation with lets say "happy to help" or something like that.
        \n. Always respond in the same combination of "Question's_language" and "Questions_script" as detected.\n
        \n- If language = "hi" and script = "Latin" → Hindi in Latin script.\n
        \n- If language = "hi" and script = "Devanagari" → Hindi in Devanagari script.\n
        \n- If language = "en" and script = "Latin" → English in Latin script.\n
        eg If the user says "Ji" which is kind of ok/yes in hindi then read the earlier context, check to what he has said ok, and reply accordingly. lets say we asked him is he willing to donate to 2 tricycle which costs 7000, and for that he said "Ji" my reply could be like "Thikey, Sahil Ji! Jab aap aage ki donation process complete kar lein, toh please humein batayega. Ek aur cheez jo hum suggest karna chahenge: agar aap isi amount mein ₹2,000 aur add karein, toh aap ek nahin, do logon ki help kar payenge!\n Hum aapke sahayta ke liye tatpar hai".
        7. Be natural and generate a reply which suits the conversation.
        8. Do not generate a response greater than 300 characters

        I have provided the elements needed for analysis and reply creation below
        \nName:- {user_name}
        \nQuestion's Language:- {Question_Language}
        \nQuestion's Script:- {Question_Script}
        \nUser's Input:- {original_message}
        '''
        response = gemini_model.generate_content(contents=prompt)
        dynamic_response = response.text.strip()
        if len(dynamic_response) > 300 or not dynamic_response:
            raise Exception("Response too long or empty")
        return dynamic_response

    except Exception as e:
        logger.error(f"{request_id}:-Dynamic ok generation failed: {e},{repr(e)}", exc_info=True)
        return f"Thik hai {user_name} ji, Narayan Seva Sansthan aapke sahayta ke liye hamesha hai, dhanyawad 🙏"

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
# Enhanced Supabase logging with fallback
# ----------------------------
@app.get("/debug/supabase", tags=["Debug"])
async def debug_supabase():
    """Debug Supabase connection"""
async def log_to_local_file(log_data: dict, request_id: str):
    """Fallback local file logging"""
try:
        if not supabase:
            return {"status": "error", "message": "Supabase client not initialized"}
        
        # Test query
        result = supabase.table("message_logs").select("id").limit(1).execute()
        os.makedirs("local_logs", exist_ok=True)
        log_file = f"local_logs/logs_{datetime.now().strftime('%Y-%m-%d')}.json"

        return {
            "status": "success",
            "message": "Supabase connection working",
            "table_exists": True,
            "test_result": "Connection successful"
        }
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                **log_data
            }, indent=2, ensure_ascii=False) + ",\n")
            
        logger.info(f"{request_id}:-Logged locally to {log_file}")
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
        logger.error(f"{request_id}:-Local logging also failed: {e}")

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
async def log_to_supabase(log_data: dict, request_id, table: str = "message_logs"):
    """Enhanced Supabase logging with proper error handling
