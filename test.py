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

    client = create_client(url, key)
    logger.info("Supabase client initialized successfully")
    return client

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

    logger.info(f"Fetching numbered content from FAQ database ({db_path})")

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

        logger.success(f"Loaded {len(numbered_content)} numbered FAQ content sections")
        return numbered_content, keywords_summary

    except Exception as e:
        logger.error(f"FAQ Database error: {str(e)}")
        return {}, {}

def llm_select_best_content(query, keywords_summary, gemini_model, request_id):
    """Let LLM select the best content number using Gemini"""
    logger.info(f"{request_id}:-LLM analyzing query for FAQ: '{query[:50]}...'")

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
                logger.success(f"{request_id}:-LLM selected content section #{selected_num}")
                return selected_num
            else:
                logger.warning(f"{request_id}:-LLM returned invalid number: {selected_number}")
                return 1
        except ValueError:
            logger.warning(f"{request_id}:-LLM returned non-numeric response: {selected_number}")
            return 1

    except Exception as e:
        logger.error(f"{request_id}:-LLM FAQ selection error: {str(e)}")
        return 1

def generate_faq_response(content, question, gemini_model, request_id):
    """Generate final FAQ answer using selected content with formal structure"""
    logger.info("Generating FAQ response")

    if not gemini_model:
        return "Respected Sir/Madam, Jai Narayan! Thank you for contacting Narayan Seva Sansthan. Our service is temporarily unavailable. Please try again shortly. With regards, Narayan Seva Sansthan"

    try:
        prompt = f"""
You are a representative of Narayan Seva Sansthan. Generate a formal, structured response using the provided content.

CONTENT TO USE:
{content}

USER QUESTION:
"{question}"

RESPONSE REQUIREMENTS:
1. Start with "Respected Sir/Madam" or use appropriate salutation
2. Include "Jai Narayan!" greeting
3. Provide clear, helpful information from the content
4. Maintain formal and respectful tone throughout
5. Structure the response with proper paragraphs
6. End with appropriate closing like "With regards," or "Thank you,"
7. Sign off with "Narayan Seva Sansthan"
8. Keep response concise but complete (4-6 sentences maximum)
9. Do not use markdown formatting
10. Ensure the response directly addresses the user's question

Generate the response:
"""

        response = gemini_model.generate_content(prompt)
        faq_answer = response.text.strip()
        
        # Ensure proper formatting
        if not faq_answer.startswith("Respected"):
            faq_answer = f"Respected Sir/Madam, Jai Narayan! {faq_answer}"
        
        if "Narayan Seva Sansthan" not in faq_answer:
            faq_answer += " With regards, Narayan Seva Sansthan"
            
        logger.success(f"{request_id}:-Formal response generated ({len(faq_answer)} chars)")
        return faq_answer.replace("\n", " ").replace("  ", " ").strip()

    except Exception as e:
        logger.error(f"{request_id}:-FAQ response generation error: {str(e)}")
        return "Respected Sir/Madam, Jai Narayan! Thank you for your query. We are experiencing technical difficulties. Please contact us directly for assistance. With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

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
- End with: "कृतज्ञता सहित, नारायण सेवा संस्थान"
- Use ONLY Hindi throughout

FOR ENGLISH RESPONSES:
- Start with: "Dear {user_name},"
- Thank for the SPECIFIC amount extracted
- Include transaction ID if found: "Transaction ID: [transaction_id]"
- Include date if found: "Date: [date]"
- Mention: "Your receipt will be sent shortly."
- Say the donation "will truly make a significant difference in achieving our goals"
- End with: "With heartfelt gratitude, Narayan Seva Sansthan"
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

        generated_response = result.get("generated_response", None)
        if generated_response:
            generated_response = generated_response.replace("\n", " ").replace("  ", " ").strip()

        return {
            "is_donation_screenshot": result.get("is_donation_screenshot", False),
            "extraction_details": extraction_details,
            "generated_response": generated_response
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

# ----------------------------
# Updated Donation response with formal structure  
# ----------------------------
async def generate_donation_response(user_name, gemini_model, request_id):
    """Generate formal donation information response"""
    if not gemini_model:
        return "Respected Sir/Madam, Jai Narayan! We are delighted to know that you wish to support our service. You can make a donation online through our website, transfer directly to our bank account, or send a cheque/money order. You may also have our volunteers visit your home, or contribute in person at our programs or branches. Every contribution of yours is truly a blessing from Narayan. With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

    try:
        prompt = f'''
Create a formal donation information response for Narayan Seva Sansthan.

RESPONSE REQUIREMENTS:
1. Start with "Respected Sir/Madam" 
2. Include "Jai Narayan!" greeting
3. Express delight about their donation interest
4. List donation options clearly:
   - Online through website
   - Direct bank transfer  
   - Cheque/money order
   - Volunteer home visits
   - In-person at programs/branches
5. Include that every contribution is a blessing
6. Maintain formal and grateful tone
7. Structure with proper paragraphs
8. End with appropriate closing

Generate the formal donation response:
'''

        response = gemini_model.generate_content(prompt)
        donation_response = response.text.strip()
        
        if not donation_response:
            raise Exception("Empty response")
            
        return donation_response.replace("\n", " ").replace("  ", " ").strip()

    except Exception as e:
        logger.error(f"{request_id}:-Donation response generation failed: {e}")
        return "Respected Sir/Madam, Jai Narayan! We are delighted to know that you wish to support our service. You can make a donation online through our website, transfer directly to our bank account, or send a cheque/money order. You may also have our volunteers visit your home, or contribute in person at our programs or branches. Every contribution of yours is truly a blessing from Narayan. With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

# Some few shot examples
FEW_SHOT_EXAMPLES = """
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
-I didn’t get receipt for 4500
-No need to send receipt pls 
-Yes  only send me the donation receipt for ten thousand also send hard copy by post
-Rasid Sohan Ram Prajapat ke Name se Mil jayega kya
-Recipt भेज दो na sir ji
-Please send receipt
-Sorry, actually I need the receipts for July 24 & August 24..Kindly do the needful 
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
-Subject: Request for Acknowledgement Receipts – July & August 2025
Dear Sir/Ma'am,
I have not yet received the acknowledgement receipts for the months of July 2025 and August 2025. May I kindly request you to share the same at the earliest.
Your support in this matter will be highly appreciated.
Thanks & regards,
Nilesh Bhagat "
-Receipt plz ??
-Rasid. Sanjeev Kumar
-"PLEASE SEND ME RECEIPT ON WHATSAPP
-NO NEED TO SEND BY POST"

Classification:- Donation Related Enquiries, Sub_Classification:- Amount Confirmation
-Hospital के लिए 100000की सेवा सहयोग भी भेजा था 
-Rs.5100 transferred from A/c ...0501 to:IMPS/P2A/5237164. Total Bal:Rs.43530.21CR. Avlbl Amt:Rs.288530.21, FFD Bal Rs.245000(25-08-2025 16:50:23) - Bank of Baroda
-Donar id 2254151
-Firstly i transferred  Rs. 501  yesterday but not debited my ac & thereafter I transfer rs.1  for checking , both the amounts  of Rs 501 & 1 to your HDFC  ac No. 500-000-759-75997 but the message comes as *there is a customer memo present on the credit amount* Screen shot of Rs. 1 is attached above. Pl  check & show this screen shot to your hdfc bank  as to why the amount is not being credited.  My bank account is with HDFC ac , Bhera enclave, Paschim vihar, N Delhi-110087  - Donor id  396872 saroj malhotra delhi cell no. 9810326214.
-"Sir I have donated 1000rs for needful 
Regards 
-Manju Agarwal 
-W/O Shri Ashok Kumar Singhal 
-R/O 6/3 A Gali barah bhai belanganj Agra"
-Ye 3000 jod jod ker mai banwa dungi
-"(New) Ms. Monika Gupta - ₹21000
Mrs. Raj Kumari Gupta - ₹9000"
Kindly acknowledge the amount I hv donated to your sanstha
-Jo Screen Shot Send kiye hai Maine
-Hi, you have sent Rs.5,500.00 from 9352351847@idfcfirst to paytmqr2810050501010uwohbemahg0@paytm using your IDFC FIRST Bank UPI.Txn ID-523676286360.
-We sent the amount for Haldi and Mehndi for two couples.
-Finally I succeeded today in transferring ₹2000/- to the sanstha a/c for feeding children on amavasya.

Classification:- Donation Related Enquiries, Sub_Classification:- Donation Payment Information
-Can we send contribution in the above account?
-Can we send contribution in the above account with Bank of India?
-Pls send pay tm link for donation
-There are two different account numbers. Which account should I transfer
-Pls send donation details ifsc code account
-Pl send P N  B  Actt No
-ke nama se karna he
-State bank of india
-Mam plz barcode bhej dijiye
-I am requesting you to send Bank details or QR for donations.
-Qr code beje
-Send scanner
-Please send your account details for donations

Classification:- Donation Related Enquiries, Sub_Classification:- KYC Update
-PAN no. Already shared.
-"उपरोक्त Donation मेरे द्वारा मेरे नाम Rajendra Kumar Sharma से किया गया था 
-मेरा ही Pan नम्बर दिया गया था 
-कृपया गलती सुधारने की कोशिश कीजिए 
-अन्यथा Pan नंबर बदलने का कष्ट करें 
-This is my PAN"
-Adhar & pan
-My PAN NO is ACIPR 0141F
-"My PAN. is
-AFSPA 3996 C"
-My PAN is
-PAN NO.
-My above pan no is. Correct.
-Receiptisto inthe name of GaneshiLal yadav. Pan is also in same name. Itis already registered with you.
-Pan number AVKPS1316G
-"PAN NO
-AIFPC3542E"
-Pan no AGVPS9012M
-"AMBPK4143P
-PAN NUMBER"
-Kindly send my last year's donation recepit for income tax
-For getting 80G benefit for last year
-इस में पूरा एड्रेस लिखा हुआ है

Classification:- Donation Related Enquiries, Sub_Classification:- In-Kind Donation
-Can I donate wheelchairs to the sanstha?
-I have some old clothes to donate, where can I send them?
-Want to donate books for children, please guide.
-Can we give medical equipment for the hospital?
-Donating food items for the camp, how to proceed?

Classification:- Donation Related Enquiries, Sub_Classification:- Recurring Donation
-How to set up monthly donations?
-Can I pledge ₹500 every month?
-Is there an option for recurring donations via UPI?
-Want to donate regularly, please share details.
-How do I start a monthly contribution plan?

Classification:- Donation Related Enquiries, Sub_Classification:- Post-Donation Related
-I have sent ₹5000, please confirm receipt.
-Donation of ₹2000 done, what's next?
-Payment completed for Amavasya drive, kindly acknowledge.

Classification:- Donation Related Enquiries, Sub_Classification:- Send Sadhak Related
-Can a sadhak collect my donation from Jaipur?
-Please send a sadhak to my address for donation pickup.
-I want to give donation through a sadhak, share details.

Classification:- Donation Related Enquiries, Sub_Classification:- Property Donation
-I want to donate my land to the sanstha, how to proceed?
-Can I give a property for your hospital project?
-Interested in donating a plot, please guide.

Classification:- Donation Related Enquiries, Sub_Classification:- FD & Will Related
-Can I include sanstha in my will?
-How to set up a fixed deposit for Narayan Seva?
-Want to donate via FD, share the process.

Classification:- Donation Related Enquiries, Sub_Classification:- CSR Donation Interest
-My company wants to contribute via CSR, whom to contact?
-Interested in CSR donation, please send details.
-Can we discuss CSR funding for your programs?

Classification:- General, Sub_Classification:- Greeting
-Om Gajananaya namah. Om Mitraye namah. Radhe Radhe. Jai Sada Shiv. Jai Sarvamangala Mata. Jai Ragya Mata. Jai Bhadrakaali Mata. Jai Sharada Mata. Jai Annapurna Mata. Jai Sheetla Mata. Jai Bhoomi Mata. Jai Mangalmurti Hanuman. Om Swami Nathishoraye namah. Guru kripa. Mangalamay Mangalvaar. Orzu
" *जय श्री राधाकृष्ण*
  *श्रीकृष्णावतार*
-*फिर भगवान् से मांगने की बजाये निकटता बनाओ तो सब कुछ अपने आप मिलना शुरू हो जायेगा ।*
-* जय श्री गणेश जी जय श्री कृष्ण *शुभ रात्रि  जय सियाराम"
-Jay Shri Ram
-Ram ram ji
-राधे राधे 
-Gud Nyt Yu Nd Yr Family Members
-Jai naryana
-जय श्री श्याम जी 
-Hi
- OK   Jay shree Radhey Krishna
-"- 
-Namah vishnu in service of needy"
-Jai Narayan 
-जय नारायण 
- राधे राधे जय श्री कृष्ण
-हम आपके संस्थान से 2010से जुड़े हैं 
-Ram ramji
-"जय नारायण 
-Jay shree Krishna 
-Jai jai shree shyam 
-Jai Shree Shyam
-Jai Shree Bala ji
Good morning sir, ji 
-राम राम जी  जय श्री कृष्णा जय नारायणन  हरि विष्णु जी  ।गुरूजी को चरणस्पर्श पणाम स्वीकार हो  ।

Classification:- General, Sub_Classification:- Follow-up
-Please confirm me  Narayan sevasanthan
-Jabab do
-Please
-Batao
-40 din ho gya sir
-Sir kya huaa
-Please confirm by tomorrow morning.
-"Mehandi.rashma.ki.rashi..chaqe.se.santhan.ki.khata.me.jama.karna.he
-Khaya.no.bataye"
-बताना जी
-Bahut asha hai
-Pls talk to me my phone is silent i am on line

Classification:- General, Sub_Classification:- Emoji
-
-
-
-
-
-

Classification:- General, Sub_Classification:- Interested
-I'm interested

Classification:- General, Sub_Classification:- Thanks
-Thanks Sir
-आप का बहुत धन्यवाद सेवाओं की जानकारी देने के लिए 
-जी  धन्यवाद 
-Thanks
-"Apka bahut bahut dhanyavad
-"
-Thankyou
-धन्यवाद जी जय नारायण
-धन्यवाद महोदय
-Ram ram welcome Ganeshj
-Thankyou sir

Classification:- General, Sub_Classification:- Auto Reply
-Message received, will get back soon.
-Auto: Thank you for contacting us!
-We'll respond shortly, thank you.

Classification:- General, Sub_Classification:- Ok
-Ok ji
-Thik hai
-Okay, thanks

Classification:- General Information Enquiries, Sub_Classification:- About Sansthan
-What is th sansthan
-आप हमें जानकारी दीजिए
-hr@narayanseva.org
-Seva sansthan
-जानकारी दीजिये

Classification:- General Information Enquiries, Sub_Classification:- Katha Related
-Katha karna chahta hu
-नारायण सेवा संस्थान के माध्यम से यदि कोई आयोजन हो तो बताएगा।कथा के लिए
-Narayan Seva Sansthan se. Ham judna chahte hain. aur katha bhi karna chahte hain.  iske liye hamen aap sampurn jankari pradan Karen.
-Katha karne ke liye tyar hai
-कब का डेट हो सकता है कथा हमारी
-जय श्री राम नारायण सेवा संस्थान उदयपुर आपका हार्दिक अभिनंदन कभी आप एक मौका दीजिए कथा करने के लिए जय श्री राम

Classification:- General Information Enquiries, Sub_Classification:- Enquiry Visit Related
-We will arrive to UDAIPUR on 30 August in morning ,train arrival time is 8AM.Because we are coming first time to Santha so please confirm me.
-PNR:2339554510,TRN:20473,DOJ:29-08-25,SCH DEP:19:40,3A,DEE-UDZ,SANJAY KR GUPTA+1,B4 17 ,B4 20 ,Fare:002140,Please carry physical ticket. IR-CRIS
-"3 sal ki hai 
-Jila-kanpur nagar uttar pardesh 
-Kripya krke hme koi date de de angle month ki jisse ham wha tym se ake apke sanshthan me dikha sake"
-समय निकालकर सस्ता में भी आने की कोसिस करेंगे
-Ana kaha par h ye bata do aap
-UTR no 388309480581

Classification:- General Information Enquiries, Sub_Classification:- Divyang Vivah Couple
-अभी शादी कब है
-Send marriage programs card send

Classification:- General Information Enquiries, Sub_Classification:- Camp Related
-Shani amavasya ka kya hai
-Mumbai mein aapka camp kahan hai
-सर वितरण कब तक होगा

Classification:- General Information Enquiries, Sub_Classification:- Tax Related
-Kindly send my last year's donation recepit for income tax
-For getting 80G benefit for last year
-I need form for income tax filling for the donation I have done earliers

Classification:- General Information Enquiries, Sub_Classification:- School Enquiry Related
-Us time pese pareeksha aa gahi thi to me sekhne ke liye nahi aa para tha
-20.9.25ke bad muje computer course karna h

Classification:- General Information Enquiries, Sub_Classification:- Job Related
-आप के यहां विकलांगो को part time जॉब मिलेगा क्या

Classification:- General Information Enquiries, Sub_Classification:- Financial Help
-I whant money for work then I can have food for me.

Classification:- General Information Enquiries, Sub_Classification:- Program Impact
-How many people benefited from your camps last year?
-What impact did the food distribution program have?
-Can you share success stories of beneficiaries?
-How many divyang couples were supported this year?

Classification:- General Information Enquiries, Sub_Classification:- Annual Report Request
-Can I see last year's annual report?
-Please share the financial report for 2024.
-Is the annual report available online?
-Send me the transparency report for donations.

Classification:- General Information Enquiries, Sub_Classification:- Suggestion
-Can you start a vocational training program in Delhi?
-I suggest adding more camps in rural areas.
-Please consider online donation tracking for donors.

Classification:- General Information Enquiries, Sub_Classification:- Donation Purpose Related Information
-What will my donation be used for?
-How is the money spent in the hospital project?
-Please explain the purpose of the Amavasya campaign.

Classification:- General Information Enquiries, Sub_Classification:- Invitation Card Required
-Can you send an invitation for the upcoming event?
-Please share the card for Divyang Vivah program.
-Need invitation for the next camp in Udaipur.

Classification:- General Information Enquiries, Sub_Classification:- Woh Related
-What is the World of Humanity hospital project?
-When will the new hospital open?
-Tell me about Woh initiative.

Classification:- General Information Enquiries, Sub_Classification:- Event Related
-Any events planned for next month?
-What's the schedule for the annual function?
-Please share details of upcoming programs.

Classification:- General Information Enquiries, Sub_Classification:- Naturopathy Related
-Do you offer naturopathy treatments?
-Is there a naturopathy center at the sansthan?
-Details about natural therapy services?

Classification:- General Information Enquiries, Sub_Classification:- Orphanage Related Query
-How can I support the orphanage?
-Do you have facilities for orphan children?
-Tell me about your orphanage program.

Classification:- General Information Enquiries, Sub_Classification:- Management Contact Details Required
-Who is the head of the sansthan?
-Can I get contact details of the management team?
-Please share the email of the program coordinator.

Classification:- General Information Enquiries, Sub_Classification:- Ashram / Physiotherapy Information
-What are the facilities at your ashram?
-Can I visit the physiotherapy center?
-Details about physiotherapy services at sansthan?

Classification:- General Information Enquiries, Sub_Classification:- Transportation Help Required
-Can you arrange transport for my visit to Udaipur?
-Need help with travel to the camp, any support?
-Is transportation provided for beneficiaries?

Classification:- Medical / Treatment Enquiries, Sub_Classification:- Hospital Enquiry Related
-"I had gone under CARARACT surgery for my Right eye 2 times on 16th and on 23rd of this month due to some technicap problem in my R eye. Hence I am observing screen seeing of mobile, lap top and TV.
-Please don't send any messages for coming 10 days."
-Sir. Can I get him examined at the hospital on Sunday?
-हा, hospital का शुभारंभ कब होगा????
-Apke yaha dikhana hai
-Docter ko dikhaya hai to docter bole therapy kraao or dwa pilaooo nashe jo hai tight hai
-"NEW ADDRESS 
MURARI  LAL  AGRAWAL  
-1802 A WING MODI SPACES GANGES BUILDING  OPPOSITE  BHAGWATI  HOSPITAL  BORIVALI  WEST  MUMBAI  400103"

Classification:- Medical / Treatment Enquiries, Sub_Classification:- Artificial Limb Related
-Pair bana ki nahi sir
-मैं एक विकलांग व्यक्ति हूं मेरा पैर नहीं है
-Left leg m hai sir ...
-Sir mujhe one leg m polio h mujhe thik karana hai
-A person name Raja has lost his left arm in an accident  how he can get a artificial arm...
-Ujjain se चिमनगंज थाने के आगे वाली झुग्गी झोपड़ी मे रहते हैं और ये मेरा लड़का है जिसका पैर कट गया था एक्सीडेंट में और मे ठेला लगाती हूं फ्रूट का छोटा सा
" जय श्री महाँकाल
*त्रिलोकेशं नील15कण्ठं*
           *गंगाधरं सदाशिवम् ।*
*मृत्युञ्जयं महादेवं*
           *नमामि  तं  शंकरम् ।।*
भावार्थ: तीनों लोकों के स्वामी, नीलकण्ठ, गंगा को धारण करने वाले, हमेशा कल्याण करने वाले, मृत्यु पर विजय प्राप्त करने वाले, महादेव - शंकर जी की वंदना करता हूॅं।

 द्वादश ज्योतिर्लिंग में तीसरे उज्जैन स्थित दक्षिणमुखी स्वयम्भू बाबा महाँकाल का आज प्रातः 4 बजे प्रारम्भ भस्म आरती श्रंगार दर्शन - 25 अगस्त 2025 शिव प्रिय सोमवार"
"पंडित श्री संतोष शास्त्री अनपूर्णा गऊ शाला शिव शक्ति खाटू श्याम बाबा परिवार नर्मदा तट मंडला वाले।
-Mo. 9753020200,7999867569
-Janm se hi biklang hai

Classification:- Medical / Treatment Enquiries, Sub_Classification:- Aids & Appliances Related
-Do you provide hearing aids for the needy?
-Can I get a wheelchair for my mother?
-Need crutches for a patient, how to apply?

Classification:- Community Outreach Enquiries, Sub_Classification:- Awareness Program
-When is the next disability awareness event?
-Any campaigns planned for World Disability Day?
-Please share details of awareness programs in Delhi.
-What events are you holding for community outreach?
-Is there an awareness drive for rural areas?

Classification:- Community Outreach Enquiries, Sub_Classification:- Local Community Support
-What help is available in Mumbai for divyang?
-Any support programs in Jaipur for the poor?
-Can you assist with local community needs in Udaipur?
-What services are offered in rural Gujarat?
-Is there a community program in my area?

Classification:- Fundraising Campaign Enquiries, Sub_Classification:- Campaign Details
-Tell me about the Amavasya food drive.
-What is the goal of the current fundraising campaign?
-Details of the hospital fundraising event?
-When is the next fundraiser for children?
-How can I learn about your active campaigns?

Classification:- Fundraising Campaign Enquiries, Sub_Classification:- Sponsorship Interest
-Can my company sponsor a fundraiser?
-I want to support the next campaign, how to proceed?
-Is there an option to sponsor a food drive?
-How can I fund a specific campaign?
-Corporate sponsorship details for events?

Classification:- Beneficiary Support Enquiries, Sub_Classification:- Aid Application
-How do I apply for financial help?
-Can I get assistance for my child's education?
-What's the process for applying for aid?
-How to register as a beneficiary?
-Need help applying for medical support.

Classification:- Beneficiary Support Enquiries, Sub_Classification:- Beneficiary Status Update
-Is my application for aid approved?
-Status of my financial assistance request?
-When will I hear back about my application?
-Has my beneficiary form been processed?
-Update on my medical aid application?

Classification:- Ticket Related Enquiry, Sub_Classification:- KYC
-Need to update my PAN for donation records.
-Please link my Aadhaar to my donor profile.
-Can you update my KYC details for receipts?

Classification:- Ticket Related Enquiry, Sub_Classification:- Master Update
-My address has changed, please update it.
-Update my donor profile with new phone number.
-Please correct my name in the donor database.

Classification:- Ticket Related Enquiry, Sub_Classification:- Complaint Related
-The camp was not well-organized, please address.
-I faced issues with donation process, need help.
-Why was my receipt delayed? Please resolve.

Classification:- Ticket Related Enquiry, Sub_Classification:- Beneficiaries Detail Required
-Who benefited from my donation last month?
-Can I get a list of beneficiaries for my contribution?
-Please share details of people helped by my donation.

Classification:- Ticket Related Enquiry, Sub_Classification:- Amount Refund Related
-I paid twice by mistake, please refund ₹5000.
-Can you refund my extra donation of ₹2000?
-Accidentally sent ₹1000 extra, how to get refund?

Classification:- SPAM, Sub_Classification:- YouTube / Instagram Link
-https://www.youtube.com/watch?v=xyz
-Check my Instagram post: @randomuser
-Share this video: https://youtu.be/abc123
-Follow us on Insta for updates!
-See our latest reel on Instagram

Classification:- SPAM, Sub_Classification:- Spammy Message
-Hu
-Click here to win prizes
-Buy now, limited offer!
-Just saying hi
-Unrelated message with no context
"""

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
    prompt = f""" You are a sophisticated classification AI for Narayan Seva Sansthan.
    Your primary task is to analyze the user's input and return a single, valid JSON object with no extra text or explanations.

    The user's input might be a direct text message OR a detailed transcription of an image.
    When analyzing an image transcription, pay close attention to any text explicitly quoted from the image itself, as this represents the core user communication. Read the whole and classify based on the main element along with the tone of the image.

    Analyze the following input:
    - User Message: {message}
    - Here are some Few Shot Examples: {FEW_SHOT_EXAMPLES}

    Based on the following examples, classify the given message into one of these categories. Return a JSON object with the following schema:

    1. "classification": Choose the best fit from the list below, considering the conversation history for context.
        - Donation Related Enquiries
        - General
        - General Information Enquiries
        - Medical / Treatment Enquiries
        - Community Outreach Enquiries
        - Fundraising Campaign Enquiries
        - Beneficiary Support Enquiries
        - Spam
        - Ticket Related Enquiry

    2. "Sub_Classification": Based on the "classification", choose one from the relevant list along with explanation:
        Donation Related Enquiries, Announce Related, When a donor wants to make a donation, related announcements.
        Donation Related Enquiries, Post-Donation Related, When a donor shares the donation amount, details are required after deposit.
        Donation Related Enquiries, Amount Confirmation, To confirm whether the received amount is correctly recorded by the organization.
        Donation Related Enquiries, Donation Payment Information, Information required before a donor makes a donation.
        Donation Related Enquiries, KYC Update, After donation, KYC details are sent for updating receipts.
        Donation Related Enquiries, Receipts Related, Sending receipt details to donors after donation.
        Donation Related Enquiries, Send Sadhak Related, When a donor wants to send donation via a sadhak, including address details.
        Donation Related Enquiries, Property Donation, When a donor wants to donate property to the organization.
        Donation Related Enquiries, FD & Will Related, When a donor wants to donate FD or Will in the organization's name.
        Donation Related Enquiries, CSR Donation Interest, When a company or donor wants to make a CSR donation.
        Donation Related Enquiries, In-Kind Donation, When a donor wants to donate materials instead of money.
        Donation Related Enquiries, Recurring Donation, Setting up recurring or monthly donations.
        General, Emoji, When a number, image, or emoji is received unrelated to the organization.
        General, Follow-up, When a patient or donor follows up for confirmation on a previous message.
        General, Greeting, Greetings-related messages received.
        General, Interested, When someone expresses interest ("I'm interested") in the services or donation.
        General, Thanks, When someone sends thanks or welcome messages.
        General, Auto Reply, Automatic replies sent in response to broadcast messages.
        General, Ok
        General Information Enquiries, Suggestion, When a donor or patient shares suggestions regarding donations.
        General Information Enquiries, About Sansthan, Basic information about the organization.
        General Information Enquiries, Camp Related, When someone requests camp-related information.
        General Information Enquiries, Divyang Vivah Couple, When messages are sent regarding marriage of specially-abled couples.
        General Information Enquiries, Donation Purpose Related Information, When someone asks about the purpose of donations or events.
        General Information Enquiries, Enquiry Visit Related, When a donor inquires about visiting the organization or patient stay.
        General Information Enquiries, Financial Help, When someone asks for financial support from the organization.
        General Information Enquiries, Invitation Card Required, When an event invitation card is requested.
        General Information Enquiries, Job Related, When someone asks about job opportunities in the organization.
        General Information Enquiries, Katha Related, Messages related to requesting Katha information.
        General Information Enquiries, Woh Related, Information about the World of Humanity new hospital.
        General Information Enquiries, Event Related, When a patient or donor inquires about an event.
        General Information Enquiries, Tax Related, Messages regarding income tax and donations.
        General Information Enquiries, School Enquiry Related, Admission inquiries for NCA or affiliated schools.
        General Information Enquiries, Naturopathy Related, Messages regarding naturopathy services.
        General Information Enquiries, Orphanage Related Query, Messages about orphanage services or children care.
        General Information Enquiries, Management Contact Details Required, Messages requesting management contact details.
        General Information Enquiries, Ashram / Physiotherapy Information, Messages about ashram or physiotherapy center information.
        General Information Enquiries, Transportation Help Required, Messages requesting transportation help for donors or patients.
        General Information Enquiries, Program Impact, Questions about the impact or outcomes of programs.
        General Information Enquiries, Annual Report Request, Requests for annual reports or transparency data.
        Medical / Treatment Enquiries, Artificial Limb Related, Patient inquiries related to artificial limbs.
        Medical / Treatment Enquiries, Hospital Enquiry Related, Hospital-related information requests.
        Medical / Treatment Enquiries, Aids & Appliances Related, Requests related to medical aids and appliances.
        Community Outreach Enquiries, Awareness Program, Questions about campaigns to raise awareness.
        Community Outreach Enquiries, Local Community Support, Inquiries about local support programs.
        Fundraising Campaign Enquiries, Campaign Details, Questions about specific fundraising drives.
        Fundraising Campaign Enquiries, Sponsorship Interest, Interest in sponsoring a fundraising campaign.
        Beneficiary Support Enquiries, Aid Application, Requests to apply for aid or support.
        Beneficiary Support Enquiries, Beneficiary Status Update, Checking on aid or support application status.
        Ticket Related Enquiry, KYC, After receipt completion, updating donor KYC information.
        Ticket Related Enquiry, Master Update, Updating donor profile like address, name, number, email, etc.
        Ticket Related Enquiry, Receipts Related, When a donor receipt is created but donor has not received a hard copy.
        Ticket Related Enquiry, Complaint Related, Donor or patient complaints regarding services or donations.
        Ticket Related Enquiry, Beneficiaries Detail Required, When patient list is required after a donor's contribution.
        Ticket Related Enquiry, Physiotherapy Center Open, When a donor inquires about opening a physiotherapy center.
        Ticket Related Enquiry, Receipt Book Related, When a branch member requests receipt book details.
        Ticket Related Enquiry, Vocational Course Related, When a student requests information on vocational courses.
        Ticket Related Enquiry, Branch Membership Request, When a donor wants to become a branch member.
        Ticket Related Enquiry, Camp Related, When a donor wants to organize a camp through the organization.
        Ticket Related Enquiry, Katha Related, When someone wants to organize Katha through the organization.
        Ticket Related Enquiry, Amount Refund Related, Messages about double payments or refund requests.
        Ticket Related Enquiry, Sansthan Documents Request, Requests for organization-related documents for promotion purposes.
        Ticket Related Enquiry, Bhojan Seva Sponsorship, When a donor wants to sponsor Bhojan Seva.
        Ticket Related Enquiry, CSR Document Required, When a company requests CSR-related documents.
        Spam, Spammy Message, Messages unrelated to the organization, considered spam.
        Spam, YouTube / Instagram Link, Links shared that are not related to the organization.

    3. "Interested_To_Donate": Set to "yes" if the user shows clear intent or asks a direct question about donating (e.g., "I want to donate," "How can I donate?"). Otherwise, set to "no".

    4. "Question_Language": (ISO code: en, hi, etc.)
        - This is the language of the CORE message.
        - If the input is an image transcription, this MUST be the language of the text *from the image*.
        - If it's a direct message, it's the language of that message.
        eg- Jai narayan Narayan Seva Sansthan :- the Question Language is Hindi

    5. "Question_Script": (e.g., Latin, Devanagari, etc.)
        - This is the script of the CORE message.
        - Example: For "kaise ho", language is "hi" and script is "Latin". For "कैसे हो", language is "hi" and script is "Devanagari".

    eg for Question Language and Question Script, suppose the user Question is "Jai narayan Narayan Seva Sansthan" the Question Language is Hindi and Question Script is Latin
    eg for Question Language and Question Script, suppose the user Question is "मैं दान करना चाहता हूँ" the Question Language is Hindi and Question Script is Devanagari

    6. "confidence": Your confidence level in the classification: "HIGH", "MEDIUM", or "LOW".

    7. "reasoning": A brief, one-sentence explanation for the classification choices, explain why you think it should be under the defined classification and sub-classification.

    Add the classification and the subclassification in the same Dictionary variable classification with a pipe separator as given in the below format
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
        response = gemini_model.generate_content(prompt)
        result_text = response.text.strip().replace("```json\n", "").replace("\n```", "").strip()
        logger.debug(f"{request_id}:-Raw Gemini response: {result_text}")
        result = json.loads(result_text)
        logger.info(
            f"{request_id}:-Message classified as: {result.get('classification')} with confidence: {result.get('confidence')}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"{request_id}:-JSON parsing error in classification: {str(e)}, Raw response: {result_text}")
        return {"classification": "General|No_Module", "confidence": "LOW",
                "reasoning": f"JSON parsing error: {str(e)}", "Interested_To_Donate": "no",
                "Question_Language": "hi", "Question_Script": "Devanagari"}
    except Exception as e:
        logger.error(f"{request_id}:-Gemini classification error: {str(e)},{repr(e)}", exc_info=True)
        return {"classification": "General|No_Module", "confidence": "LOW",
                "reasoning": f"API error: {str(e)}", "Interested_To_Donate": "no",
                "Question_Language": "hi", "Question_Script": "Devanagari"}

# ----------------------------
# Updated Greeting response with formal structure
# ----------------------------
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
        return f"Respected {'Sir/Madam' if not user_name or user_name == 'Sevak' else user_name}, Jai Narayan! Thank you for contacting Narayan Seva Sansthan. We are happy to assist you. How may we help you today? With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

    try:
        prompt = f'''
You are a representative of Narayan Seva Sansthan. Create a formal greeting response.

USER INFORMATION:
- Name: {user_name if user_name and user_name != 'Sevak' else 'Sir/Madam'}
- Original Message: {original_message}
- Language: {Question_Language}
- Script: {Question_Script}

RESPONSE REQUIREMENTS:
1. Start with appropriate salutation (use name if available)
2. Include "Jai Narayan!" greeting
3. Acknowledge their message warmly
4. Offer assistance in a formal manner
5. Maintain respectful and professional tone
6. Structure with proper paragraphs
7. End with appropriate closing
8. Keep response concise (3-4 sentences)

IMPORTANT: Respond in the same language and script as the user's message.

Generate the formal greeting response:
'''

        response = gemini_model.generate_content(prompt)
        dynamic_response = response.text.strip()
        
        if len(dynamic_response) > 400 or not dynamic_response:
            raise Exception("Response too long or empty")
            
        return dynamic_response.replace("\n", " ").replace("  ", " ").strip()

    except Exception as e:
        logger.error(f"{request_id}:-Dynamic greeting generation failed: {e}")
        return f"Respected {'Sir/Madam' if not user_name or user_name == 'Sevak' else user_name}, Jai Narayan! Thank you for contacting Narayan Seva Sansthan. We are happy to assist you. How may we help you today? With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

# ----------------------------
# Updated Follow-Up response with formal structure
# ----------------------------
async def LLM_reply_follow_up(
        Question_Script,
        Question_Language,
        original_message: str,
        user_name: str,
        gemini_model,
        request_id):
    if not gemini_model:
        return f"Respected {'Sir/Madam' if not user_name or user_name == 'Sevak' else user_name}, Jai Narayan! Thank you for your follow-up. We appreciate your patience and will address your query shortly. With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

    try:
        prompt = f'''
You are a representative of Narayan Seva Sansthan. Create a formal follow-up response.

USER INFORMATION:
- Name: {user_name if user_name and user_name != 'Sevak' else 'Sir/Madam'}
- Follow-up Message: {original_message}
- Language: {Question_Language}
- Script: {Question_Script}

RESPONSE REQUIREMENTS:
1. Start with appropriate salutation
2. Include "Jai Narayan!" greeting  
3. Acknowledge their follow-up politely
4. Provide reassurance about their query
5. Offer specific assistance or timeline if possible
6. Maintain professional and caring tone
7. Structure with proper paragraphs
8. Keep response concise but helpful

Generate the formal follow-up response:
'''

        response = gemini_model.generate_content(prompt)
        dynamic_response = response.text.strip()
        
        if len(dynamic_response) > 400 or not dynamic_response:
            raise Exception("Response too long or empty")
            
        return dynamic_response.replace("\n", " ").replace("  ", " ").strip()

    except Exception as e:
        logger.error(f"{request_id}:-Dynamic follow_up generation failed: {e}")
        return f"Respected {'Sir/Madam' if not user_name or user_name == 'Sevak' else user_name}, Jai Narayan! Thank you for your follow-up. We appreciate your patience and will address your query shortly. With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

# ----------------------------
# Updated Ok response with formal structure
# ----------------------------
async def LLM_reply_ok(
        Question_Script,
        Question_Language,
        original_message: str,
        user_name: str,
        gemini_model,
        request_id):
    if not gemini_model:
        return f"Respected {'Sir/Madam' if not user_name or user_name == 'Sevak' else user_name}, Jai Narayan! Thank you for your confirmation. We are here to assist you further if needed. With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

    try:
        prompt = f'''
You are a representative of Narayan Seva Sansthan. Create a formal response to an "Ok" or confirmation message.

USER INFORMATION:
- Name: {user_name if user_name and user_name != 'Sevak' else 'Sir/Madam'}
- Message: {original_message}
- Language: {Question_Language}
- Script: {Question_Script}

RESPONSE REQUIREMENTS:
1. Start with appropriate salutation
2. Include "Jai Narayan!" greeting
3. Acknowledge their confirmation
4. Offer further assistance if needed
5. Maintain professional and polite tone
6. Structure with proper paragraphs
7. Keep response concise (2-3 sentences)

Generate the formal response:
'''

        response = gemini_model.generate_content(prompt)
        dynamic_response = response.text.strip()
        
        if len(dynamic_response) > 300 or not dynamic_response:
            raise Exception("Response too long or empty")
            
        return dynamic_response.replace("\n", " ").replace("  ", " ").strip()

    except Exception as e:
        logger.error(f"{request_id}:-Dynamic ok generation failed: {e}")
        return f"Respected {'Sir/Madam' if not user_name or user_name == 'Sevak' else user_name}, Jai Narayan! Thank you for your confirmation. We are here to assist you further if needed. With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

# ----------------------------
# NEW: Receipt Response (Donation Related Enquiries|Receipts Related)
# ----------------------------
async def generate_receipt_response(
        message_text: str,
        user_name: str,
        question_language: str,
        question_script: str,
        gemini_model,
        request_id
) -> str:
    if not gemini_model:
        return """Respected Sir/ Ma'am, Jai Narayan! Thank you for your generous donation to Narayan Seva Sansthan. Attaching herewith the receipt for your reference. Kindly let us know if you require a hard copy as well. 🙏,""".replace("\n", " ").replace("  ", " ").strip()

    try:
        prompt = f'''
You are a representative of Narayan Seva Sansthan. Generate a receipt acknowledgment.

USER INFO:
- Name: {user_name}
- Message: {message_text}
- Language/Script: {question_language}/{question_script}

EXTRACT from message: amount (e.g., ₹5100), date (DD/MM/YYYY), donor_name (full name).

RESPONSE RULES (Match user's language/script):
- Generic: "Respected Sir/Ma'am, Jai Narayan! Thank you... Attaching receipt... Hard copy? 🙏"
- Personalized: "Dear [FULL NAME], Thank you for ₹[AMOUNT].00. Date: [DATE]. Receipt soon... With gratitude, Narayan Seva Sansthan (Online Mode)"

EXAMPLES:
- "Today transferred Rs 5100" -> "Dear Luxmi Diwadi, Thank you for ₹5100.00. Date: 23/09/2025..."
- "Send receipt" -> Generic
- "Thank you... await receipt for Darshana Pandya" -> Generic (no amount/date)

Generate exact response:
'''
        response = gemini_model.generate_content(prompt)
        ai_response = response.text.strip()
        if len(ai_response) > 500 or not ai_response:
            raise Exception("Invalid response")
        return ai_response.replace("\n", " ").replace("  ", " ").strip()

    except Exception as e:
        logger.error(f"{request_id}:-Receipt response failed: {e}")
        return """Respected Sir/ Ma'am, Jai Narayan! Thank you for your generous donation to Narayan Seva Sansthan. Attaching herewith the receipt for your reference. Kindly let us know if you require a hard copy as well. 🙏,""".replace("\n", " ").replace("  ", " ").strip()

# ----------------------------
# NEW: Amount Confirmation Response (Donation Related Enquiries|Amount Confirmation)
# ----------------------------
async def generate_amount_confirmation_response(
        message_text: str,
        user_name: str,
        question_language: str,
        question_script: str,
        gemini_model,
        request_id
) -> str:
    if not gemini_model:
        if question_language == "hi":
            return """आदरणीय CP GUPTA JI,  जय नारायण ! आपने ₹3,000 की सहायता राशि हेतु जो pay-in slip भरी है, उसमें अकाउंट होल्डर का नाम “Narayan Seva Sansthan” होना चाहिए। कृपया इसे ध्यान में रखते हुए सही नाम से ट्रांजैक्शन करें। धन्यवाद।""".replace("\n", " ").replace("  ", " ").strip()
        return """Respected Umi Ji, Jai Narayan! Thank you very much for your generous contribution. It will be put to the best use to help those in need. We kindly request you to share the transaction/reference number for confirmation of the amount.""".replace("\n", " ").replace("  ", " ").strip()

    try:
        prompt = f'''
Generate amount confirmation response.

USER INFO: Name: {user_name}, Message: {message_text}, Lang/Script: {question_language}/{question_script}

RULES: Match examples exactly. Hindi for Hindi queries.
- "I have put money... Nat West" -> English request for txn ID
- "Is name se paisa nhi jaha hai" -> Hindi name fix
- "कन्या भोजन..." -> Hindi thanks for specific purpose

Generate:
'''
        response = gemini_model.generate_content(prompt)
        ai_response = response.text.strip()
        return ai_response.replace("\n", " ").replace("  ", " ").strip() if ai_response else "Fallback generic"

    except Exception as e:
        logger.error(f"{request_id}:-Amount confirmation failed: {e}")
        # Fallback based on language
        if question_language == "hi":
            return """आदरणीय CP GUPTA JI,  जय नारायण ! आपने ₹3,000 की सहायता राशि हेतु जो pay-in slip भरी है, उसमें अकाउंट होल्डर का नाम “Narayan Seva Sansthan” होना चाहिए। कृपया इसे ध्यान में रखते हुए सही नाम से ट्रांजैक्शन करें। धन्यवाद।""".replace("\n", " ").replace("  ", " ").strip()
        return """Respected Umi Ji, Jai Narayan! Thank you very much for your generous contribution. It will be put to the best use to help those in need. We kindly request you to share the transaction/reference number for confirmation of the amount.""".replace("\n", " ").replace("  ", " ").strip()

# ----------------------------
# NEW: Post-Donation Response (Donation Related Enquiries|Post-Donation Related)
# ----------------------------
async def generate_post_donation_response(
        message_text: str,
        user_name: str,
        question_language: str,
        question_script: str,
        gemini_model,
        request_id
) -> str:
    if not gemini_model:
        return """Respected  Latha ji Jai Narayan! We are deeply saddened to hear about the passing of your son. Our heartfelt condolences to you and your family in this difficult time. We truly appreciate your support even in such a moment of grief. Please be assured that we will carry out the necessary process as per your request. May God give you strength and peace.""".replace("\n", " ").replace("  ", " ").strip()

    try:
        prompt = f'''
Generate condolence post-donation response.

USER: {user_name}, Message: {message_text}, Lang: {question_language}

Match example: "He is no more... depositing" -> Condolence + assurance.

Generate:
'''
        response = gemini_model.generate_content(prompt)
        return response.text.strip().replace("\n", " ").replace("  ", " ").strip()

    except Exception as e:
        logger.error(f"{request_id}:-Post-donation failed: {e}")
        return """Respected Latha ji Jai Narayan! We are deeply saddened to hear about the passing of your son. Our heartfelt condolences to you and your family in this difficult time. We truly appreciate your support even in such a moment of grief. Please be assured that we will carry out the necessary process as per your request. May God give you strength and peace.""".replace("\n", " ").replace("  ", " ").strip()

supabase: Client = None
gemini_model = None
numbered_content = {}
keywords_summary = {}

# ----------------------------
# Lifespan handler
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, gemini_model, numbered_content, keywords_summary
    logger.info("Starting FastAPI app on port {}", os.getenv('PORT', 10000))

    try:
        supabase = get_supabase_client()
        logger.info("Supabase client setup completed successfully")
    except Exception as e:
        logger.error("Supabase connection failed: {}", e)

    try:
        gemini_model = get_gemini_client()
        logger.info("Gemini AI client setup completed successfully")
    except Exception as e:
        logger.error("Gemini AI initialization failed: {}", e)

    numbered_content, keywords_summary = fetch_numbered_data()
    if not numbered_content:
        logger.warning("FAQ content could not be loaded. FAQ chatbot functionality will be limited.")

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
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetime_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_recursive(item) for item in obj]
    return obj

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
async def log_to_supabase(log_data: dict, request_id, table: str = "message_logs"):
    try:
        if supabase:
            serialized_data = serialize_datetime_recursive(log_data.copy())
            result = supabase.table(table).insert(serialized_data).execute()
            logger.info(f"{request_id}:-Successfully logged to Supabase table '{table}' (inserted {len(serialized_data)} fields)")
        else:
            logger.error(f"{request_id}:-Supabase client not initialized, skipping log")
    except Exception as e:
        logger.error(f"{request_id}:-Supabase log failed: {e}")

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

# ----------------------------
# /message endpoint with AI classification and FAQ logic
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

    await log_to_supabase(log_data, request_id)

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
                                                  "Respected Sir/Madam, Jai Narayan! Thank you for your donation! We will process it shortly and send you the receipt. With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip())
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
    main_classification, sub_classification = classification.split("|") if "|" in classification else (classification, "No_Module")

    if main_classification == "Spam" and sub_classification == "Spammy Message":
        ai_response = """Jai Narayan Thank you for your warm wishes. Your blessings and support inspire us to continue serving differently-abled brothers and sisters with love and care  """.replace("\n", " ").replace("  ", " ").strip()

    elif main_classification == "General":
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
            ai_response = f"Respected {'Sir/Madam' if not user_name or user_name == 'Sevak' else user_name}, Jai Narayan! Thank you for contacting Narayan Seva Sansthan. How may we assist you today? With regards, Narayan Seva Sansthan".replace("\n", " ").replace("  ", " ").strip()

    elif main_classification == "Donation Related Enquiries":
        if sub_classification == "Receipts Related":
            ai_response = await generate_receipt_response(
                message_text, user_name, question_language, question_script, gemini_model, request_id
            )
        elif sub_classification == "Amount Confirmation":
            ai_response = await generate_amount_confirmation_response(
                message_text, user_name, question_language, question_script, gemini_model, request_id
            )
        elif sub_classification == "Announce Related":
            ai_response = await generate_donation_response(user_name, gemini_model, request_id)
        elif sub_classification == "Post-Donation Related":
            ai_response = await generate_post_donation_response(
                message_text, user_name, question_language, question_script, gemini_model, request_id
            )
        else:
            # Existing: interested_to_donate check + FAQ
            if interested_to_donate == "yes" or "donation" in message_text.lower():
                ai_response = await generate_donation_response(user_name, gemini_model, request_id)
            else:
                selected_content_num = llm_select_best_content(message_text, keywords_summary, gemini_model, request_id)
                selected_content = numbered_content.get(selected_content_num, "No relevant content found.")
                ai_response = generate_faq_response(selected_content, message_text, gemini_model, request_id)

    else:
        # Existing FAQ for all other categories
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
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="WhatsApp Message Processor",
        version="1.0.0"
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
