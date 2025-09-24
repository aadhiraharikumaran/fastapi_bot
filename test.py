```python
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from loguru import logger
import sys
import sqlite3
import json
import supabase
from dotenv import load_dotenv
import google.generativeai as genai
import httpx
import re
import asyncio
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure logger
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

# Supabase client
def get_supabase_client() -> supabase.Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY) must be set")
        raise ValueError("Missing Supabase configuration")
    if os.getenv("SUPABASE_SERVICE_KEY"):
        logger.info("Using Supabase service key (bypasses RLS)")
    else:
        logger.warning("Using anon key - ensure RLS policies allow inserts")
    return supabase.create_client(url, key)

# Gemini AI client
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")  # Updated to match provided code
        logger.info("Gemini client and model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Gemini AI initialization failed: {e}")
        return None

# FAQ database fetch
def fetch_numbered_data():
    db_path = "extracted_data.db"
    if not os.path.exists(db_path):
        logger.error(f"FAQ Database file not found at '{db_path}'. Returning default FAQs.")
        return {1: "Default FAQ content: Please contact support for details."}, {1: "default keywords"}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT rowid, keywords, content FROM extracted_data")
        rows = cursor.fetchall()
        conn.close()
        numbered_data = {row[0]: row[2] for row in rows}
        numbered_keywords = {row[0]: row[1] for row in rows}
        logger.success(f"Loaded {len(numbered_data)} numbered FAQ content sections")
        return numbered_data, numbered_keywords
    except Exception as e:
        logger.error(f"Failed to fetch FAQ data: {e}")
        return {1: "Default FAQ content"}, {1: "default keywords"}

# LLM FAQ selection
def llm_select_best_content(query, keywords_summary, gemini_model, request_id):
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
        logger.error(f"{request_id}:-LLM FAQ selection error: {e}")
        return 1

# Generate FAQ response
def generate_faq_response(content, question, gemini_model, request_id):
    logger.info(f"{request_id}:-Generating FAQ response")
    if not gemini_model:
        return "Sorry, our FAQ service is temporarily unavailable."
    try:
        prompt = f"""
You are a humble Sevak of Narayan Seva Sansthan.
Answer the following question kindly and devotionally, using this content:
Content: {content}
Question: {question}
Provide a short, sweet, and crisp answer in plain text (max 3-4 sentences).
Do not add extra commentary.
"""
        response = gemini_model.generate_content(prompt)
        faq_answer = response.text.strip()
        logger.success(f"{request_id}:-Final response generated ({len(faq_answer)} chars)")
        return faq_answer
    except Exception as e:
        logger.error(f"{request_id}:-FAQ response generation error: {e}")
        return f"Sorry, could not generate an answer at the moment: {str(e)}"

# Image analysis
async def analyze_image_with_gemini(image_url: str, gemini_model, request_id) -> dict:
    if not gemini_model:
        return {"transcription": "", "status": "error", "error": "Gemini client not available"}
    try:
        logger.info(f"{request_id}:-Analyzing image from URL: {image_url}")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
        mime_type = resp.headers.get('content-type', 'image/jpeg').split(';')[0]
        image_part = {"mime_type": mime_type, "data": resp.content}
        response = gemini_model.generate_content(["Explain what is in this image clearly and in detail.", image_part])
        transcription = response.text.strip()
        logger.info(f"{request_id}:-Image analysis completed. Transcription length: {len(transcription)}")
        return {"transcription": transcription, "status": "success", "error": None}
    except Exception as e:
        logger.error(f"{request_id}:-Image analysis error: {e}")
        return {"transcription": "", "status": "error", "error": str(e)}

# Donation processing
async def process_donation_transcript(transcript: str, user_name: str, gemini_model, request_id) -> dict:
    if not transcript or not gemini_model:
        return {"is_donation_screenshot": False, "extraction_details": {}, "generated_response": None}
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
            cleaned_details = {k: v for k, v in extraction_details.items() if v is not None and v != "null" and str(v).strip()}
            extraction_details = cleaned_details
        if result.get("is_donation_screenshot") and 'detected_language' not in extraction_details:
            extraction_details['detected_language'] = 'hindi'
        logger.info(f"{request_id}:-Donation analysis result: {result.get('is_donation_screenshot')}, Details: {extraction_details}")
        return {
            "is_donation_screenshot": result.get("is_donation_screenshot", False),
            "extraction_details": extraction_details,
            "generated_response": result.get("generated_response", None)
        }
    except Exception as e:
        logger.error(f"{request_id}:-Error in donation processing: {e}")
        return {"is_donation_screenshot": False, "extraction_details": {}, "generated_response": None}

# Few-shot examples
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
-Subject: Request for Acknowledgement Receipts – July & August 2025
Dear Sir/Ma’am,
I have not yet received the acknowledgement receipts for the months of July 2025 and August 2025. May I kindly request you to share the same at the earliest.
Your support in this matter will be highly appreciated.
Thanks & regards,
Nilesh Bhagat 🙏🏻"
-Receipt plz ??
-Rasid. Sanjeev Kumar
-"PLEASE SEND ME RECEIPT ON WHATSAPP
-NO NEED TO SEND BY POST"

Classification:- Donation Related Enquiries, Sub_Classification:- Amount Confirmation
-Hospital के लिए 100000की सेवा सहयोग भी भेजा था 🙏
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
-Finally I succeeded today in transferring ₹2000/- to the sanstha a/c for feeding children on amavasya.🙏

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
-अन्यथा Pan नम्बर बदलने का कष्ट करें 
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
-Om Gajananaya namah. Om Mitraye namah. Radhe Radhe. Jai Sada Shiv. Jai Sarvamangala Mata. Jai Ragya Mata. Jai Bhadrakaali Mata. Jai Sharada Mata. Jai Annapurna Mata. Jai Sheetla Mata. Jai Bhoomi Mata. Jai Mangalmurti Hanuman. Om Swami Nathishoraye namah. Guru kripa. Mangalamay Mangalvaar. Orzu🙏🙏🙏
"🙏🌼🌼 *जय श्री राधाकृष्ण*🌼🌼🙏
  🙏🌺🌺 *श्रीकृष्णावतार*🌺🌺🙏
-*फिर भगवान् से मांगने की बजाये निकटता बनाओ तो सब कुछ अपने आप मिलना शुरू हो जायेगा ।*
-*🌷 जय श्री गणेश जी जय श्री कृष्ण 🌷*शुभ रात्रि  जय सियाराम"
-Jay Shri Ram
-Ram ram ji
-राधे राधे 🏵️🙏
-Gud Nyt Yu Nd Yr Family Members
-Jai naryana
-जय श्री श्याम जी 🙏
-Hi
-🙏 OK   Jay shree Radhey Krishna
-"-🙏🌹🥭🪔🥥🕉️🇮🇳🙇‍♂️
-Namah vishnu in service of needy"
-Jai Narayan 👏
-जय नारायण 🙏🙏
-🙏🙏🙏 राधे राधे जय श्री कृष्ण
-हम आपके संस्थान से 2010से जुड़े हैं 🙏
-Ram ramji
-"जय नारायण 
-Jay shree Krishna 🙏
-Jai jai shree shyam 🙏🌹🙌
-Jai Shree Shyam
-Jai Shree Bala ji
Good morning sir, ji 🙏🌹
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
-🙏🏻
-🌹🙏🌹
-👏👏👏
-👍👍
-❤️❤️
-👏

Classification:- General, Sub_Classification:- Interested
-I'm interested

Classification:- General, Sub_Classification:- Thanks
-Thanks Sir
-आप का बहुत धन्यवाद सेवाओं की जानकारी देने के लिए 🙏
-जी 💐 धन्यवाद 👏🏻
-Thanks
-"Apka bahut bahut dhanyavad
-🙏🙏🙏"
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
-Can I see last year’s annual report?
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
-What’s the schedule for the annual function?
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
"🌈🎺🎊🥁जय श्री महाँकाल
*त्रिलोकेशं नीलकण्ठं*
           *गंगाधरं सदाशिवम् ।*
*मृत्युञ्जयं महादेवं*
           *नमामि  तं  शंकरम् ।।*
🌈भावार्थ: तीनों लोकों के स्वामी, नीलकण्ठ, गंगा को धारण करने वाले, हमेशा कल्याण करने वाले, मृत्यु पर विजय प्राप्त करने वाले, महादेव - शंकर जी की वंदना करता हूॅं।

🥁🎊🎺🌈द्वादश ज्योतिर्लिंग में तीसरे उज्जैन स्थित दक्षिणमुखी स्वयम्भू बाबा महाँकाल का आज प्रातः 4 बजे प्रारम्भ भस्म आरती श्रंगार दर्शन - 25 अगस्त 2025 शिव प्रिय सोमवार"
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
-Can I get assistance for my child’s education?
-What’s the process for applying for aid?
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

# Classification function
def classify_message_with_gemini(message: str, gemini_model, request_id) -> dict:
    if not gemini_model:
        logger.error(f"{request_id}:-Gemini model not initialized, returning default classification")
        return {
            "classification": "General|No_Module",
            "confidence": "LOW",
            "reasoning": "Gemini client not available",
            "Interested_To_Donate": "no",
            "Question_Language": "hi",
            "Question_Script": "Devanagari"
        }
    if not message or not message.strip():
        logger.warning(f"{request_id}:-Empty message received, returning default classification")
        return {
            "classification": "General|No_Module",
            "confidence": "MEDIUM",
            "reasoning": "Empty or whitespace message",
            "Interested_To_Donate": "no",
            "Question_Language": "hi",
            "Question_Script": "Devanagari"
        }
    prompt = f"""
You are a sophisticated classification AI for Narayan Seva Sansthan.
Your primary task is to analyze the user's input and return a single, valid JSON object with no extra text or explanations.
The user's input might be a direct text message OR a detailed transcription of an image.
When analyzing an image transcription, pay close attention to any text explicitly quoted from the image itself, as this represents the core user communication. Read the whole and classify based on the main element along with the tone of the image.
Analyze the following input:
- User Message: {message}
- Here are some Few Shot Examples: {FEW_SHOT_EXAMPLES}
Based on the following examples, classify the given message into one of these categories. Return a JSON object with the following schema:
1. "Classification": Choose the best fit from the list below, considering the conversation history for context.
    - Donation Related Enquiries
    - General
    - General Information Enquiries
    - Medical / Treatment Enquiries
    - Community Outreach Enquiries
    - Fundraising Campaign Enquiries
    - Beneficiary Support Enquiries
    - Spam
    - Ticket Related Enquiry
2. "Sub_Classification": Based on the "Classification", choose one from the relevant list along with explanation:
    Donation Related Enquiries, Announce Related, When a donor wants to make a donation, related announcements.
    Donation Related Enquiries, Post-Donation Related, When a donor shares the donation amount, details are required after deposit.
    Donation Related Enquiries, Amount Confirmation, To confirm whether the received amount is correctly recorded by the organization.
    Donation Related Enquiries, Donation Payment Information, Information required before a donor makes a donation.
    Donation Related Enquiries, KYC Update, After donation, KYC details are sent for updating receipts.
    Donation Related Enquiries, Receipts Related, Sending receipt details to donors after donation.
    Donation Related Enquiries, Send Sadhak Related, When a donor wants to send donation via a sadhak, including address details.
    Donation Related Enquiries, Property Donation, When a donor wants to donate property to the organization.
    Donation Related Enquiries, FD & Will Related, When a donor wants to donate FD or Will in the organization’s name.
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
    Ticket Related Enquiry, Beneficiaries Detail Required, When patient list is required after a donor’s contribution.
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
7. "reasoning": A brief, one-sentence explanation for your classification choices, explain why you think it should be under the defined classification and sub-classification.
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
        logger.info(f"{request_id}:-Message classified as: {result.get('classification')} with confidence: {result.get('confidence')}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"{request_id}:-JSON parsing error in classification: {str(e)}, Raw response: {result_text}")
        return {
            "classification": "General|No_Module",
            "confidence": "LOW",
            "reasoning": f"JSON parsing error: {str(e)}",
            "Interested_To_Donate": "no",
            "Question_Language": "hi",
            "Question_Script": "Devanagari"
        }
    except Exception as e:
        logger.error(f"{request_id}:-Gemini classification error: {str(e)}")
        return {
            "classification": "General|No_Module",
            "confidence": "LOW",
            "reasoning": f"API error: {str(e)}",
            "Interested_To_Donate": "no",
            "Question_Language": "hi",
            "Question_Script": "Devanagari"
        }

# Greeting response
async def LLM_reply_greeting(Question_Script, Question_Language, original_message: str, user_name: str, gemini_model, WA_Msg_Type, request_id) -> str:
    if not gemini_model:
        return f"🙏 Jai Shree Narayan {user_name}! Narayan Seva Sansthan se sampark karne ke liye dhanyawad, mai apki kaise sahayta kar sakti hu?"
    try:
        if WA_Msg_Type and WA_Msg_Type.lower() == "image":
            prompt = f"""
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
Name: {user_name}
Question's Language: {Question_Language}
Question's Script: {Question_Script}
User's Question: {original_message}
"""
            response = gemini_model.generate_content(prompt)
            dynamic_response = response.text.strip()
            if len(dynamic_response) > 300 or not dynamic_response:
                raise Exception("Response too long or empty")
            return dynamic_response
        else:
            prompt = f"""
You are Narayan Seva Sansthan's Assistant.
Your name is Priya, introduce yourself along with the greeting.
Your role in this step is ONLY to generate a greeting reply.
You can be a bit creative, show that you are very happy to receive message from them, don't literally say it
Ask how can I help you
Rules:
1. Understand the previous chat history, use if required otherwise stick to your task for being a Greetings bot
3. If the user_name is recognized (old user) and their name is provided, always include their name naturally in the greeting.
    - Example: "Hello Ramesh ji! How are you today?"
    - Example: "नमस्ते सीमा जी! आप कैसी हैं?"
4. Always respond in the same combination of "Question's_language" and "Questions_script" as detected.
    - If language = "hi" and script = "Latin" → Hindi in Latin script.
    - If language = "hi" and script = "Devanagari" → Hindi in Devanagari script.
    - If language = "en" and script = "Latin" → English in Latin script.
5. Do not translate into any other language or script. Always mirror the detected language + script.
6. Do not add any explanations, tags, or formatting. Only output the greeting message.
eg if the user says "Radhe radhe" you too should reply with " Radhe Radhe Rahul ji, ..... ?"
eg if the user says "Hello" you should reply with "Hi Rahul ji,...."
eg if the user says "जय श्री कृष्णा" you should reply "जय श्री कृष्णा साहिल जी,.....
7. Use Emojis whenever necessary
Name: {user_name}
Question's Language: {Question_Language}
Question's Script: {Question_Script}
User's Question: {original_message}
"""
            response = gemini_model.generate_content(prompt)
            dynamic_response = response.text.strip()
            if len(dynamic_response) > 300 or not dynamic_response:
                raise Exception("Response too long or empty")
            return dynamic_response
    except Exception as e:
        logger.error(f"{request_id}:-Dynamic greeting generation failed: {e}")
        return f"🙏 Jai Shree Narayan {user_name}! Narayan Seva Sansthan se sampark karne ke liye dhanyawad, mai apki kaise sahayta kar sakti hu?"

# Follow-up response
async def LLM_reply_follow_up(Question_Script, Question_Language, original_message: str, user_name: str, gemini_model, request_id):
    if not gemini_model:
        return f"Jai Shree Narayan {user_name}!, hum aapki baat samajh rahe hain, aapki maang jald puri karne ki koshish karenge 🙏\nAdhik jankari ke liye iss number par sampark kijiye: +91-294 66 22 222\ndhanyawad🙏"
    try:
        prompt = f"""
You are Priya, Narayan Seva Sansthan's friendly assistant.
Your only role here is to generate a reply to FOLLOW UP related MESSAGE.
Be natural, positive, and show happiness in receiving the message.
Rules:
1. Firstly read the chat History and try to identify with what respect is the user asking follow-up about and then based on that design the reply to follow-up message.
2. The language should be compassionate and avoid being overly formal or robotic
3. If user_name is provided, always include it naturally with "ji".
4. Mirror the detected Question's Language and Script exactly.
    - If language = "hi" and script = "Latin" → Hindi in Latin script.
    - If language = "hi" and script = "Devanagari" → Hindi in Devanagari script.
    - If language = "en" and script = "Latin" → English in Latin script.
eg. "if user says "jawab do please" and earlier he asked about the donation receipt, firstly since the question language is hindi and script is english "Ji Sahil Ji, so your reply should somewhat be like(whatever the username is), shama chahta hu, apki donation receipt ko hum jald sei jald aap tak pohochane ki koshish krenge.\nDeri ke liye maafi chahte hai.🙏"
3. The message must reference the specific context of the last communication.
4. Provide assurance that their request will be completed as soon as possible.
5. Give them a positive reply
6. not to long not to short, sweet and simple
7. Be apologetic.
8. Do not generate a response greater than 300 characters
Name: {user_name}
Question's Language: {Question_Language}
Question's Script: {Question_Script}
User's Input: {original_message}
"""
        response = gemini_model.generate_content(prompt)
        dynamic_response = response.text.strip()
        if len(dynamic_response) > 300 or not dynamic_response:
            raise Exception("Response too long or empty")
        return dynamic_response
    except Exception as e:
        logger.error(f"{request_id}:-Dynamic follow_up generation failed: {e}")
        return f"Jai Shree Narayan {user_name}!, hum aapki baat samajh rahe hain, aapki maang jald puri karne ki koshish karenge 🙏\nAdhik jankari ke liye iss number par sampark kijiye: +91-294 66 22 222\ndhanyawad🙏"

# Ok response
async def LLM_reply_ok(Question_Script, Question_Language, original_message: str, user_name: str, gemini_model, request_id):
    if not gemini_model:
        return f"Thik hai {user_name} ji, Narayan Seva Sansthan aapke sahayta ke liye hamesha hai, dhanyawad 🙏"
    try:
        prompt = f"""
You are Priya, Narayan Seva Sansthan's friendly assistant.
Your only role here is to generate a reply to Ok related MESSAGE.
Be natural, positive, and show happiness in receiving the message.
Rules:
1. Firstly read the chat History and try to identify with what respect is the user is saying Ok about and then based on that design the reply to Ok related message.
2. The language should be compassionate and avoid being overly formal or robotic
3. Mirror the detected Question's Language and Script exactly.
4. If user asked a question you replied to it and then if the user replied with ok, then first reply reply to his okay and then ask a follow up question if necessary.
5. If he asked anything about donation and then we gave him the information, reply to that okay and then try to persuade him about the donation.
6. Be Natural while replying to ok. if you feel there's nothing to add up end the conversation with lets say "happy to help" or something like that.
    - If language = "hi" and script = "Latin" → Hindi in Latin script.
    - If language = "hi" and script = "Devanagari" → Hindi in Devanagari script.
    - If language = "en" and script = "Latin" → English in Latin script.
eg If the user says "Ji" which is kind of ok/yes in hindi then read the earlier context, check to what he has said ok, and reply accordingly. lets say we asked him is he willing to donate to 2 tricycle which costs 7000, and for that he said "Ji" my reply could be like "Thikey, Sahil Ji! Jab aap aage ki donation process complete kar lein, toh please humein batayega. Ek aur cheez jo hum suggest karna chahenge: agar aap isi amount mein ₹2,000 aur add karein, toh aap ek nahin, do logon ki help kar payenge!\n Hum aapke sahayta ke liye tatpar hai".
7. Be natural and generate a reply which suits the conversation.
8. Do not generate a response greater than 300 characters
Name: {user_name}
Question's Language: {Question_Language}
Question's Script: {Question_Script}
User's Input: {original_message}
"""
        response = gemini_model.generate_content(prompt)
        dynamic_response = response.text.strip()
        if len(dynamic_response) > 300 or not dynamic_response:
            raise Exception("Response too long or empty")
        return dynamic_response
    except Exception as e:
        logger.error(f"{request_id}:-Dynamic ok generation failed: {e}")
        return f"Thik hai {user_name} ji, Narayan Seva Sansthan aapke sahayta ke liye hamesha hai, dhanyawad 🙏"

# Supabase logging
async def log_to_supabase(log_data: dict, request_id: str, table: str = "message_logs", update_mode: bool = False):
    try:
        if not supabase:
            logger.error(f"{request_id}:-Supabase client not available")
            return
        serialized_data = serialize_datetime_recursive(log_data.copy())
        if update_mode:
            result = supabase.table(table).update(serialized_data).eq("request_id", request_id).execute()
            logger.debug(f"{request_id}:-Successfully updated Supabase log entry")
        else:
            result = supabase.table(table).insert(serialized_data).execute()
            logger.debug(f"{request_id}:-Successfully inserted Supabase log entry")
        return result
    except Exception as e:
        logger.error(f"{request_id}:-Supabase log failed: {e}")
        return None

# Serialize datetime objects
def serialize_datetime_recursive(obj):
    if isinstance(obj, dict):
        return {k: serialize_datetime_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_recursive(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

# Forward to replica system
async def forward_message_to_replica(payload: dict, request_id):
    replica_url = "https://nss-code-replica.onrender.com/message"
    try:
        safe_payload = serialize_datetime_recursive(payload)
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(replica_url, json=safe_payload)
            logger.info(f"{request_id}:-Forwarded message to replica. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"{request_id}:-Failed to forward message to replica: {e}")

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, gemini_model, numbered_content, keywords_summary
    logger.info("Starting FastAPI app on port {}", os.getenv('PORT', 10000))
    try:
        supabase = get_supabase_client()
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
    try:
        gemini_model = get_gemini_client()
    except Exception as e:
        logger.error(f"Gemini AI initialization failed: {e}")
    numbered_content, keywords_summary = fetch_numbered_data()
    if not numbered_content:
        logger.warning("FAQ content could not be loaded. FAQ chatbot functionality will be limited.")
    yield
    logger.info("Application shutdown complete")

# FastAPI app
app = FastAPI(
    title="WhatsApp Message Processor with AI Classification",
    description="WhatsApp message processing service with AI classification and Supabase logging",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
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
    Status: Optional[str] = None
    Donor_Name: Optional[str] = None

class MessageResponse(BaseModel):
    phone_number: str
    ai_response: str
    ai_reason: str
    WA_Auto_Id: Optional[int] = None
    WA_Message_Id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str

# Health endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="WhatsApp Message Processor",
        version="1.0.0"
    )

# Message processing endpoint
@app.post("/message", response_model=MessageResponse, tags=["Message Processing"])
async def handle_message(request: MessageRequest):
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    phone_number = request.MobileNo or request.WA_Msg_To or "Unknown"
    user_name = request.Donor_Name or request.Wa_Name or "Sevak"
    wa_msg_type = request.WA_Msg_Type.lower() if request.WA_Msg_Type else "text"

    # Initial log data
    initial_log_data = {
        "request_id": request_id,
        "endpoint": "/message",
        "method": "POST",
        "status": "processing",
        "processing_start_time": start_time.isoformat(),
        "raw_request": request.model_dump(exclude_none=True),
        "wa_auto_id": request.WA_Auto_Id,
        "wa_in_out": request.WA_In_Out,
        "account_code": request.Account_Code,
        "wa_received_at": request.WA_Received_At.isoformat() if request.WA_Received_At else None,
        "ng_code": request.NGCode,
        "wa_name": request.Wa_Name,
        "mobile_no": phone_number,
        "wa_msg_to": request.WA_Msg_To,
        "wa_msg_text": request.WA_Msg_Text,
        "wa_msg_type": wa_msg_type,
        "integration_type": request.Integration_Type,
        "wa_message_id": request.WA_Message_Id,
        "wa_url": request.WA_Url,
        "donor_name": user_name,
        "created_at": start_time.isoformat()
    }

    await log_to_supabase(initial_log_data, request_id)

    try:
        # Classify the message
        classification_result = classify_message_with_gemini(request.WA_Msg_Text or "", gemini_model, request_id)
        classification = classification_result.get("classification", "General|No_Module")
        confidence = classification_result.get("confidence", "LOW")
        reasoning = classification_result.get("reasoning", "No classification provided")
        interested_to_donate = classification_result.get("Interested_To_Donate", "no")
        question_language = classification_result.get("Question_Language", "hi")
        question_script = classification_result.get("Question_Script", "Devanagari")

        # Initialize response variables
        ai_response = "Sorry, I couldn't process your request. Please provide more details."
        image_analysis = None

        # Handle image messages
        if wa_msg_type == "image" and request.WA_Url:
            image_analysis = await analyze_image_with_gemini(request.WA_Url, gemini_model, request_id)
            initial_log_data["image_transcription"] = image_analysis.get("transcription")
            initial_log_data["image_analysis_status"] = image_analysis.get("status")
            initial_log_data["image_analysis_error"] = image_analysis.get("error")
            if image_analysis.get("status") == "success" and image_analysis.get("transcription"):
                donation_result = await process_donation_transcript(
                    image_analysis["transcription"], user_name, gemini_model, request_id
                )
                initial_log_data["donation_analysis"] = donation_result
                if donation_result.get("is_donation_screenshot"):
                    ai_response = donation_result.get("generated_response", "Thank you for your donation! We'll process it soon.")
                    reasoning = "Image-based donation processed"
                    initial_log_data["ai_response"] = ai_response
                    initial_log_data["status"] = "success"
                    await log_to_supabase(initial_log_data, request_id, update_mode=True)
                    await forward_message_to_replica(request.model_dump(exclude_none=True), request_id)
                    return MessageResponse(
                        phone_number=phone_number,
                        ai_response=ai_response,
                        ai_reason=reasoning,
                        WA_Auto_Id=request.WA_Auto_Id,
                        WA_Message_Id=request.WA_Message_Id
                    )

        # Handle specific classifications
        main_classification, sub_classification = classification.split("|") if "|" in classification else (classification, "No_Module")
        if main_classification == "General":
            if sub_classification == "Greeting":
                ai_response = await LLM_reply_greeting(
                    question_script, question_language, request.WA_Msg_Text or "", user_name, gemini_model, wa_msg_type, request_id
                )
            elif sub_classification == "Follow-up":
                ai_response = await LLM_reply_follow_up(
                    question_script, question_language, request.WA_Msg_Text or "", user_name, gemini_model, request_id
                )
            elif sub_classification == "Ok":
                ai_response = await LLM_reply_ok(
                    question_script, question_language, request.WA_Msg_Text or "", user_name, gemini_model, request_id
                )
            else:
                ai_response = f"🙏 Jai Shree Narayan {user_name}! Thank you for contacting Narayan Seva Sansthan. How can I assist you today?"
        elif main_classification in ["Donation Related Enquiries", "Ticket Related Enquiry"] and interested_to_donate == "yes":
            ai_response = f"🙏 Jai Shree Narayan {user_name}! Thank you for your interest in donating. Please share your preferred donation method (e.g., UPI, bank transfer) or visit https://x.ai/donate for details."
        else:
            selected_content_num = llm_select_best_content(request.WA_Msg_Text or "", keywords_summary, gemini_model, request_id)
            selected_content = numbered_content.get(selected_content_num, "No relevant content found.")
            ai_response = generate_faq_response(selected_content, request.WA_Msg_Text or "", gemini_model, request_id)
            reasoning = "FAQ response generated based on selected content"

        # Update log with final data
        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        final_log_data = initial_log_data.copy()
        final_log_data.update({
            "status": "success",
            "processing_end_time": end_time.isoformat(),
            "processing_duration_ms": duration_ms,
            "ai_classification": classification,
            "ai_confidence": confidence,
            "ai_reasoning": reasoning,
            "interested_to_donate": interested_to_donate,
            "question_language": question_language,
            "question_script": question_script,
            "ai_response": ai_response,
            "ai_reason": reasoning,
            "image_transcription": image_analysis.get("transcription") if image_analysis else None,
            "donation_analysis": image_analysis if image_analysis else None,
            "updated_at": end_time.isoformat()
        })

        await log_to_supabase(final_log_data, request_id, update_mode=True)
        await forward_message_to_replica(request.model_dump(exclude_none=True), request_id)

        return MessageResponse(
            phone_number=phone_number,
            ai_response=ai_response,
            ai_reason=reasoning,
            WA_Auto_Id=request.WA_Auto_Id,
            WA_Message_Id=request.WA_Message_Id
        )

    except Exception as e:
        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        error_log_data = initial_log_data.copy()
        error_log_data.update({
            "status": "error",
            "processing_end_time": end_time.isoformat(),
            "processing_duration_ms": duration_ms,
            "error_type": "internal_error",
            "error_message": str(e),
            "updated_at": end_time.isoformat()
        })
        await log_to_supabase(error_log_data, request_id, update_mode=True)
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

#### Step 2: Deploy the Updated Code
1. **Save the File**:
   - Save the code above as `test.py` in your project repository, overwriting the existing file.
   - Use a plain text editor (e.g., VS Code, Notepad++) to avoid introducing curly quotes or other formatting issues.

2. **Commit and Push**:
   ```bash
   git add test.py
   git commit -m "Fix SyntaxError by removing diff marker and consolidating code"
   git push origin main
   ```

3. **Redeploy on Render**:
   - Go to Render’s dashboard > Your service > Trigger a redeploy (via **Manual Deploy** or automatic webhook).
   - Monitor the deployment logs for:
     - `"Starting FastAPI app on port 10000"`
     - `"Using Supabase service key (byp
