```python
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from loguru import logger
import sqlite3
import json
import supabase
from dotenv import load_dotenv
import google.generativeai as genai
import httpx

# Load environment variables
load_dotenv()

# Configure logger
logger.add("logs/app.log", rotation="500 MB", retention="10 days", level=os.getenv("LOG_LEVEL", "INFO"))

# Initialize FastAPI app
app = FastAPI(title="WhatsApp Message Processor", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Using Supabase service key (bypasses RLS)")
    except Exception as e:
        logger.error(f"Supabase initialization failed: {e}")

# Initialize Gemini AI client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel("gemini-1.5-pro")
        logger.info("Gemini client and model initialized successfully")
    except Exception as e:
        logger.error(f"Gemini AI initialization failed: {e}")

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
        return numbered_data, numbered_keywords
    except Exception as e:
        logger.error(f"Failed to fetch FAQ data: {e}")
        return {1: "Default FAQ content"}, {1: "default keywords"}

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

def serialize_datetime_recursive(obj):
    if isinstance(obj, dict):
        return {k: serialize_datetime_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_recursive(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

# Gemini classification (customize with your FEW_SHOT_EXAMPLES)
async def classify_message_with_gemini(message: str, is_image: bool = False) -> Dict[str, Any]:
    try:
        if not gemini_client:
            return {
                "classification": "General|Unknown",
                "confidence": "LOW",
                "reasoning": "Gemini client not initialized",
                "Interested_To_Donate": "No",
                "Question_Language": "English",
                "Question_Script": "Latin"
            }
        
        # Replace with your actual FEW_SHOT_EXAMPLES and classification logic
        prompt = f"""
        Classify the following message into one of these categories:
        - Donation Related Enquiries (Announce Related, Receipts Related, Amount Confirmation, Donation Payment Information, etc.)
        - General (Greeting, Follow-up, Emoji, Thanks, etc.)
        - General Information Enquiries (About Sansthan, Katha Related, etc.)
        - Medical / Treatment Enquiries
        - Community Outreach Enquiries
        - Fundraising Campaign Enquiries
        - Beneficiary Support Enquiries
        - Ticket Related Enquiry
        - SPAM

        Message: {message}
        """
        response = gemini_client.generate_content(prompt)
        # Mock response (replace with actual parsing)
        return {
            "classification": "General|Greeting" if "jai" in message.lower() else "Donation Related Enquiries|Donation Payment Information",
            "confidence": "HIGH",
            "reasoning": f"Message contains {'greeting' if 'jai' in message.lower() else 'donation intent'}",
            "Interested_To_Donate": "Yes" if "donate" in message.lower() else "No",
            "Question_Language": "English",
            "Question_Script": "Latin"
        }
    except Exception as e:
        logger.error(f"Gemini classification failed: {e}")
        return {
            "classification": "General|Unknown",
            "confidence": "LOW",
            "reasoning": f"Classification error: {str(e)}",
            "Interested_To_Donate": "No",
            "Question_Language": "English",
            "Question_Script": "Latin"
        }

# Health endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "WhatsApp Message Processor",
        "version": "1.0.0"
    }

# Message endpoint
@app.post("/message", response_model=MessageResponse, tags=["Message Processing"])
async def handle_message(request: MessageRequest):
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    phone_number = request.MobileNo or request.WA_Msg_To or "Unknown"
    user_name = request.Donor_Name or request.Wa_Name or "User"
    wa_msg_type = request.WA_Msg_Type or "text"

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

    # Log initial data
    await log_to_supabase(initial_log_data, request_id)

    try:
        # Initialize default classification result
        classification_result = {
            "classification": "General|Unknown",
            "confidence": "LOW",
            "reasoning": "No classification performed",
            "Interested_To_Donate": "No",
            "Question_Language": "English",
            "Question_Script": "Latin"
        }
        ai_response = "Sorry, I couldn't process your request. Please provide more details."
        reasoning = "Default response due to invalid input or processing error"
        image_analysis = None

        # Validate and process message
        if wa_msg_type.lower() == "text" and request.WA_Msg_Text:
            classification_result = await classify_message_with_gemini(request.WA_Msg_Text)
            classification = classification_result.get("classification", "General|Unknown")
            reasoning = classification_result.get("reasoning", "No reasoning provided")

            if classification.startswith("General|Greeting"):
                ai_response = f"Jai Shree Ram {user_name} ji, mai Priya hu, aapki kaise sahayta kar sakti hu? 🙏"
            elif classification.startswith("Donation Related Enquiries"):
                ai_response = f"🙏 Jai Shree Narayan {user_name}! Thank you for your interest in donating. Please share your preferred donation method (e.g., UPI, bank transfer) or visit https://x.ai/donate for details."
            else:
                # FAQ lookup
                numbered_data, numbered_keywords = fetch_numbered_data()
                ai_response = numbered_data.get(1, "Sorry, our FAQ service is temporarily unavailable.")
                reasoning = "FAQ lookup performed"

        elif wa_msg_type.lower() == "image" and request.WA_Url:
            image_analysis = {"transcription": "Donation of ₹1000", "status": "success"}  # Replace with actual image analysis
            classification_result = await classify_message_with_gemini(image_analysis["transcription"], is_image=True)
            reasoning = classification_result.get("reasoning", "Image-based donation")
            ai_response = f"जय नारायण {user_name} जी! ₹1000 के आपके दान के लिए हार्दिक धन्यवाद। आपकी रसीद जल्द ही भेजी जाएगी।"

        else:
            reasoning = "Invalid message type or missing text/URL"
            classification_result["reasoning"] = reasoning

        # Update log with final data
        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        final_log_data = initial_log_data.copy()
        final_log_data.update({
            "status": "success",
            "processing_end_time": end_time.isoformat(),
            "processing_duration_ms": duration_ms,
            "ai_classification": classification_result.get("classification"),
            "ai_confidence": classification_result.get("confidence"),
            "ai_reasoning": classification_result.get("reasoning"),
            "interested_to_donate": classification_result.get("Interested_To_Donate"),
            "question_language": classification_result.get("Question_Language"),
            "question_script": classification_result.get("Question_Script"),
            "ai_response": ai_response,
            "ai_reason": reasoning,
            "image_transcription": image_analysis.get("transcription") if image_analysis else None,
            "donation_analysis": image_analysis if image_analysis else None,
            "updated_at": end_time.isoformat()
        })

        await log_to_supabase(final_log_data, request_id, update_mode=True)

        return MessageResponse(
            phone_number=phone_number,
            ai_response=ai_response,
            ai_reason=reasoning,
            WA_Auto_Id=request.WA_Auto_Id,
            WA_Message_Id=request.WA_Message_Id
        )

    except Exception as e:
        # Log error
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

#### Step 3: Deploy the Updated Code
1. **Save the File**:
   - Save the code above as `test.py` in your project repository.
   - Ensure no stray characters (e.g., `curl` commands or JSON snippets) are included in the file.

2. **Commit and Push**:
   ```bash
   git add test.py
   git commit -m "Fix SyntaxError and ensure proper Supabase logging"
   git push origin main
   ```

3. **Redeploy on Render**:
   - Go to Render’s dashboard > Your service > Trigger a redeploy (e.g., via **Manual Deploy** or automatic webhook).
   - Monitor the deployment logs for success or errors.

4. **Verify Deployment**:
   - Check Render logs for:
     - `"Starting FastAPI app on port 10000"`
     - `"Using Supabase service key (bypasses RLS)"`
     - No `SyntaxError` or import errors.

#### Step 4: Test with Postman
1. **Send the Corrected Request**:
   - Use the `curl` command you provided, but fix `WA_Msg_Type`:
     ```bash
     curl -X 'POST' \
       'https://fastapi-bot-rosu.onrender.com/message' \
       -H 'accept: application/json' \
       -H 'Content-Type: application/json' \
       -d '{
         "WA_Auto_Id": 0,
         "WA_In_Out": "IN",
         "Account_Code": 0,
         "WA_Received_At": "2025-09-24T17:18:58.652Z",
         "NGCode": 0,
         "Wa_Name": "Aadhira",
         "MobileNo": "9388012299",
         "WA_Msg_To": "9388012299",
         "WA_Msg_Text": "How to send 25k",
         "WA_Msg_Type": "text",
         "Integration_Type": "whatsapp",
         "WA_Message_Id": "msg_123456",
         "WA_Url": null,
         "Status": "success",
         "Donor_Name": "Aadhira"
       }'
     ```
   - **Postman Equivalent**:
     - Method: POST
     - URL: `https://fastapi-bot-rosu.onrender.com/message`
     - Headers: `Content-Type: application/json`, `accept: application/json`
     - Body (raw JSON):
       ```json
       {
         "WA_Auto_Id": 0,
         "WA_In_Out": "IN",
         "Account_Code": 0,
         "WA_Received_At": "2025-09-24T17:18:58.652Z",
         "NGCode": 0,
         "Wa_Name": "Aadhira",
         "MobileNo": "9388012299",
         "WA_Msg_To": "9388012299",
         "WA_Msg_Text": "How to send 25k",
         "WA_Msg_Type": "text",
         "Integration_Type": "whatsapp",
         "WA_Message_Id": "msg_123456",
         "WA_Url": null,
         "Status": "success",
         "Donor_Name": "Aadhira"
       }
       ```

2. **Expected Response**:
   ```json
   {
     "phone_number": "9388012299",
     "ai_response": "🙏 Jai Shree Narayan Aadhira! Thank you for your interest in donating. Please share your preferred donation method (e.g., UPI, bank transfer) or visit https://x.ai/donate for details.",
     "ai_reason": "Message contains donation intent",
     "WA_Auto_Id": 0,
     "WA_Message_Id": "msg_123456"
   }
   ```

3. **Check Supabase Logs**:
   - Go to Supabase > Table Editor > `message_logs`.
   - Query recent entries:
     ```sql
     SELECT 
         request_id, status, wa_msg_text, ai_response, classification, 
         confidence, reasoning, processing_duration_ms, created_at
     FROM message_logs 
     WHERE created_at > NOW() - INTERVAL '10 minutes'
     ORDER BY created_at DESC;
     ```
   - Expected entry:
     - `wa_msg_text`: `"How to send 25k"`
     - `ai_response`: Donation response text
     - `classification`: `"Donation Related Enquiries|Donation Payment Information"`
     - `status`: `"success"`
     - `mobile_no`: `"9388012299"`
     - `donor_name`: `"Aadhira"`

4. **Monitor Render Logs**:
   - Check for:
     - `"Successfully inserted Supabase log entry"`
     - `"Successfully updated Supabase log entry"`
     - Any `"Supabase log failed"` errors.

#### Step 5: Update Supabase Schema (if Needed)
If you haven’t applied the schema update yet, run this in Supabase’s SQL Editor:
```sql
ALTER TABLE message_logs 
ADD COLUMN IF NOT EXISTS wa_msg_text TEXT,
ADD COLUMN IF NOT EXISTS ai_response TEXT,
ADD COLUMN IF NOT EXISTS ai_reason TEXT,
ADD COLUMN IF NOT EXISTS classification TEXT,
ADD COLUMN IF NOT EXISTS confidence TEXT,
ADD COLUMN IF NOT EXISTS reasoning TEXT,
ADD COLUMN IF NOT EXISTS interested_to_donate TEXT,
ADD COLUMN IF NOT EXISTS question_language TEXT,
ADD COLUMN IF NOT EXISTS question_script TEXT,
ADD COLUMN IF NOT EXISTS donor_name TEXT,
ADD COLUMN IF NOT EXISTS mobile_no TEXT,
ADD COLUMN IF NOT EXISTS wa_msg_type TEXT,
ADD COLUMN IF NOT EXISTS image_transcription TEXT,
ADD COLUMN IF NOT EXISTS donation_analysis JSONB,
ADD COLUMN IF NOT EXISTS processing_end_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS processing_duration_ms INTEGER,
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE message_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow service key inserts" ON message_logs FOR ALL USING (true);
```

#### Step 6: Handle FAQ Database (Optional)
The logs previously indicated `extracted_data.db` is missing, which affects FAQ responses. To fix:
1. Create a local SQLite database:
   ```sql
   CREATE TABLE extracted_data (
       keywords TEXT,
       content TEXT
   );
   INSERT INTO extracted_data (keywords, content) VALUES
   ('donation purpose', 'Your donation supports our programs for the specially-abled.'),
   ('amavasya campaign', 'The Amavasya campaign provides food to the needy.');
   ```
2. Save as `extracted_data.db` and upload to your Render project’s root directory.
3. Redeploy to include the file.

#### Troubleshooting
1. **SyntaxError Persists**:
   - If the error remains, share the contents of `test.py` around line 357 (e.g., lines 350–360) or the full file if possible.
   - Check for unclosed quotes, stray JSON, or incorrect indentation.

2. **Supabase Logging Issues**:
   - If logs don’t appear, verify `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in Render’s Environment settings.
   - Check Render logs for `"Supabase log failed"`.

3. **Gemini Classification**:
   - The `classify_message_with_gemini` function is a placeholder. If you have the original `FEW_SHOT_EXAMPLES` or classification logic, share it to integrate properly.

4. **Render Cold Start**:
   - If requests time out, retry after a few seconds (Render free tiers may have delays).

If the deployment fails again or you get new errors, share the Render logs or Postman response, and I’ll help pinpoint the issue. Once deployed, the application should handle your `curl` request correctly and log to Supabase with all fields!
