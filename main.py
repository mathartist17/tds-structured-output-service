"""
ELI15 INVOICE EXTRACTOR
=======================

Think of this program like a helpful robot that:
1. Listens on the internet for invoice text
2. Reads the messy text using an AI brain (LLM)
3. Pulls out the important bits (vendor, amount, currency, date)
4. Returns clean, organized data

Let's build it step by step!
"""

# Step 1: Import the tools we need
# ================================
from fastapi import FastAPI, HTTPException  # FastAPI = web server builder
from pydantic import BaseModel, Field       # Pydantic = data validator (keeps data clean)
import re                                    # re = find patterns in text
from datetime import datetime                # Parse and work with dates
import httpx                                 # Send requests to local LLM
import json                                  # Work with JSON data
from typing import Optional                  # Type hints for clarity

# Step 2: Define the input and output shapes
# ===========================================
# Think of these like templates - they define what data looks like

class InvoiceRequest(BaseModel):
    """What the user sends us - just invoice text"""
    text: str  # A string containing the invoice

class InvoiceResponse(BaseModel):
    """What we send back - structured, clean data"""
    vendor: str           # Company name (e.g., "Acme Corp")
    amount: float         # Price (e.g., 99.50)
    currency: str         # Currency code (e.g., "USD")
    date: str            # Due date in YYYY-MM-DD format

# Step 3: Create the FastAPI app
# ==============================
app = FastAPI(title="Invoice Extractor", version="1.0")

# Step 4: Build the LLM prompt and extraction logic
# ==================================================
def extract_invoice_fields(invoice_text: str) -> dict:
    """
    The brain of the operation!
    
    What it does:
    1. Sends the invoice text to a local LLM (Ollama)
    2. Asks it to extract the 4 fields
    3. Parses the response into clean data
    """
    
    # The "instruction" we give the LLM
    # We're asking it to be a strict accountant - extract ONLY the 4 fields
    prompt = f"""Extract the invoice fields from this text. Return ONLY valid JSON with these 4 fields:
- vendor (company name)
- amount (total due as number)
- currency (3-letter code like USD, EUR, GBP)
- date (YYYY-MM-DD format)

If a field is missing, make your best guess or use null. Do NOT include any other text.

Invoice text:
{invoice_text}

Return ONLY the JSON object, no other text:
"""

    try:
        # OPTION A: Using Ollama (local LLM) - if you have it installed
        # ============================================================
        # Uncomment this section if you have Ollama running locally
        # (Download from https://ollama.ai)
        
        response = httpx.post(
            "https://parrot-postcard-mooned.ngrok-free.dev/api/generate",  # Ollama's address
            json={
                "model": "mistral",                   # Or llama2, neural-chat, etc.
                "prompt": prompt,
                "stream": False,
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_text = result.get("response", "")
        else:
            raise ConnectionError("Could not reach Ollama")
    
    except Exception as ollama_error:
        print(f"Ollama unavailable, using fallback. Error: {ollama_error}")
        
        # OPTION B: Fallback - extract using regex patterns
        # ================================================
        # If no LLM available, we manually look for patterns in the text
        # This is like having a backup accountant who uses a checklist
        
        llm_text = json.dumps({
            "vendor": extract_vendor_fallback(invoice_text),
            "amount": extract_amount_fallback(invoice_text),
            "currency": extract_currency_fallback(invoice_text),
            "date": extract_date_fallback(invoice_text),
        })
    
    # Step 5: Parse the LLM's JSON response
    # =====================================
    try:
        # Find JSON in the response (LLM might add extra text)
        json_match = re.search(r'\{.*\}', llm_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            extracted = json.loads(json_str)
        else:
            raise ValueError("No JSON found in LLM response")
        
        # Step 6: Clean and validate each field
        # ======================================
        
        # Vendor: just take it as-is (string)
        vendor = str(extracted.get("vendor", "Unknown Vendor")).strip()
        
        # Amount: convert to float
        amount = extracted.get("amount", 0)
        if isinstance(amount, str):
            amount = float(re.sub(r'[^\d.]', '', amount))
        amount = float(amount)
        
        # Currency: uppercase 3-letter code
        currency = str(extracted.get("currency", "USD")).strip().upper()
        if len(currency) != 3:
            currency = "USD"  # fallback
        
        # Date: ensure YYYY-MM-DD format
        date_str = str(extracted.get("date", ""))
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', date_str)
        if date_match:
            date = date_match.group()
        else:
            date = datetime.now().strftime("%Y-%m-%d")  # fallback to today
        
        return {
            "vendor": vendor,
            "amount": amount,
            "currency": currency,
            "date": date,
        }
    
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"Parse error: {e}. LLM response was: {llm_text}")
        # Return best-effort defaults
        return {
            "vendor": extract_vendor_fallback(invoice_text),
            "amount": extract_amount_fallback(invoice_text),
            "currency": "USD",
            "date": datetime.now().strftime("%Y-%m-%d"),
        }


# Step 7: Fallback pattern matching (for when LLM isn't available)
# ================================================================

def extract_vendor_fallback(text: str) -> str:
    """Look for company names in the invoice text"""
    # Common patterns: "Bill from X" or "Vendor: X" or capitalized words
    patterns = [
        r'from\s+([A-Z][A-Za-z\s&.-]+?)(?:\s+Ltd|\s+Corp|\s+Inc|;|$)',
        r'vendor\s*[:=]\s*([A-Za-z\s&.-]+?)(?:;|$)',
        r'^([A-Z][A-Za-z\s&.-]+?)(?:;|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "Unknown Vendor"


def extract_amount_fallback(text: str) -> float:
    """Look for currency amounts like $50.00 or 100 USD"""
    # Search for numbers that look like prices
    amount_patterns = [
        r'(?:total|amount|due|price|invoice)\s*[:=]?\s*\$?([0-9]+\.?[0-9]*)',
        r'\$\s?([0-9]+\.?[0-9]*)',
        r'([0-9]+\.?[0-9]*)\s*(?:USD|EUR|GBP)',
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return 0.0


def extract_currency_fallback(text: str) -> str:
    """Look for currency codes like USD, EUR, GBP"""
    match = re.search(r'\b(USD|EUR|GBP|CAD|AUD|JPY)\b', text, re.IGNORECASE)
    return match.group(1).upper() if match else "USD"


def extract_date_fallback(text: str) -> str:
    """Look for dates in YYYY-MM-DD format"""
    match = re.search(r'\d{4}-\d{2}-\d{2}', text)
    if match:
        return match.group()
    # Try other date formats and convert
    # This is a simplified version
    return datetime.now().strftime("%Y-%m-%d")


# Step 8: Create the actual endpoint
# ===================================
@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(request: InvoiceRequest) -> InvoiceResponse:
    """
    The main function that receives requests from the internet.
    
    When someone POST /extract with invoice text, this function:
    1. Checks the input isn't empty
    2. Calls our extraction logic
    3. Validates the output
    4. Sends back clean JSON
    """
    
    # Safety check: don't accept empty text
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Invoice text cannot be empty"
        )
    
    # Call our extraction function
    try:
        fields = extract_invoice_fields(request.text)
    except Exception as e:
        # If something breaks, return 422 (validation error) not 500 (server crash)
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract invoice fields: {str(e)}"
        )
    
    # Create response (Pydantic validates automatically)
    return InvoiceResponse(
        vendor=fields["vendor"],
        amount=fields["amount"],
        currency=fields["currency"],
        date=fields["date"],
    )


# Step 9: Health check endpoint (for testing)
# ============================================
@app.get("/health")
async def health():
    """Check if the server is alive"""
    return {"status": "alive", "service": "invoice_extractor"}
