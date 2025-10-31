import os
import json
import requests
import re
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
# .memory import ConversationBufferMemory

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

grok_api_key = os.getenv("GROQ_API_KEY") 

required_fields = [
    "year",
    "developer_name",
    "anniversary_start",
    "anniversary_end",
    "coupon_code",
    "points",
]

data_store: Dict[str, str] = {field: None for field in required_fields}
confirmation_received = False
memory = ConversationBufferMemory(return_messages=True)

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=grok_api_key,
)

def is_form_complete(data: Dict[str, str]) -> bool:
    return all(data.get(field) not in [None, ""] for field in required_fields)

def parse_llm_response(llm_response: str) -> Dict[str, str]:
    try:
        # 1️⃣ Try to match properly formatted code block: ```json\n{...}\n```
        match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', llm_response, re.IGNORECASE)
        if not match:
            # 2️⃣ Fallback: match if the bot just printed "json\n{...}" without backticks
            match = re.search(r'json\s*({[\s\S]*?})', llm_response, re.IGNORECASE)

        if match:
            extracted_json = match.group(1)
            parsed = json.loads(extracted_json)
            return {k: v for k, v in parsed.items() if k in required_fields}
        else:
            print("⚠️ Still no JSON block found in LLM response.")
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")
    return {}




@app.post("/chat")
async def chat(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    global data_store, confirmation_received
    body = await request.json()
    user_input = body.get("message", "")
    developer_list = body.get("dev_names", [])
    user_info = body.get("user_info", {})

    if not isinstance(developer_list, list):
        return {
            "message": "❌ 'dev_names' must be a list of developer names.",
            "data": None,
            "complete": False,
        }

    if user_input.strip().lower() in ["yes", "confirm", "submit"] and is_form_complete(data_store):
        confirmation_received = True
        payload = {
            'year': str(data_store['year']),
            'developer_name': str(data_store['developer_name']),
            'userdetails[anniversary_start_date][0]': str(data_store['anniversary_start']),
            'userdetails[anniversary_end_date][0]': str(data_store['anniversary_end']),
            'referral_code': str(data_store['coupon_code']),
            'points': str(data_store['points']),
            'userdetails[anniversary_point][0]': '20',
            'name': user_info.get('user_name', ''),
            'email': user_info.get('user_email', ''),
            'mobile': user_info.get('user_phone', ''),
            'amount': '500',
            'userid': str(user_info.get('user_id', '')),
            'offer_status': '2',
            'status': '1',
            'created_by': '1',
        }
        try:
            response = requests.post(
                "http://35.155.124.107/api/developercontracts/add-newcontract",
                data=payload,
                headers={'Authorization': authorization or ''}
            )
            res = response.json()
            if res.get('status') == True:
                memory.chat_memory.add_ai_message("✅ Estimate submitted.")
                data_store = {field: None for field in required_fields}
                confirmation_received = False
                memory.clear()
                return {
                    "message": "All fields collected! Submitted the estimate offer.",
                    "data": res,
                    "complete": True,
                }
        except Exception as e:
            return {
                "message": f"❌ API call failed: {str(e)}",
                "data": None,
                "complete": False,
            }

    elif user_input.strip().lower() in ["no", "cancel"]:
        confirmation_received = False
        data_store = {field: None for field in required_fields}
        #memory.chat_memory.add_ai_message("Submission cancelled. You can modify the values.")
        memory.clear()
        return {
            "message": "Submission cancelled. You can modify the values.",
            "data": None,
            "complete": False,
        }

       system_prompt = (
    "Always greet the user in start."
    "You're a helpful assistant responsible for chatting and collecting the contract info from the user. "
    "Collect only the following fields:\n"
    "Be a conversational bot and directly ask these below fields in order by asking through sentences and do not specify the format in brackets."
    "Dont validate the instructions everytime in chat,just  directly respond to user query or address the next step."
    "- year (number)\n"
    "- developer_name (string)\n"
    "- anniversary_start (YYYY-MM-DD)\n"
    "- anniversary_end (YYYY-MM-DD)\n"
    "- coupon_code (string)\n"
    "- points (number)\n\n"
    "Ask only for the missing fields."
    "If the user does'nt provide coupon code, then tell the user that it is necessary to proceed otherwise your contract will not be submitted "
    f" If asking for developer_name, suggest one from this list: {', '.join([dev for dev in developer_list if dev]) if developer_list else ''}. "
    "Respond naturally to the user. At the end of your message, include only the fields you collected "
    "in this turn inside a JSON block wrapped in triple backticks like this:\n```json\n{{ ... }}\n```"
    "Note- Don't provide the json in chat everytime ."
    "This block is for internal use only — do not mention or explain it to the user."
)



    messages = memory.chat_memory.messages + [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    memory.chat_memory.add_user_message(user_input)

    bot_reply = llm(messages).content
    print('bot_reply', bot_reply)
    extracted = parse_llm_response(bot_reply)

    for key, value in extracted.items():
        if key == "developer_name":
            matched = next((d for d in developer_list if d.lower() == value.lower()), None)
            if matched:
                data_store[key] = matched
            else:
                return {
                    "message": f"'{value}' is not a valid developer. Choose from: {', '.join(developer_list)}",
                    "data": None,
                    "complete": False,
                }
        else:
            data_store[key] = value
    print('data_store', data_store)
    if is_form_complete(data_store) and not confirmation_received:
        summary = "\n".join([f"- {k}: {v}" for k, v in data_store.items()])
        memory.chat_memory.add_ai_message(f"📋 All fields collected:\n{summary}\n\nDo you want to submit this? (Yes/No)")
        return {
            "message": f"All fields collected:\n{summary}\n\n Please type (Yes/No) to submit.\n",
            "data": None,
            "complete": False,
        }

    missing_fields = [f for f in required_fields if data_store[f] in [None, ""]]
    user_friendly_reply = re.sub(r'(?:```(?:json)?|json)?\s*{[\s\S]*?}\s*```?', '', bot_reply, flags=re.IGNORECASE).strip()

    memory.chat_memory.add_ai_message(bot_reply)
    return {
        "message": user_friendly_reply,
        "data": None,
        "complete": False,

    }


