import logging
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
import uvicorn
import httpx
from dotenv import dotenv_values
from pydantic import BaseModel
from typing import Optional
import uuid
import redis
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


env_values = dotenv_values(".env")
TORCHSERVE_ADDRESS = env_values.get("TORCHSERVE_ADDRESS")
REDIS_HOST = env_values.get("REDIS_HOST")
REDIS_PORT = env_values.get("REDIS_PORT")
URL_INFERENCE = env_values.get("URL_INFERENCE")
MAIL_SERVER = env_values.get("MAIL_SERVER")
SMTP_PORT = env_values.get("SMTP_PORT")
CONNECTION_PORT = env_values.get("CONNECTION_PORT")
EMAIL = env_values.get("EMAIL")
LOGIN = env_values.get("LOGIN")
PASSWORD = env_values.get("PASSWORD")

logging.basicConfig(level=logging.INFO, filename='logs/app.log', filemode='a', format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
try:
    response = redis_client.ping()
    if response:
        print("Successfully connected to Redis")
    else:
        raise HTTPException(status_code=500, detail="Failed to connect to Redis")
except redis.ConnectionError as e:
    raise HTTPException(status_code=500, detail=f"Redis connection error: {e}")


MESSAGE_STREAM_RETRY_TIMEOUT = 30000
app = FastAPI()

# add CORS so our web page can connect to our api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    seq_light: str
    seq_heavy: str
    seq_rbd: str
    selectivity: bool
    email: Optional[str] = None


def validate_sequence(sequence: str):
    if not sequence:
        raise HTTPException(status_code=400, detail="Sequence cannot be empty")
    if not all(char.isalpha() and char.isupper() for char in sequence):
        raise HTTPException(status_code=400, detail="Sequence must contain only uppercase letters A-Z")


def send_email(to_email: str, inference_uuid: str):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = to_email
        msg['Subject'] = 'Inference Result'

        msg.attach(MIMEText(
            f"Your job {inference_uuid} has finished.\n\n"
            "Details should be available at:\n\n"
            f"{URL_INFERENCE}{inference_uuid}\n\n"
            "-- RBD-AIM server queue\n\n"
            "Please note: The result will be available via this link for the next 24 hours.",
            'plain'
        ))

        server = smtplib.SMTP(MAIL_SERVER, SMTP_PORT)
        server.connect(MAIL_SERVER, CONNECTION_PORT)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(LOGIN, PASSWORD)
        result = server.sendmail(EMAIL, to_email, msg.as_string())
        server.quit()

        if not result:
            logger.info(f"Email sent successfully to: {to_email}")
        else:
            for recipient, (code, resp) in result.items():
                logger.info("Failed to send email to %s: %s (code %d)", recipient, resp, code)
    except Exception as e:
        logger.info(f"Error sending email to: {to_email}: {e}")


async def torchserve_inference(inference_uuid: str, seq_light: str, seq_heavy: str, seq_rbd: str, selectivity: bool, user_email=None):
    try:
        logger.info(f"{inference_uuid}: Start inference")

        async with (httpx.AsyncClient() as client):
            response = await client.post(f"{TORCHSERVE_ADDRESS}/predictions/OpenFold", timeout=None,
                json={"seq_light": seq_light, "seq_heavy": seq_heavy, "seq_rbd": seq_rbd, "selectivity": selectivity})

            logger.info(f"{inference_uuid}: Torchserve response {response.status_code}")

            if response.status_code == 200:
                redis_client.set(inference_uuid, json.dumps(response.json()), ex=86400)
                redis_client.set(f"{inference_uuid}_status", "completed", ex=86400)

                logger.info(f"{inference_uuid}: user email {user_email}")
                if user_email:
                    send_email(user_email, inference_uuid)

                logger.info(f"{inference_uuid}: return inference result")
            else:
                logger.info(f"{inference_uuid}: response error: {response.text}")
                redis_client.set(f"{inference_uuid}_status", "failed", ex=86400)
    except httpx.RequestError as exc:
        logger.info(f"{inference_uuid}: inference request error: {exc}")
        redis_client.set(f"{inference_uuid}_status", "failed", ex=86400)


async def inference_sse(inference_uuid: str):
    try:
        timeout = 5400  # 1 час 30 минут
        check_interval = 5  # секунды
        elapsed_time = 0

        while elapsed_time < timeout:
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval

            status = redis_client.get(f"{inference_uuid}_status")

            if status:
                yield {
                    "event": "status",
                    "id": inference_uuid,
                    "retry": 3000,
                    "data": {"status": status.decode()},
                }

                if status == b"completed":
                    result = redis_client.get(inference_uuid)
                    if result:
                        yield {
                            "event": "result",
                            "id": inference_uuid,
                            "retry": 3000,
                            "data": json.loads(result),
                        }
                    break
                elif status == b"failed":
                    error_message = redis_client.get(inference_uuid)
                    yield {
                        "event": "error",
                        "id": inference_uuid,
                        "retry": 3000,
                        "data": json.loads(error_message),
                    }
                    break
            else:
                logger.info(f"{inference_uuid}: Status not found")
                yield {
                    "event": "error",
                    "id": inference_uuid,
                    "retry": 3000,
                    "data": "status not found",
                }
                break

            if elapsed_time >= timeout:
                yield {
                    "event": "timeout",
                    "id": inference_uuid,
                    "retry": 3000,
                    "data": {"message": "The task did not complete in the expected time"},
                }
                logger.info(f"{inference_uuid}: the task did not complete in the expected time")
                break
    except httpx.RequestError as exc:
        logger.info(f"{inference_uuid}: inference request error: {exc}")
        yield None


@app.post("/inference")
async def post_inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    validate_sequence(request.seq_light)
    validate_sequence(request.seq_heavy)
    validate_sequence(request.seq_rbd)

    inference_uuid = str(uuid.uuid4())
    redis_client.set(f"{inference_uuid}_status", "processing", ex=86400)
    background_tasks.add_task(torchserve_inference, inference_uuid, request.seq_light, request.seq_heavy, request.seq_rbd, request.selectivity, request.email)

    return {"inference_uuid": inference_uuid}


@app.get("/inference/{light}/{heavy}/{rbd}/selectivity={selectivity}")
async def get_inference(background_tasks: BackgroundTasks, light: str, heavy: str, rbd: str, selectivity: bool, email=None):
    validate_sequence(light)
    validate_sequence(heavy)
    validate_sequence(rbd)

    inference_uuid = str(uuid.uuid4())
    redis_client.set(f"{inference_uuid}_status", "processing", ex=86400)
    background_tasks.add_task(torchserve_inference, inference_uuid, light, heavy, rbd, selectivity, email)

    return {"inference_uuid": inference_uuid}


@app.get("/inference/{inference_uuid}/status")
async def get_inference_status(inference_uuid: str):
    return EventSourceResponse(inference_sse(inference_uuid))


@app.get("/inference/{inference_uuid}")
async def load_inference(inference_uuid: str):
    logger.info(f"Try to load inference result: {inference_uuid}")
    result = redis_client.get(inference_uuid)

    if result:
        logger.info(f"inference result {inference_uuid} found")
        return {"data": json.loads(result)}
    else:
        logger.info(f"inference result {inference_uuid} not found")
        raise HTTPException(status_code=404, detail="Result not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
