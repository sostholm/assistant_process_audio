import psycopg
import os
from dataclasses import dataclass
from typing import List, Optional

POSTGRES_USER     = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "assistant_testing")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "192.168.0.218")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
# Connect to your PostgreSQL database
conn = psycopg.connect(
    dbname=POSTGRES_DATABASE,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT
)

@dataclass
class UserVoiceRecognition:
    user_id: str
    nick_name: Optional[str]
    voice_recognition: List[bytes]

def get_users_voice_recognition() -> List[UserVoiceRecognition]:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            u.user_id,
            u.nick_name AS nick_name,
            vr.voice_recognition AS voice_data
        FROM users u
        LEFT JOIN voice_recognition vr ON u.user_id = vr.user_id;
    """)
    rows = cursor.fetchall()

    cursor.execute("""
        SELECT
            ai.ai_id,
            ai.ai_name,
            vr.voice_recognition AS voice_data
        FROM ai 
        LEFT JOIN voice_recognition vr ON ai.ai_id = vr.ai_id;""")
    
    rows += cursor.fetchall()
    cursor.close()

    results = []
    for row in rows:
        user_id, nick_name, voice_data = row
        if voice_data is None:
            continue
        results.append(UserVoiceRecognition(
            user_id=user_id,
            nick_name=nick_name,
            voice_recognition=voice_data
        ))
    return results
