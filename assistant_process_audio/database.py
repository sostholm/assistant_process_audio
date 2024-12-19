import psycopg
import os
from dataclasses import dataclass
from typing import List, Optional

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
# Connect to your PostgreSQL database
conn = psycopg.connect(
    dbname="assistant_v2",
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host="192.168.0.218",
    port="5432"
)


@dataclass
class UserVoiceRecognition:
    user_id: str
    name: str
    nick_name: Optional[str]
    voice_recognition: List[bytes]

def get_users_voice_recognition() -> List[UserVoiceRecognition]:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            u.user_id,
            CASE WHEN u.user_profile_id IS NOT NULL 
                THEN up.full_name
                ELSE a.ai_name
            END AS name,
            CASE WHEN u.user_profile_id IS NOT NULL 
                THEN up.nick_name
                ELSE a.ai_name
            END AS nick_name,
            array_agg(vr.voice_recognition) AS voice_data
        FROM users u
        LEFT JOIN user_profile up ON u.user_profile_id = up.user_profile_id
        LEFT JOIN ai a ON u.ai_profile_id = a.ai_id
        LEFT JOIN voice_recognition vr ON u.user_id = vr.user_id
        GROUP BY u.user_id, up.full_name, up.nick_name, a.ai_name;
    """)
    rows = cursor.fetchall()

    # Each row: (user_id, name, nick_name, voice_data)
    # voice_data may be None if there are no recordings.
    results = []
    for row in rows:
        user_id, name, nick_name, voice_data = row
        if voice_data is None:
            voice_data = []
        results.append(UserVoiceRecognition(
            user_id=user_id,
            name=name,
            nick_name=nick_name,
            voice_recognition=voice_data
        ))

    return results