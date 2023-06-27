import json
from typing import List, Any

from modules.database.mysql import db


def get_history(pdf_hash: str, user_id: str) -> List[Any]:
    final_res = []
    results = db.ChatPaper.select().order_by(db.ChatPaper.created_at.desc()).where(db.ChatPaper.pdf_hash == pdf_hash,
                                                                                   db.ChatPaper.user_id == user_id)
    for res in results:
        final_res.append({"sender": 'user', "content": res.query})
        final_res.append({"sender": 'bot', "content": res.content})
    return final_res


def add_history(pdf_hash: str, user_id: str, query: str, content: str,
                token_cost: int, pages):
    db.ChatPaper.create(pdf_hash=pdf_hash, user_id=user_id, query=query, content=content, token_cost=token_cost,
                        pages=json.dumps(pages))
