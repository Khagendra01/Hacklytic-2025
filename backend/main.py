from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mongodb_set import get_set_user
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/get_user/{user_id}")
def auth_user(user_id: int):
    return get_set_user(user_id)

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
