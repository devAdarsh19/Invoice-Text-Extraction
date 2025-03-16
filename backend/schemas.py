from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str
    
class TokenData(BaseModel):
    username: str
    
class User(BaseModel):
    username: str 
    email: str | None = None
    hashed_password: str | None = None
    disabled: bool | None = None
    
class UserInDb(User):
    hashed_password: str




