from pydantic import BaseModel

# Define Request Schema
class UserRequest(BaseModel):
    username: str
    password: str

# Define Response Schema
class UserResponse(BaseModel):
    user_id: int
    username: str
    created_at: str

# Example of a nested schema for more complex responses
class PostResponse(BaseModel):
    post_id: int
    title: str
    content: str
    author: UserResponse