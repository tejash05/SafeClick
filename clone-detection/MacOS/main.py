from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from website_analysis import check_clone

# Create FastAPI app
app = FastAPI()

# Health Check Route (Avoid 404)
@app.get("/")
def home():
    return {"message": "SafeClick API is running!"}

# Request body schema
class CompareRequest(BaseModel):
    site1: str
    site2: str

# Ensure "/compare" Route Exists
@app.post("/compare")
async def compare_sites(request: CompareRequest):
    """Endpoint to compare two websites."""
    try:
        result = await check_clone(request.site1, request.site2)
        return result
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}